"""Seven measured, three-epoch source ablations after the main campaign.

Includes full-train control and random-removal controls matched by image count.
Requires the tested portable helpers and source-list preparer embedded above.
"""
import csv
import datetime
import gc
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
def run_source_ablations(campaign_path, data_root, manifest_path, original_data_yaml, expected_manifest):
    campaign = Path(campaign_path)
    original = json.loads((campaign / 'comparison.json').read_text())
    prepared = json.loads((campaign / 'prepared.json').read_text())
    if original['status'] != 'batch_complete' or original['manifest_sha256'] != expected_manifest:
        raise ValueError('Expected completed, verified main campaign')
    if torch.cuda.device_count() < 2:
        raise ValueError('Expected both T4 GPUs')
    output = campaign / ('source_ablations_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    subsets = prepare_source_ablations(data_root, manifest_path, output / 'subsets', expected_manifest, prepared['classes'])
    with Path(manifest_path).open(encoding='utf-8-sig', newline='') as stream:
        validation = [row for row in csv.DictReader(stream) if row['split'] == 'val']
    for row in validation:
        if file_sha(Path(data_root) / row['image_path']) != row['image_sha256'] or file_sha(Path(data_root) / row['label_path']) != row['label_sha256']:
            raise ValueError('Validation payload bytes differ')
    base = dict(original['runs']['R01_recover_L1A1']['config'])
    parent = base['model']
    if file_sha(parent) != original['parent_sha256']:
        raise ValueError('Parent checkpoint differs')
    protocol = original['protocol']
    if str(original_data_yaml) != protocol['data']:
        raise ValueError('Evaluation dataset path differs')
    jobs = dict(D00_full_control=dict(data_yaml=str(original_data_yaml), train_images=12000,
                                     train_boxes=33272, train_identity_sha256=expected_manifest,
                                     identity_note='Full release manifest hash; other subsets hash selected train payload identities.'))
    jobs.update(subsets['subsets'])
    state = dict(status='running', manifest_sha256=expected_manifest, protocol=protocol,
                 parent_sha256=original['parent_sha256'], target_mAP50=.80, runs={}, errors={},
                 validation_fingerprint_sha256=subsets['validation_fingerprint_sha256'],
                 full_visual_review_complete=False, image_label_files_changed=False,
                 budget_note='Three epochs per job; source/random pairs share image count, batches and steps. Counts across pairs differ.',
                 selection_note='Epochs, datasets and NMS selected on validation; no independent test claim.')
    if 'SOURCE_ABLATION_TRAIN_SOURCE' in globals():
        (output / 'executed_source.py').write_text(SOURCE_ABLATION_TRAIN_SOURCE)
    def preserve(label):
        (output / 'comparison.json').write_text(json.dumps(state, indent=2))
        selected_paths = {Path(run['checkpoint']).resolve() for run in state['runs'].values() if run.get('checkpoint')}
        sources = []
        for path in output.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if relative.parts[0] == 'train' and relative.parts[1] != label and label != 'FINAL':
                continue
            if relative.parts[0] == 'evaluation' and not relative.parts[1].startswith(label + '_') and label != 'FINAL':
                continue
            if label == 'FINAL' and path.suffix == '.pt' and path.resolve() not in selected_paths:
                continue
            sources.append(path)
        archive = campaign.parent / (output.name + '_' + label + '.zip')
        digest = verified_bundle(output, sources, archive)
        print('SOURCE_ABLATION_VERIFIED_ARCHIVE', archive, digest, flush=True)
    class SourceNMSValidator(DetectionValidator):
        def postprocess(self, predictions):
            items = super().postprocess(predictions)
            for item in items:
                raw = torch.cat((item['bboxes'], item['conf'][:, None], item['cls'][:, None]), dim=1)
                merged = merge_detections([raw], iou=.6)
                item.update(bboxes=merged[:, :4], conf=merged[:, 4], cls=merged[:, 5], extra=merged[:, 6:])
            return items
    for label, selection in jobs.items():
        if shutil.disk_usage(output).free < 2 * 1024**3:
            state['status'] = 'stopped_disk_space'
            preserve('FINAL')
            return state
        config = dict(base, name=label, epochs=3, data=selection['data_yaml'], project=str(output / 'train'))
        get_cfg(overrides=config)
        config_path = output / (label + '_config.yaml')
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        record = dict(status='training', selection=selection, config=config, start=time.time())
        state['runs'][label] = record
        try:
            print('SOURCE_ABLATION_TRAIN_START', label, json.dumps(config), flush=True)
            subprocess.run([shutil.which('yolo'), 'detect', 'train', f'cfg={config_path}'], check=True, cwd=str(output))
            run = output / 'train' / label
            with (run / 'results.csv').open(newline='') as stream:
                history = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
            epoch = select_epoch(history, 3)
            checkpoint = run / 'weights' / f'epoch{epoch-1}.pt'
            record.update(status='trained', selected_epoch=epoch, epoch_metrics=history,
                          checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint))
            preserve(label)
            record['measurements'] = {}
            for mode, validator in [('native', DetectionValidator), ('extra_nms_06', SourceNMSValidator)]:
                model = YOLO(str(checkpoint))
                if [model.names[index] for index in range(8)] != prepared['classes'] or not model.model.end2end:
                    raise ValueError('Head or ordered classes differ')
                metrics = model.val(validator=validator, project=str(output / 'evaluation'), name=label + '_' + mode,
                                    exist_ok=False, **protocol)
                measured = dict(manifest_sha256=expected_manifest, protocol=protocol, checkpoint_sha256=file_sha(checkpoint),
                                postprocessing=mode, mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                                precision=float(metrics.box.mp), recall=float(metrics.box.mr), speed=metrics.speed)
                record['measurements'][mode] = measured
                (output / (label + '_' + mode + '.json')).write_text(json.dumps(measured, indent=2))
                print('SOURCE_ABLATION_METRIC', label, mode, json.dumps(measured), flush=True)
                del model, metrics
                gc.collect()
                torch.cuda.empty_cache()
            record.update(status='complete', elapsed_seconds=time.time() - record['start'])
        except Exception:
            record['status'] = 'error'
            state['errors'][label] = traceback.format_exc()
            print('SOURCE_ABLATION_ERROR', label, state['errors'][label], flush=True)
        preserve(label)
    state['status'] = 'complete_with_errors' if state['errors'] else 'complete'
    preserve('FINAL')
    print('SOURCE_ABLATION_BATCH_DONE', json.dumps(state), flush=True)
    return state
