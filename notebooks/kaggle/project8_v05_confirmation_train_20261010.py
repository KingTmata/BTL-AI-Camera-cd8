"""Run after the main batch: seeds, longer training, freeze/unfreeze and conditional KD."""
import csv
import datetime
import gc
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
def run_confirmation(campaign_path, data_root, manifest_path, expected_manifest):
    campaign = Path(campaign_path)
    original = json.loads((campaign / 'comparison.json').read_text())
    prepared = json.loads((campaign / 'prepared.json').read_text())
    if ultralytics.__version__ != '8.4.165' or torch.cuda.device_count() < 2:
        raise ValueError('Expected pinned environment and both T4 GPUs')
    if file_sha(manifest_path) != expected_manifest or original['manifest_sha256'] != expected_manifest:
        raise ValueError('Manifest differs')
    if prepared['dataset_manifest_sha256'] != expected_manifest:
        raise ValueError('Prepared dataset differs')
    with Path(manifest_path).open(encoding='utf-8-sig', newline='') as stream:
        validation = [row for row in csv.DictReader(stream) if row['split'] == 'val']
    if len(validation) != 2500 or sum(int(row['boxes']) for row in validation) != 8359:
        raise ValueError('Validation counts differ')
    for row in validation:
        for path_key, hash_key in [('image_path', 'image_sha256'), ('label_path', 'label_sha256')]:
            if file_sha(Path(data_root) / row[path_key]) != row[hash_key]:
                raise ValueError('Validation bytes differ: ' + row['image_id'])
    output = campaign / ('confirmation_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
    # Metadata defines known scene groups; every validation GT remains in place.
    # These two controls must start from fresh COCO nano, never a leaked warm start.
    output.mkdir(exist_ok=False)
    group_controls = prepare_group_controls(data_root, manifest_path, output / 'group_controls',
                                           expected_manifest, prepared['classes'])
    plan = build_confirmation_plan(original, output, group_controls=group_controls)
    winner = original['runs'][plan['winning_nano']]
    if file_sha(winner['checkpoint']) != winner['checkpoint_sha256']:
        raise ValueError('Selected nano bytes differ')
    if file_sha(winner['config']['model']) != original['parent_sha256']:
        raise ValueError('Seed initializer differs')
    if plan['teacher_better_native']:
        teacher = original['runs'][plan['teacher']]
        if file_sha(teacher['checkpoint']) != teacher['checkpoint_sha256']:
            raise ValueError('Teacher bytes differ')
        for checkpoint in [winner['checkpoint'], teacher['checkpoint']]:
            model = YOLO(checkpoint)
            if not model.model.end2end or [model.names[i] for i in range(8)] != prepared['classes']:
                raise ValueError('Teacher/student head or classes differ')
            del model
    for job in plan['jobs']:
        get_cfg(overrides=job['config'])
        if job.get('initialization_kind') == 'official_coco_pretrained':
            initializer = YOLO(job['config']['model'])
            if not initializer.model.end2end or initializer.model.yaml.get('scale') != 'n':
                raise ValueError('Expected official YOLO26 nano initializer')
            source_path = Path(initializer.ckpt_path).resolve()
            job['config']['model'] = str(source_path)
            job['initializer_sha256'] = file_sha(source_path)
            del initializer
    state = dict(status='running', plan=plan, manifest_sha256=expected_manifest,
                 protocol=original['protocol'], runs={}, errors={}, test_used=False,
                 validation_images=2500, validation_boxes=8359, full_visual_review_complete=False)
    if 'CONFIRMATION_TRAIN_SOURCE' in globals():
        (output / 'executed_source.py').write_text(CONFIRMATION_TRAIN_SOURCE)
        state['source_sha256'] = file_sha(output / 'executed_source.py')
    class ConfirmationNMSValidator(DetectionValidator):
        def postprocess(self, predictions):
            items = super().postprocess(predictions)
            if not self.end2end or self.args.nms:
                raise ValueError('Expected native end-to-end detections')
            for item in items:
                raw = torch.cat((item['bboxes'], item['conf'][:, None], item['cls'][:, None]), dim=1)
                merged = merge_detections([raw], iou=.6)
                item.update(bboxes=merged[:, :4], conf=merged[:, 4], cls=merged[:, 5], extra=merged[:, 6:])
            return items
    def preserve(label):
        temporary = output / 'comparison.json.partial'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / 'comparison.json')
        selected = {Path(run['checkpoint']).resolve() for run in state['runs'].values() if run.get('checkpoint')}
        sources = [path for path in output.rglob('*') if path.is_file() and path.suffix != '.partial'
                   and (path.suffix != '.pt' or path.resolve() in selected)]
        archive = campaign.parent / (output.name + '_' + label + '.zip')
        print('CONFIRMATION_VERIFIED_ARCHIVE', archive, verified_bundle(output, sources, archive), flush=True)
    print('CONFIRMATION_PLAN', json.dumps(plan), flush=True)
    for job in plan['jobs']:
        label, config = job['label'], job['config']
        if job['requires'] and state['runs'].get(job['requires'], {}).get('status') != 'complete':
            state['runs'][label] = dict(status='skipped_dependency', requires=job['requires'])
            preserve(label)
            continue
        if shutil.disk_usage(output).free < 2 * 1024**3:
            state['status'] = 'stopped_disk_space'
            preserve('FINAL')
            return state
        record = dict(status='training', config=config, start=time.time())
        if 'train_selection' in job:
            record['train_selection'] = job['train_selection']
        state['runs'][label] = record
        try:
            record['initializer_sha256'] = file_sha(config['model'])
            if job.get('initializer_sha256') and record['initializer_sha256'] != job['initializer_sha256']:
                raise ValueError('Fresh pretrained initializer bytes differ')
            if config.get('distill_model'):
                record['teacher_sha256'] = file_sha(config['distill_model'])
            config_path = output / (label + '_config.yaml')
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            print('CONFIRMATION_TRAIN_START', label, json.dumps(config), flush=True)
            subprocess.run([shutil.which('yolo'), 'detect', 'train', f'cfg={config_path}'], check=True, cwd=str(output))
            run = output / 'train' / label
            with (run / 'results.csv').open(newline='') as stream:
                history = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
            epoch = select_epoch(history, config['epochs'])
            checkpoint = run / 'weights' / f'epoch{epoch-1}.pt'
            record.update(status='trained', selected_epoch=epoch, epoch_metrics=history,
                          checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint), measurements={})
            for mode, validator in [('native', DetectionValidator), ('extra_nms_06', ConfirmationNMSValidator)]:
                model = YOLO(str(checkpoint))
                if [model.names[i] for i in range(8)] != prepared['classes']:
                    raise ValueError('Ordered classes differ')
                metrics = model.val(validator=validator, project=str(output / 'evaluation'), name=label + '_' + mode,
                                    exist_ok=False, **original['protocol'])
                if int(sum(metrics.nt_per_class)) != 8359:
                    raise ValueError('Evaluated GT count differs')
                measurement = dict(manifest_sha256=expected_manifest, protocol=original['protocol'],
                    checkpoint_sha256=file_sha(checkpoint), postprocessing=mode, mAP50=float(metrics.box.map50),
                    mAP50_95=float(metrics.box.map), precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                    speed=metrics.speed, per_class={metrics.names[int(cls)]: dict(AP50=float(metrics.box.ap50[index]),
                    AP50_95=float(metrics.box.ap[index])) for index, cls in enumerate(metrics.box.ap_class_index)})
                record['measurements'][mode] = measurement
                print('CONFIRMATION_METRIC', label, mode, json.dumps(measurement), flush=True)
                del model, metrics
                gc.collect()
                torch.cuda.empty_cache()
            record.update(status='complete', elapsed_seconds=time.time() - record['start'])
        except Exception:
            record['status'] = 'error'
            state['errors'][label] = traceback.format_exc()
            print('CONFIRMATION_ERROR', label, state['errors'][label], flush=True)
        preserve(label)
    state['status'] = 'complete_with_errors' if state['errors'] else 'complete'
    preserve('FINAL')
    print('CONFIRMATION_BATCH_DONE', json.dumps(state), flush=True)
    return state
