"""Run only after the campaign finishes, with its verified dataset still loaded.

The notebook builder embeds portable inference, raw-cache and replay helpers.
All results are actual 2,500-image validation; selecting them is still val tuning.
"""
import datetime
import gc
import json
import math
import traceback
from functools import partial
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
import yaml


def build_inference_jobs(chosen):
    if not chosen or len(chosen) > 3 or len(set(chosen)) != len(chosen):
        raise ValueError('Select one to three distinct measured checkpoints')
    flip = ((1., False), (1., True))
    scales = ((1., False), (.83, True), (.67, False))
    jobs = [(chosen[0] + '_single_view_parity', [chosen[0]], ((1., False),))]
    jobs.extend((name + '_flip', [name], flip) for name in chosen)
    jobs.extend((name + '_three_views', [name], scales) for name in chosen)
    if len(chosen) >= 2:
        jobs.extend([('ensemble_top2', chosen[:2], ((1., False),)),
                     ('ensemble_top2_flip', chosen[:2], flip)])
    if len(chosen) == 3:
        jobs.append(('ensemble_top3', chosen, ((1., False),)))
    return jobs


def _require_scores_parity(actual, expected):
    for key in ('mAP50', 'mAP50_95', 'precision', 'recall'):
        values = float(actual[key]), float(expected[key])
        if not all(math.isfinite(value) for value in values) or abs(values[0] - values[1]) > 1e-6:
            raise ValueError('Inference metric parity failed: ' + key)
    if 'per_class' in actual or 'per_class' in expected:
        if set(actual.get('per_class', {})) != set(expected.get('per_class', {})):
            raise ValueError('Inference class coverage differs')
        for name, expected_values in expected['per_class'].items():
            for key in ('AP50', 'AP50_95'):
                values = float(actual['per_class'][name][key]), float(expected_values[key])
                if not all(math.isfinite(value) for value in values) or abs(values[0] - values[1]) > 1e-6:
                    raise ValueError('Inference class parity failed: ' + name + ' ' + key)


def require_full_cache_parity(samples, actual, replayed):
    if len(samples) != 2500 or len({sample['im_name'] for sample in samples}) != 2500:
        raise ValueError('Missing or duplicate validation images')
    if sum(len(sample['target_cls']) for sample in samples) != 8359:
        raise ValueError('Full validation ground truth differs')
    _require_scores_parity(actual, replayed)


def _metric_record(metrics):
    return dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
        precision=float(metrics.box.mp), recall=float(metrics.box.mr),
        per_class={metrics.names[int(cls)]: dict(AP50=float(metrics.box.ap50[j]),
            AP50_95=float(metrics.box.ap[j])) for j, cls in enumerate(metrics.box.ap_class_index)})


def _capture_views(paths, views, protocol, output, label):
    holders = []
    def factory(**kwargs):
        validator = CacheViewsValidator(views=views, companions=paths[1:], merge_iou=.6, **kwargs)
        holders.append(validator)
        return validator
    metrics = YOLO(paths[0]).val(validator=factory, project=str(output), name=label,
        exist_ok=False, **protocol)
    if len(holders) != 1:
        raise ValueError('Unexpected raw-cache validator count')
    return metrics, holders[0].cached_samples


def _smoke_raw_pipeline(paths, views, protocol, output, label):
    data = yaml.safe_load(Path(protocol['data']).read_text())
    val_root = Path(data['path']) / data['val']
    images = sorted(path for path in val_root.iterdir() if path.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:3]
    if len(images) != 3:
        raise ValueError('Three actual validation images are required for raw-cache smoke')
    image_list, smoke_yaml = output / (label + '_smoke_val.txt'), output / (label + '_smoke_data.yaml')
    try:
        image_list.write_text('\n'.join(path.as_posix() for path in images) + '\n')
        smoke_yaml.write_text(yaml.safe_dump(dict(data, val=str(image_list)), sort_keys=False))
        smoke_protocol = dict(protocol, data=str(smoke_yaml), plots=False)
        captured, samples = _capture_views(paths, views, smoke_protocol, output / 'smoke', label + '_cache')
        control = YOLO(paths[0]).val(validator=partial(ExplicitViewsValidator, views=views,
            companions=paths[1:], merge_iou=.6), project=str(output / 'smoke'),
            name=label + '_direct', exist_ok=False, **smoke_protocol)
        if len(samples) != 3 or len({sample['im_name'] for sample in samples}) != 3:
            raise ValueError('Incomplete smoke cache')
        boxes = sum(len(sample['target_cls']) for sample in samples)
        if boxes != int(sum(control.nt_per_class)):
            raise ValueError('Smoke ground truth differs')
        replayed = replay_metrics(samples, dict(captured.names), iou=.6, device='cuda:0',
            conf=protocol['conf'], max_det=protocol['max_det'])
        _require_scores_parity(_metric_record(captured), _metric_record(control))
        _require_scores_parity(_metric_record(replayed), _metric_record(control))
        return dict(status='PASS', images=3, boxes=boxes, members=len(paths), views=views,
            raw_rows=sum(len(sample['predictions']) for sample in samples),
            metrics=_metric_record(control), full_validation_AP_claim=False)
    finally:
        for path in (image_list, smoke_yaml):
            if path.exists():
                if not path.resolve().is_relative_to(output.resolve()):
                    raise ValueError('Smoke file resolves outside output')
                path.unlink()


def run_inference_followup(campaign_path, data_yaml, expected_manifest):
    campaign = Path(campaign_path)
    state = json.loads((campaign / 'comparison.json').read_text())
    prepared = json.loads((campaign / 'prepared.json').read_text())
    if state['status'] != 'batch_complete':
        raise ValueError('Wait until the training batch completes')
    if prepared['dataset_manifest_sha256'] != expected_manifest or state['manifest_sha256'] != expected_manifest:
        raise ValueError('Ground truth identity differs')
    if ultralytics.__version__ != '8.4.165' or not torch.cuda.is_available():
        raise ValueError('Expected pinned Kaggle GPU environment')
    data_yaml = Path(data_yaml)
    if data_yaml.read_text() != (campaign / 'training_data.yaml').read_text():
        raise ValueError('Dataset YAML differs from the completed campaign')
    protocol = state['protocol']
    if str(data_yaml) != protocol['data']:
        raise ValueError('Validation data path differs')
    measurements = {name: run['measurements']['extra_nms_06']
                    for name, run in state['runs'].items() if run['status'] == 'complete'}
    ranked = rank_results(measurements)
    chosen, hashes = [], set()
    for name in ranked:
        record = measurements[name]
        checkpoint = Path(record['checkpoint'])
        if file_sha(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint bytes differ: ' + name)
        if record['checkpoint_sha256'] not in hashes:
            chosen.append(name)
            hashes.add(record['checkpoint_sha256'])
        if len(chosen) == 3:
            break
    if not chosen:
        raise ValueError('No complete measured checkpoints')
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = campaign / ('inference_followup_' + timestamp)
    output.mkdir(exist_ok=False)
    jobs = build_inference_jobs(chosen)
    report = dict(status='running', manifest_sha256=expected_manifest, protocol=protocol,
                  chosen=chosen, jobs=jobs, results={}, replays={}, smoke={}, errors={},
                  source_sha256=None,
                  full_visual_review_complete=False,
                  selection_note='Validation-selected checkpoints and pipelines; not independent test.',
                  timing_note='Raw capture copies rows to CPU and changes overhead; replay AP has no deployment FPS claim.')
    if 'INFERENCE_FOLLOWUP_SOURCE' in globals():
        (output / 'executed_source.py').write_text(INFERENCE_FOLLOWUP_SOURCE)
        report['source_sha256'] = file_sha(output / 'executed_source.py')

    def preserve(label):
        (output / 'comparison.json').write_text(json.dumps(report, indent=2))
        sources = [path for path in output.rglob('*') if path.is_file() and path.suffix != '.zip']
        archive = campaign.parent / (output.name + '_' + label + '.zip')
        digest = verified_bundle(output, sources, archive)
        print('INFERENCE_VERIFIED_ARCHIVE', archive, digest, flush=True)

    for label, members, views in jobs:
        paths = [measurements[name]['checkpoint'] for name in members]
        samples, metrics = None, None
        try:
            models = [YOLO(path) for path in paths]
            for model in models:
                if not model.model.end2end or [model.names[i] for i in range(8)] != prepared['classes']:
                    raise ValueError('Detection head or ordered classes differ')
            del models
            report['smoke'][label] = _smoke_raw_pipeline(paths, views, protocol, output, label)
            print('INFERENCE_RAW_SMOKE', label, json.dumps(report['smoke'][label]), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            metrics, samples = _capture_views(paths, views, protocol, output / 'evaluation', label)
            replayed = replay_metrics(samples, dict(metrics.names), iou=.6, device='cuda:0',
                conf=protocol['conf'], max_det=protocol['max_det'])
            require_full_cache_parity(samples, _metric_record(metrics), _metric_record(replayed))
            record = dict(manifest_sha256=expected_manifest, protocol=protocol,
                          checkpoints=[dict(name=name, path=measurements[name]['checkpoint'],
                                            sha256=measurements[name]['checkpoint_sha256']) for name in members],
                          views=views, merge_iou=.6, speed=metrics.speed,
                          validation_images=2500, validation_boxes=8359, test_used=False,
                          full_raw_cache_anchor_parity=True, **_metric_record(metrics))
            if label.endswith('_single_view_parity'):
                reference = measurements[members[0]]
                _require_scores_parity(record, reference)
            report['results'][label] = record
            (output / (label + '.json')).write_text(json.dumps(record, indent=2))
            print('INFERENCE_METRIC', label, json.dumps(record), flush=True)
            names = dict(metrics.names)
            metrics = None
            del replayed
            gc.collect()
            torch.cuda.empty_cache()
            report['replays'][label] = {}
            for threshold in (.3, .4, .5, .6, .7, .8):
                replay = replay_metrics(samples, names, iou=threshold, device='cuda:0',
                    conf=protocol['conf'], max_det=protocol['max_det'])
                if int(sum(replay.nt_per_class)) != 8359:
                    raise ValueError('Replay target count differs')
                result = dict(record, merge_iou=threshold, **_metric_record(replay))
                result.pop('speed')
                result['timing_note'] = 'AP replay only; no inference or camera FPS measurement.'
                report['replays'][label][str(threshold)] = result
                print('INFERENCE_NMS_METRIC', label, threshold, json.dumps(result), flush=True)
                del replay
        except Exception:
            report['errors'][label] = traceback.format_exc()
            print('INFERENCE_ERROR', label, report['errors'][label], flush=True)
            if label.endswith('_single_view_parity'):
                report['status'] = 'stopped_parity_failure'
                preserve('parity_failure')
                return report
        samples, metrics = None, None
        gc.collect()
        torch.cuda.empty_cache()
        preserve(label)
    report['status'] = 'complete_with_errors' if report['errors'] else 'complete'
    variants = {label + '_nms_' + threshold: record for label, thresholds in report['replays'].items()
                for threshold, record in thresholds.items()}
    report['ranking_with_single_view_references'] = rank_results(dict(measurements, **report['results'], **variants))
    preserve('FINAL')
    print('INFERENCE_FOLLOWUP_DONE', json.dumps(report), flush=True)
    return report
