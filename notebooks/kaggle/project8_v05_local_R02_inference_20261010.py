"""Actual full validation on local CPU while Kaggle trains independently.

Run from the project root. Requires the downloaded R02 epoch4 checkpoint. Uses the identical CPU validation YAML as the baseline/R01 run.
CPU FP32 measurements form their own table, separate from Kaggle GPU FP16.
"""
import csv
import datetime
import gc
import json
import shutil
import traceback
from functools import partial
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from kaggle_tools.experiment_campaign import file_sha, rank_results, verified_bundle
from kaggle_tools.inference_variants import ExplicitViewsValidator
from kaggle_tools.validation_replay import CacheDetectionValidator, replay_metrics

root = Path.cwd()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
output = root / 'reports/campaign_20261010' / ('local_R02_inference_' + stamp)
output.mkdir(exist_ok=False)
shutil.copy2(Path(__file__), output / 'executed_source.py')
manifest_sha = '22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
dataset = root / 'data/dataset/project8_v0.5_merged_20261008'
assert file_sha(dataset / 'manifest.csv') == manifest_sha
assert ultralytics.__version__ == '8.4.165'
with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
    validation = [row for row in csv.DictReader(stream) if row['split'] == 'val']
assert len(validation) == 2500 and sum(int(row['boxes']) for row in validation) == 8359
print('LOCAL_FULL_VALIDATION_PREFLIGHT_START', output, flush=True)
for index, row in enumerate(validation, 1):
    for path_key, sha_key in [('image_path', 'image_sha256'), ('label_path', 'label_sha256')]:
        assert file_sha(dataset / row[path_key]) == row[sha_key], row['image_id']
    if index % 500 == 0:
        print('LOCAL_VAL_PAYLOAD_VERIFIED', index, '/2500', flush=True)
data = yaml.safe_load((root / 'configs/data_v05.yaml').read_text())
data.pop('test', None)
data['path'] = dataset.as_posix()
data_yaml = root / 'reports/campaign_20261010/local_inference_20261009_180723/validation_data.yaml'
assert data_yaml.read_text(encoding='utf-8') == yaml.safe_dump(data, sort_keys=False)
shutil.copy2(data_yaml, output / 'validation_data.yaml')
torch.set_num_threads(4)
protocol = dict(data=str(data_yaml), split='val', imgsz=1280, batch=8, device='cpu',
                workers=0, conf=.001, iou=.7, max_det=300, rect=True,
                quantize=None, nms=False, augment=False, plots=False)
checkpoints = {
    'R02': (root / 'reports/campaign_20261010/R02/train/R02_light_longer/weights/epoch3.pt',
            'c0b19c81af77842a59a596a057548641ca920175600edd61dd6bac620afaa0c1'),
}
state = dict(status='running', target_mAP50=.80, manifest_sha256=manifest_sha,
             runtime=dict(ultralytics=ultralytics.__version__, torch=str(torch.__version__),
                          device='CPU', quantize=None, torch_threads=4),
             validation_images=2500, validation_boxes=8359, full_visual_review_complete=False,
             test_used=False, protocol=protocol, results={}, errors={},
             selection_note='NMS/view tuning on validation; not independent test.',
             protocol_note='Local CPU FP32 table; separate from Kaggle CUDA FP16.',
             timing_note='R01 flip validation may run concurrently on this host. Accuracy protocol identical; timings are not a controlled latency benchmark.')


def preserve(label):
    (output / 'comparison.json').write_text(json.dumps(state, indent=2))
    payload = [path for path in output.rglob('*') if path.is_file() and path.suffix != '.zip']
    archive = output.parent / (output.name + '_' + label + '.zip')
    digest = verified_bundle(output, payload, archive)
    print('LOCAL_VALIDATED_ARCHIVE', archive, digest, flush=True)


def record_metrics(metrics, label, checkpoint, checkpoint_sha, **extra):
    assert int(sum(metrics.nt_per_class)) == 8359
    record = dict(manifest_sha256=manifest_sha, protocol=protocol,
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                  mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                  precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                  per_class={metrics.names[int(cls)]: dict(AP50=float(metrics.box.ap50[index]),
                             AP50_95=float(metrics.box.ap[index])) for index, cls in enumerate(metrics.box.ap_class_index)}, **extra)
    state['results'][label] = record
    (output / (label + '.json')).write_text(json.dumps(record, indent=2))
    print('LOCAL_FULL_VAL_METRIC', label, json.dumps(record), flush=True)
    return record


preserve('preflight')
for label, (checkpoint, checkpoint_sha) in checkpoints.items():
    try:
        assert file_sha(checkpoint) == checkpoint_sha
        holder = []

        def factory(**kwargs):
            validator = CacheDetectionValidator(**kwargs)
            holder.append(validator)
            return validator

        model = YOLO(str(checkpoint))
        assert model.model.end2end and [model.names[index] for index in range(8)] == list(data['names'].values())
        metrics = model.val(validator=factory, project=str(output / 'evaluation'), name=label + '_cache',
                            exist_ok=False, **protocol)
        native = record_metrics(metrics, label + '_native', checkpoint, checkpoint_sha, postprocessing='native', speed=metrics.speed)
        assert len(holder) == 1 and len(holder[0].cached_samples) == 2500
        samples, names = holder[0].cached_samples, dict(model.names)
        torch.save(dict(samples=samples, names=names, manifest_sha256=manifest_sha,
                        checkpoint_sha256=checkpoint_sha, protocol=protocol), output / (label + '_cache.pt'))
        replayed = replay_metrics(samples, names)
        for key, value in [('mAP50', replayed.box.map50), ('mAP50_95', replayed.box.map),
                           ('precision', replayed.box.mp), ('recall', replayed.box.mr)]:
            assert abs(native[key] - value) < 1e-10, ('cache/native parity', key)
        del model, metrics, holder
        gc.collect()
        for threshold in (.3, .4, .5, .6, .7, .8):
            replayed = replay_metrics(samples, names, iou=threshold)
            record_metrics(replayed, label + '_nms_' + str(threshold), checkpoint, checkpoint_sha,
                           postprocessing='class_aware_extra_nms', extra_nms_iou=threshold, cache_native_parity=True)
        del samples
        preserve(label)
    except Exception:
        state['errors'][label] = traceback.format_exc()
        print('LOCAL_FULL_VAL_ERROR', label, state['errors'][label], flush=True)
        preserve(label + '_error')
    gc.collect()
label, (checkpoint, checkpoint_sha) = 'R02', checkpoints['R02']
try:
    metrics = YOLO(str(checkpoint)).val(validator=partial(ExplicitViewsValidator, views=((1., False), (1., True))),
        project=str(output / 'evaluation'), name='R02_flip', exist_ok=False, **protocol)
    record_metrics(metrics, 'R02_flip', checkpoint, checkpoint_sha,
                   postprocessing='explicit_flip_views_nms_06', views=((1., False), (1., True)), speed=metrics.speed)
except Exception:
    state['errors']['R02_flip'] = traceback.format_exc()
    print('LOCAL_FULL_VAL_ERROR', 'R02_flip', state['errors']['R02_flip'], flush=True)
state['status'] = 'complete_with_errors' if state['errors'] else 'complete'
state['ranking'] = rank_results(state['results'])
preserve('FINAL')
print('LOCAL_FULL_VALIDATION_DONE', json.dumps(state), flush=True)
