"""One full raw three-view pass, official AP replay across NMS IoUs, Markdown only."""
import csv
import datetime
import gc
import json
import os
import re
import shutil
from functools import partial
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from kaggle_tools.experiment_campaign import file_sha
from kaggle_tools.inference_variants import ExplicitViewsValidator
from kaggle_tools.validation_replay import replay_metrics
from kaggle_tools.view_replay import CacheViewsValidator

root = Path.cwd().resolve()
log = root / 'reports/PROJECT8_V05_LOG.md'
metadata_path = root / '.temp/single_best_selection_20261010.json'
references = re.findall(r'### R07_three_views — full validation hoàn tất\s+```json\s+(.*?)\s+```', log.read_text(encoding='utf-8'), re.S)
assert references, 'Wait for full R07 three-view validation'
reference = json.loads(references[-1])
checkpoint = root / '.temp/project8_v05_candidates/R07_for_nms.pt'
published = root / 'weights/campaign_20261010/BEST_Project8_v05.pt'
initial_best = json.loads(metadata_path.read_text())
assert file_sha(published) == initial_best['checkpoint_sha256']
expected_sha = 'ecb6329999b05fe6399cd8d303fb4c36ade4b07c263bd299abb88e28d34d6785'
manifest_sha = '22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
views = ((1., False), (.83, True), (.67, False))
assert ultralytics.__version__ == '8.4.165'
assert reference['pipeline'] == 'R07_three_views' and reference['extra_nms_iou'] == .6
assert file_sha(checkpoint) == reference['checkpoint_sha256'] == expected_sha
assert reference['manifest_sha256'] == manifest_sha
protocol = reference['protocol']
assert protocol['device'] == 'cpu' and protocol['imgsz'] == 1280 and protocol['quantize'] is None
assert not protocol['nms'] and not protocol['augment'] and not protocol['plots']
dataset = root / 'data/dataset/project8_v0.5_merged_20261008'
assert file_sha(dataset / 'manifest.csv') == manifest_sha
with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = [row for row in csv.DictReader(stream) if row['split'] == 'val']
assert len(rows) == 2500 and sum(int(row['boxes']) for row in rows) == 8359
print('TTA_NMS_PREFLIGHT_START', flush=True)
for index, row in enumerate(rows, 1):
    for path_key, hash_key in [('image_path', 'image_sha256'), ('label_path', 'label_sha256')]:
        assert file_sha(dataset / row[path_key]) == row[hash_key], row['image_id']
    if index % 500 == 0:
        print('TTA_NMS_PAYLOAD_VERIFIED', index, flush=True)
torch.set_num_threads(4)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
output = root / '.temp' / ('TTA_NMS_' + stamp)
output.mkdir(exist_ok=False)
names = {0:'person', 1:'table', 2:'chair', 3:'laptop', 4:'cell phone', 5:'backpack', 6:'book', 7:'bottle'}


def append(title, record):
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n\n### ' + title + '\n\n```json\n' + json.dumps(record, indent=2) + '\n```\n')


def record(metrics):
    return dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                per_class={metrics.names[int(cls)]: dict(AP50=float(metrics.box.ap50[index]),
                AP50_95=float(metrics.box.ap[index])) for index, cls in enumerate(metrics.box.ap_class_index)})


def parity(a, b):
    for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        if abs(a[key] - b[key]) > 1e-6:
            raise ValueError('Actual/replay parity failed: ' + key)


def capture(data_yaml, label, count):
    holder = []
    def factory(**kwargs):
        validator = CacheViewsValidator(views=views, merge_iou=.6, **kwargs)
        holder.append(validator)
        return validator
    model = YOLO(str(checkpoint))
    assert dict(model.names) == names and model.model.end2end
    metrics = model.val(validator=factory, project=str(output), name=label, exist_ok=False,
                        **dict(protocol, data=str(data_yaml)))
    assert len(holder) == 1 and len(holder[0].cached_samples) == count
    samples = holder[0].cached_samples
    measured = record(metrics)
    del model, holder
    gc.collect()
    return samples, measured


append('R07 TTA × NMS sweep — bắt đầu', dict(status='running', timestamp_utc=stamp,
    checkpoint_sha256=expected_sha, manifest_sha256=manifest_sha, views=views, protocol=protocol,
    source_sha256=file_sha(__file__), collector_sha256=file_sha(root / 'kaggle_tools/view_replay.py'),
    replay_sha256=file_sha(root / 'kaggle_tools/validation_replay.py'),
    validation_images=2500, validation_boxes=8359, test_used=False,
    cache_note='Unmerged restored predictions in RAM only, before class-aware NMS and global max_det.'))

try:
    # Real integration smoke: a control forward pass and raw replay must agree.
    smoke_list = output / 'smoke_val.txt'
    smoke_list.write_text('\n'.join((dataset / row['image_path']).as_posix() for row in rows[:3]) + '\n')
    data = yaml.safe_load(Path(protocol['data']).read_text())
    assert 'test' not in data and Path(data['path']).resolve() == dataset
    smoke_yaml = output / 'smoke_data.yaml'
    smoke_yaml.write_text(yaml.safe_dump(dict(data, val=str(smoke_list)), sort_keys=False))
    smoke_samples, captured = capture(smoke_yaml, 'smoke_capture', 3)
    replayed = record(replay_metrics(smoke_samples, names, iou=.6))
    control = YOLO(str(checkpoint)).val(validator=partial(ExplicitViewsValidator, views=views),
        project=str(output), name='smoke_control', exist_ok=False, **dict(protocol, data=str(smoke_yaml)))
    parity(captured, record(control))
    parity(replayed, record(control))
    append('R07 raw TTA cache — real3-image integration PASS', dict(status='PASS', images=3,
        raw_rows=sum(len(sample['predictions']) for sample in smoke_samples),
        ground_truth_boxes=sum(len(sample['target_cls']) for sample in smoke_samples),
        metrics_equal_within=1e-6, used_for_full_AP_claim=False))
    del smoke_samples, control
    print('TTA_NMS_REAL_SMOKE_PASS', flush=True)
    print('TTA_NMS_FULL_CAPTURE_START', flush=True)
    samples, actual = capture(protocol['data'], 'full_capture', 2500)
    assert sum(len(sample['target_cls']) for sample in samples) == 8359
    assert len({sample['im_name'] for sample in samples}) == 2500
    parity(actual, reference)
    anchor = record(replay_metrics(samples, names, iou=.6))
    parity(anchor, actual)
    append('R07 full raw capture — parity PASS', dict(status='PASS', images=2500, boxes=8359,
        raw_rows=sum(len(sample['predictions']) for sample in samples),
        exact_reference_pipeline=reference['pipeline'], reference=actual, metrics_equal_within=1e-6))
    print('TTA_NMS_FULL_ANCHOR_PARITY_PASS', flush=True)
    results = {}
    for threshold in (.3, .4, .5, .6, .7, .8):
        metrics = replay_metrics(samples, names, iou=threshold)
        assert int(sum(metrics.nt_per_class)) == 8359
        result = dict(record(metrics), pipeline='R07_three_views_nms_' + str(threshold),
            checkpoint_sha256=expected_sha, manifest_sha256=manifest_sha, protocol=protocol,
            views=views, extra_nms_iou=threshold, postprocessing='explicit_views_class_aware_nms',
            validation_images=2500, validation_boxes=8359, test_used=False,
            timing_note='AP replay only; no camera latency/FPS measurement.')
        results[str(threshold)] = result
        append('R07 three-view NMS IoU' + str(threshold) + ' — full replay hoàn tất', result)
        print('TTA_NMS_FULL_METRIC', json.dumps(result), flush=True)
        del metrics
    best = max(results.values(), key=lambda item: (item['mAP50'], item['mAP50_95']))
    current = json.loads(metadata_path.read_text())
    assert file_sha(checkpoint) == expected_sha
    assert file_sha(published) == current['checkpoint_sha256']
    assert {k:v for k,v in protocol.items() if k != 'data'} == {k:v for k,v in current['protocol'].items() if k != 'data'}
    assert current['manifest_sha256'] == manifest_sha
    if (best['mAP50'], best['mAP50_95']) > (current['mAP50'], current['mAP50_95']):
        best['kept_as_best'] = True
        best['replaces_pipeline'] = current['pipeline']
        temporary = published.with_suffix('.pt.partial')
        shutil.copy2(checkpoint, temporary)
        assert file_sha(temporary) == expected_sha
        os.replace(temporary, published)
        metadata_path.write_text(json.dumps(dict(best, checkpoint=str(published)), indent=2))
    else:
        best['kept_as_best'] = False
    append('R07 TTA × NMS sweep — kết thúc', dict(status='complete', results_count=6,
        winner=best, checkpoint_file_count=1, raw_cache_persisted=False))
    del samples
    gc.collect()
    smoke_list.unlink()
    smoke_yaml.unlink()
    assert checkpoint.resolve().is_relative_to((root / '.temp/project8_v05_candidates').resolve())
    assert file_sha(checkpoint) == expected_sha
    checkpoint.unlink()
    assert len(list(published.parent.glob('*.pt'))) == 1
    print('TTA_NMS_DONE', json.dumps(best), flush=True)
except Exception:
    import traceback
    append('R07 TTA × NMS sweep — lỗi', dict(status='error', traceback=traceback.format_exc()))
    raise
