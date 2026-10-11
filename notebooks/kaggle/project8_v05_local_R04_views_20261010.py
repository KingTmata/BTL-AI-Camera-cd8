"""Full CPU R04 flip/scale validation; keep one best weight and Markdown logs."""
import csv
import datetime
import json
import os
import shutil
import traceback
from functools import partial
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from kaggle_tools.experiment_campaign import file_sha, select_epoch
from kaggle_tools.inference_variants import ExplicitViewsValidator

root = Path.cwd().resolve()
log = root / 'reports/PROJECT8_V05_LOG.md'
candidate = root / '.temp/project8_v05_candidates/R04_epoch3.pt'
checkpoint_sha = '087eaca77696eb323464f4294fce43b877de430e0bb26efb55e18322a3c02758'
manifest_sha = '22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
best_metadata_path = root / '.temp/single_best_selection_20261010.json'
best_path = root / 'weights/campaign_20261010/BEST_Project8_v05.pt'
assert ultralytics.__version__ == '8.4.165'
assert file_sha(candidate) == checkpoint_sha
with (candidate.parent / 'R04_results.csv').open(encoding='utf-8-sig', newline='') as stream:
    epoch_rows = list(csv.DictReader(stream))
assert select_epoch(epoch_rows, 4) == 4
dataset = root / 'data/dataset/project8_v0.5_merged_20261008'
assert file_sha(dataset / 'manifest.csv') == manifest_sha
with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
    validation = [row for row in csv.DictReader(stream) if row['split'] == 'val']
assert len(validation) == 2500 and sum(int(row['boxes']) for row in validation) == 8359
print('R04_VIEWS_PREFLIGHT_START', flush=True)
for index, row in enumerate(validation, 1):
    for path_key, sha_key in [('image_path', 'image_sha256'), ('label_path', 'label_sha256')]:
        assert file_sha(dataset / row[path_key]) == row[sha_key], row['image_id']
    if index % 500 == 0:
        print('R04_VIEWS_VAL_PAYLOAD_VERIFIED', index, flush=True)
data = yaml.safe_load((root / 'configs/data_v05.yaml').read_text())
data.pop('test', None)
data['path'] = dataset.as_posix()
expected_names = ['person', 'table', 'chair', 'laptop', 'cell phone', 'backpack', 'book', 'bottle']
assert list(data['names'].values()) == expected_names
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
output = root / '.temp' / ('R04_views_' + stamp)
output.mkdir(exist_ok=False)
data_yaml = output / 'validation_data.yaml'
data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
torch.set_num_threads(4)
protocol = dict(data=str(data_yaml), split='val', imgsz=1280, batch=8, device='cpu', workers=0,
                conf=.001, iou=.7, max_det=300, rect=True, quantize=None, nms=False,
                augment=False, plots=False)
old_best = json.loads(best_metadata_path.read_text())
# The runtime YAML path changes after cleanup; all material settings and GT must match.
assert {key: value for key, value in protocol.items() if key != 'data'} == {
    key: value for key, value in old_best['protocol'].items() if key != 'data'}
assert file_sha(best_path) == old_best['checkpoint_sha256']


def append_record(title, record):
    text = '\n\n### ' + title + '\n\n```json\n' + json.dumps(record, indent=2) + '\n```\n'
    with log.open('a', encoding='utf-8') as stream:
        stream.write(text)


append_record('R04 CPU explicit views — bắt đầu', dict(
    timestamp_utc=stamp, status='running', checkpoint_sha256=checkpoint_sha,
    manifest_sha256=manifest_sha, validation_images=2500, validation_boxes=8359,
    source_sha256=file_sha(__file__), inference_helper_sha256=file_sha(root / 'kaggle_tools/inference_variants.py'),
    csv_sha256=file_sha(candidate.parent / 'R04_results.csv'), selected_epoch=4,
    protocol=protocol, validation_yaml_sha256=file_sha(data_yaml), test_used=False,
    views_to_measure={'R04_flip': [[1., False], [1., True]],
                      'R04_three_views': [[1., False], [.83, True], [.67, False]]},
    runtime=dict(ultralytics=ultralytics.__version__, torch=str(torch.__version__), device='CPU', torch_threads=4)))
errors = {}
for label, views in [('R04_flip', ((1., False), (1., True))),
                     ('R04_three_views', ((1., False), (.83, True), (.67, False)))]:
    try:
        assert file_sha(candidate) == checkpoint_sha
        model = YOLO(str(candidate))
        assert model.model.end2end and [model.names[index] for index in range(8)] == expected_names
        print('R04_VIEWS_FULL_VAL_START', label, flush=True)
        metrics = model.val(validator=partial(ExplicitViewsValidator, views=views),
                            project=str(output), name=label, exist_ok=False, **protocol)
        assert int(sum(metrics.nt_per_class)) == 8359
        record = dict(pipeline=label, checkpoint_sha256=checkpoint_sha, manifest_sha256=manifest_sha,
                      mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      precision=float(metrics.box.mp), recall=float(metrics.box.mr), views=views,
                      postprocessing='explicit_views_class_aware_nms_06', extra_nms_iou=.6,
                      protocol=protocol, test_used=False, validation_images=2500, validation_boxes=8359,
                      speed=metrics.speed, per_class={metrics.names[int(cls)]:
                      dict(AP50=float(metrics.box.ap50[index]), AP50_95=float(metrics.box.ap[index]))
                      for index, cls in enumerate(metrics.box.ap_class_index)})
        best = json.loads(best_metadata_path.read_text())
        if (record['mAP50'], record['mAP50_95']) > (best['mAP50'], best['mAP50_95']):
            assert file_sha(best_path) == best['checkpoint_sha256']
            temporary = best_path.with_suffix('.pt.partial')
            shutil.copy2(candidate, temporary)
            assert file_sha(temporary) == checkpoint_sha
            os.replace(temporary, best_path)
            record['replaces_pipeline'] = best['pipeline']
            record['kept_as_best'] = True
            best_metadata_path.write_text(json.dumps(dict(record, checkpoint=str(best_path)), indent=2))
        else:
            record['kept_as_best'] = False
        append_record(label + ' — full validation hoàn tất', record)
        print('R04_VIEWS_FULL_VAL_METRIC', label, json.dumps(record), flush=True)
        del model, metrics
    except Exception:
        errors[label] = traceback.format_exc()
        append_record(label + ' — lỗi', dict(status='error', traceback=errors[label]))
        print('R04_VIEWS_ERROR', label, errors[label], flush=True)
append_record('R04 CPU explicit views — kết thúc', dict(status='complete_with_errors' if errors else 'complete',
    errors=errors, kept_checkpoint=str(best_path), checkpoint_sha256=file_sha(best_path)))
if not errors:
    assert candidate.resolve().is_relative_to((root / '.temp/project8_v05_candidates').resolve())
    candidate.unlink()
print('R04_VIEWS_DONE', json.dumps(errors), flush=True)
