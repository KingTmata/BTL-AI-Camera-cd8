"""Full validation of one serialized best pipeline; log to Markdown, remove probe PT."""
import copy
import csv
import datetime
import json
import traceback
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from kaggle_tools.experiment_campaign import file_sha
from kaggle_tools.inference_variants import EndToEndViews

root = Path.cwd().resolve()
selection_path = root / '.temp/single_best_selection_20261010.json'
selection = json.loads(selection_path.read_text())
checkpoint = Path(selection['checkpoint'])
assert file_sha(checkpoint) == selection['checkpoint_sha256']
assert selection['pipeline'] == 'R04_three_views_nms_0.5'
assert ultralytics.__version__ == '8.4.165'
assert selection['validation_images'] == 2500 and selection['validation_boxes'] == 8359
protocol = selection['protocol']
assert protocol['device'] == 'cpu' and protocol['quantize'] is None
assert protocol['imgsz'] == 1280 and not protocol['augment'] and not protocol['nms']
data = yaml.safe_load(Path(protocol['data']).read_text())
dataset = Path(data['path'])
assert 'test' not in data and file_sha(dataset / 'manifest.csv') == selection['manifest_sha256']
with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = [row for row in csv.DictReader(stream) if row['split'] == 'val']
assert len(rows) == 2500 and sum(int(row['boxes']) for row in rows) == 8359
for index, row in enumerate(rows, 1):
    for p, h in [('image_path','image_sha256'),('label_path','label_sha256')]:
        assert file_sha(dataset / row[p]) == row[h], row['image_id']
    if index % 500 == 0:
        print('BEST_RELOAD_PAYLOAD_VERIFIED', index, flush=True)

torch.set_num_threads(4)
folder = root / '.temp' / ('best_pipeline_full_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
folder.mkdir(exist_ok=False)
artifact = folder / 'single_pipeline.pt'
record = dict(status='running', scope='Full 2500-image validation through serialized pipeline and default YOLO validator',
    source_sha256=file_sha(__file__), helper_sha256=file_sha(root / 'kaggle_tools/inference_variants.py'),
    component_checkpoint_sha256=selection['checkpoint_sha256'], manifest_sha256=selection['manifest_sha256'],
    protocol=protocol, validation_images=2500, validation_boxes=8359,
    views=selection['views'], extra_nms_iou=selection['extra_nms_iou'], test_used=False,
    reference_pipeline=selection['pipeline'], published_model_replaced=False)
def append(title, value):
    with (root / 'reports/PROJECT8_V05_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n### ' + title + '\n\n```json\n' + json.dumps(value, indent=2) + '\n```\n')

try:
    views = tuple(tuple(v) for v in selection['views'])
    wrapper = EndToEndViews([YOLO(str(checkpoint)).model], views=views,
        iou=selection['extra_nms_iou']).cpu().float().eval()
    wrapper.args = dict(task='detect', imgsz=1280)
    torch.save(dict(model=copy.deepcopy(wrapper), train_args=wrapper.args,
        project8_pipeline=dict(selection, inference_only=True)), artifact)
    record.update(artifact_sha256=file_sha(artifact), artifact_bytes=artifact.stat().st_size)
    del wrapper
    append('Best one-file pipeline — full reload validation bắt đầu', record)
    print('BEST_RELOAD_FULL_START', json.dumps(record), flush=True)
    measured = YOLO(str(artifact)).val(project=str(folder), name='full_reloaded', exist_ok=False, **protocol)
    actual = dict(mAP50=float(measured.box.map50), mAP50_95=float(measured.box.map),
        precision=float(measured.box.mp), recall=float(measured.box.mr),
        speed=measured.speed,
        per_class={measured.names[int(cls)]: dict(AP50=float(measured.box.ap50[j]),
            AP50_95=float(measured.box.ap[j])) for j, cls in enumerate(measured.box.ap_class_index)})
    assert int(sum(measured.nt_per_class)) == 8359
    for key in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        assert abs(actual[key] - selection[key]) <= 1e-6, key
    assert set(actual['per_class']) == set(selection['per_class'])
    for name, values in actual['per_class'].items():
        for key, value in values.items():
            assert abs(value - selection['per_class'][name][key]) <= 1e-6, (name, key)
    record.update(status='PASS', measured=actual, parity_tolerance=1e-6)
except Exception:
    record.update(status='ERROR', error=traceback.format_exc())
finally:
    if artifact.exists():
        assert artifact.resolve().is_relative_to((root / '.temp').resolve())
        artifact.unlink()
    record['temporary_artifact_removed'] = not artifact.exists()
    assert file_sha(checkpoint) == selection['checkpoint_sha256']
    assert json.loads(selection_path.read_text()) == selection
    assert len(list(checkpoint.parent.glob('*.pt'))) == 1
    append('Best one-file pipeline — full reload validation kết thúc', record)
    print('BEST_RELOAD_FULL_DONE', json.dumps(record), flush=True)
if record['status'] != 'PASS':
    raise RuntimeError(record.get('error', 'Full reload validation failed'))
