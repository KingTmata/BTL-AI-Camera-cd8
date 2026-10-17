"""Run after the four verified preflight cells of the v05 Kaggle notebook.

This source is joined with src/classroom_ai/experiments/experiment_campaign.py by the local
notebook builder. User authorized exploration; unchanged, unreviewed v05 GT.
"""
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
from ultralytics.utils.nms import TorchNMS
assert ultralytics.__version__ == '8.4.165'
assert torch.cuda.device_count() >= 2
assert file_sha(initial) == BEST_SHA256
assert PREPARED['dataset_manifest_sha256'] == MANIFEST_SHA256
assert PREPARED['training_images'] == 14500
assert PREPARED['training_boxes'] == 41631
assert OUT.name == 'BTL-AI-Camera-v05-campaign-20261010'
assert len([r for r in rows if r['split'] == 'val']) == 2500
assert not (DATA_DIR / 'images/test').exists()
class CampaignNMSValidator(DetectionValidator):
    def postprocess(self, preds):
        items = super().postprocess(preds)
        assert self.end2end and self.args.nms is False
        for item in items:
            selected = []
            for cls in item['cls'].unique():
                ids = (item['cls'] == cls).nonzero(as_tuple=False).flatten()
                keep = TorchNMS.nms(item['bboxes'][ids], item['conf'][ids], .6)
                selected.append(ids[keep])
            if selected:
                keep = torch.cat(selected)
                keep = keep[item['conf'][keep].argsort(descending=True)]
                for key in ['bboxes', 'conf', 'cls', 'extra']:
                    item[key] = item[key][keep]
        return items
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
campaign_dir = OUT / ('batch_' + stamp)
campaign_dir.mkdir(exist_ok=False)
protocol = dict(data=str(DATA_YAML), split='val', imgsz=1280, batch=8,
                device=0, conf=.001, iou=.7, nms=False, max_det=300,
                augment=False, rect=True, quantize=16, workers=2, plots=True)
state = dict(status='running', target_mAP50=.80, parent_sha256=BEST_SHA256,
             manifest_sha256=MANIFEST_SHA256, protocol=protocol,
             full_visual_review_complete=bool(TRAIN_GATE),
             selection_note='Epoch, configurations and NMS selected on validation; not independent test.',
             started_at=stamp, runs={}, errors={}, archives={})
def write_state():
    temporary = campaign_dir / 'comparison.json.partial'
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(campaign_dir / 'comparison.json')
def preserve(label):
    trial_label = label.removesuffix('_trained').removesuffix('_complete').removesuffix('_error')
    selected_paths = {Path(r['checkpoint']) for r in state['runs'].values() if r.get('checkpoint')}
    sources = []
    for p in sorted(campaign_dir.rglob('*')):
        if not p.is_file() or p.suffix not in {'.pt', '.csv', '.yaml', '.json', '.jpg', '.png', '.py', '.txt'}:
            continue
        relative = p.relative_to(campaign_dir)
        include = len(relative.parts) == 1
        if relative.parts[0] == 'train':
            include = relative.parts[1] == trial_label if label != 'FINAL' else (p.suffix != '.pt' or p in selected_paths)
        if relative.parts[0] == 'evaluation':
            include = relative.parts[1].startswith(trial_label + '_') or label == 'FINAL'
        if include and p.name != 'FILES.json':
            sources.append(p)
    archive = OUT / ('campaign_' + stamp + '_' + label + '.zip')
    digest = verified_bundle(campaign_dir, sources, archive)
    state['archives'][label] = dict(path=str(archive), sha256=digest, bytes=archive.stat().st_size)
    write_state()
    print('VERIFIED_ARCHIVE', archive, digest, flush=True)
def evaluate(checkpoint, label):
    result = {}
    checkpoint_hash = file_sha(checkpoint)
    for mode in ['native', 'extra_nms_06']:
        model = YOLO(str(checkpoint))
        assert model.task == 'detect' and [model.names[i] for i in range(8)] == NAMES
        kwargs = dict(protocol, project=str(campaign_dir / 'evaluation'),
                      name=label + '_' + mode, exist_ok=False)
        if mode == 'extra_nms_06':
            kwargs['validator'] = CampaignNMSValidator
        metrics = model.val(**kwargs)
        assert model.model.end2end
        record = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                      manifest_sha256=MANIFEST_SHA256, protocol=protocol,
                      postprocessing=mode, extra_nms_iou=.6 if mode != 'native' else None,
                      mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      speed=metrics.speed, per_class={})
        for j, cls in enumerate(metrics.box.ap_class_index):
            record['per_class'][NAMES[int(cls)]] = dict(AP50=float(metrics.box.ap50[j]),
                                                      AP50_95=float(metrics.box.ap[j]))
        result[mode] = record
        (campaign_dir / (label + '_' + mode + '.json')).write_text(json.dumps(record, indent=2))
        print('CAMPAIGN_METRIC', label, mode, json.dumps(record), flush=True)
        del model, metrics
        gc.collect()
        torch.cuda.empty_cache()
    return result
base = dict(model=str(initial), data=str(DATA_YAML), project=str(campaign_dir / 'train'),
            device='0,1', workers=2, cache=False, fraction=1., patience=0,
            imgsz=1280, batch=32, optimizer='MuSGD', lr0=.0005, lrf=1.,
            cos_lr=False, warmup_epochs=.25, warmup_bias_lr=0., warmup_momentum=.8,
            momentum=.937, weight_decay=.0005, seed=42, deterministic=True,
            flipud=0., degrees=0., shear=0., perspective=0., bgr=0., mixup=0., cutmix=0.,
            close_mosaic=0, multi_scale=False, save=True, save_period=1,
            resume=False, cls_remap=False, val=True, nms=False, max_det=300,
            conf=.001, plots=True, exist_ok=False, **LIGHT_AUG)
trials = [
    dict(name='R01_recover_L1A1', epochs=2),
    dict(name='R02_light_longer', epochs=8),
    dict(name='R03_low_lr_base', epochs=4, lr0=.0001, **BASE_AUG),
    dict(name='R04_class_weight_025', epochs=4, cls_pw=.25),
    dict(name='R05_class_weight_05', epochs=4, cls_pw=.5),
    dict(name='R06_no_mosaic', epochs=4, mosaic=0., scale=.1, translate=.025),
    dict(name='R07_AdamW', epochs=4, optimizer='AdamW', lr0=.0001),
    dict(name='R08_freeze_backbone', epochs=3, freeze=10),
    dict(name='R09_imgsz1536', epochs=3, imgsz=1536, batch=16),
    dict(name='R10_mixup_cutmix', epochs=3, mixup=.1, cutmix=.1, **BASE_AUG),
    dict(name='R11_box_cls_loss', epochs=3, box=9., cls=.7),
    dict(name='R12_yolo26s', epochs=10, model='yolo26s.pt', batch=8,
         lr0=.001, warmup_epochs=1., lrf=.1, cos_lr=True, **BASE_AUG),
]
for trial in trials:
    get_cfg(overrides=dict(base, **trial))
state['planned_trials'] = trials
(campaign_dir / 'prepared.json').write_text(json.dumps(PREPARED, indent=2))
(campaign_dir / 'training_data.yaml').write_text(DATA_YAML.read_text())
if 'CAMPAIGN_SOURCE' in globals():
    (campaign_dir / 'executed_source.py').write_text(CAMPAIGN_SOURCE)
write_state()
print('CAMPAIGN_PREFLIGHT_PASS', campaign_dir, 'epochs_total=', sum(t['epochs'] for t in trials), flush=True)
state['runs']['BASELINE'] = dict(status='complete', measurements=evaluate(initial, 'BASELINE'))
write_state()
preserve('BASELINE')
for trial in trials:
    label = trial['name']
    config = dict(base, **trial)
    if shutil.disk_usage(OUT).free < 2 * 1024**3:
        state['status'] = 'stopped_disk_space'
        write_state()
        break
    config_path = campaign_dir / (label + '_config.yaml')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    record = dict(status='training', config=config, parent_sha256=file_sha(initial)
                  if config['model'] == str(initial) else None,
                  initialization=config['model'], start=time.time())
    state['runs'][label] = record
    write_state()
    try:
        print('CAMPAIGN_TRAIN_START', label, json.dumps(config), flush=True)
        subprocess.run([shutil.which('yolo'), 'detect', 'train', f'cfg={config_path}'],
                       check=True, cwd=str(campaign_dir))
        run_dir = campaign_dir / 'train' / label
        with (run_dir / 'results.csv').open(newline='') as stream:
            history = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
        epoch = select_epoch(history, config['epochs'])
        selected = run_dir / 'weights' / f'epoch{epoch-1}.pt'
        assert selected.is_file()
        record.update(status='trained', epoch_metrics=history, selected_epoch=epoch,
                      checkpoint=str(selected), checkpoint_sha256=file_sha(selected))
        write_state()
        preserve(label + '_trained')
        record['measurements'] = evaluate(selected, label)
        record.update(status='complete', elapsed_seconds=time.time() - record['start'])
        write_state()
        preserve(label + '_complete')
        print('CAMPAIGN_TRIAL_COMPLETE', label, flush=True)
    except Exception as exc:
        record['status'] = 'error'
        state['errors'][label] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        write_state()
        preserve(label + '_error')
        print('CAMPAIGN_TRIAL_ERROR', label, traceback.format_exc(), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
else:
    state['status'] = 'batch_complete'
write_state()
preserve('FINAL')
print('CAMPAIGN_BATCH_DONE', state['status'], campaign_dir, flush=True)
