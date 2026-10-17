"""Build an isolated E3 original campaign without modifying application defaults."""
import ast
import json
from pathlib import Path
from classroom_ai.paths import PROJECT_ROOT as ROOT
source = json.loads((ROOT/'notebooks/kaggle/cp07_e3_aug1280_e7_retry1.ipynb').read_text(encoding='utf-8'))
cells = source['cells']
config = ''.join(cells[1]['source']).replace('checkpoint07-e3-aug1280-e7-20261014-retry1', 'e3-original-confidence-tiles-ablation-20261014')
config = config.replace('CONFIG_VERIFIED E3 AUG7', 'CONFIG_VERIFIED E3 ORIGINAL CAMPAIGN')
config = config.replace('# 1. Six method experiments; user authorized launch', '# 1. Five independent method experiments; user authorized pilot')
config += '\nCAMPAIGN_BUDGET_HOURS = 5.0\n'
setup = ''.join(cells[2]['source'])
setup = setup[:setup.index('AUGMENTATIONS =')]
core = ''.join(cells[5]['source'])
# Keep validation and inference in FP32 while avoiding Ultralytics' deprecated half flag.
core = core.replace('half=False', 'quantize=32').replace('halfFalse', 'quantize32')
start = core.index('    config.update(name=source')
end = core.index('\nstarted = time.monotonic()', start)
replacement = '''    base = dict(config, epochs=7, patience=0, imgsz=1280, batch=32, save_period=-1, bgr=0.)
    groups = [
        ('CONTROL', {}, []),
        ('LIGHT_CONTRAST', {}, [A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.20, p=.30),
                               A.RandomGamma(gamma_limit=(85,115), p=.15), A.CLAHE(clip_limit=(1,2), p=.05)]),
        ('GEOMETRY', dict(degrees=8., shear=1., perspective=.0001), []),
        ('GRAY', {}, [A.ToGray(p=.15)]),
        ('INVERT', {}, [A.InvertImg(p=.05)]),
    ]
    for group, changes, transforms in groups:
        serialized = [A.to_dict(t) for t in transforms]
        from ultralytics.data.augment import Albumentations as YOLOAlbumentations
        aug_probe = YOLOAlbumentations(transforms=serialized)
        assert aug_probe.transform is not None and not aug_probe.contains_spatial
        assert len(aug_probe.transform.transforms) == len(transforms)
        jobs.append(dict(base, **changes, name=source+'_'+group+'_1280_E7_B32', augmentations=serialized))
    print('INITIAL_VERIFIED', source, weight_hash, flush=True)

assert len(jobs)==5
assert all(j['epochs']==7 and j['imgsz']==1280 and j['batch']==32 and j['device']=='0,1' for j in jobs)
assert all(j['model']==str(initials['E3_AdamW_lr1e4']) and j['optimizer']=='AdamW' and j['lr0']==.0001 for j in jobs)
print('CAMPAIGN_PREFLIGHT_PASS original E3; five independent runs,7epoch each,1280,batch32,T4x2', flush=True)
print('CAMPAIGN_LIST', json.dumps(jobs, indent=2), flush=True)
'''
core = core[:start] + replacement + core[end:]
core = core.replace('2.5*3600', 'CAMPAIGN_BUDGET_HOURS*3600')
core = core.replace("print('PROGRESS', config['name'], 'epoch', epoch, '/', config['epochs'],\n                              'mAP50-95', fitness, flush=True)",
                    "print('PROGRESS_FULL', config['name'], json.dumps(latest), flush=True)")
core = core.replace("per_class_AP50_95=dict(zip(NAMES, map(float, metrics.box.maps))),", "per_class=json.loads(json.dumps(metrics.summary(), default=lambda v: v.item())), per_class_AP50_95=dict(zip(NAMES, map(float, metrics.box.maps))),")
anchor = 'preserve_state()\nfor config in jobs:'
core = core.replace(anchor, '''preserve_state()
state['baseline_native1280']=native_eval(initials['E3_AdamW_lr1e4'], 'original_native1280')
preserve_state()
state['status']='confidence_and_tiles'
preserve_state()
try:
    state['inference']=inference_campaign(initials['E3_AdamW_lr1e4'], [r for r in train_val if r['split']=='val'],
                                        DATA_DIR, OUT/'inference', NAMES, started+CAMPAIGN_BUDGET_HOURS*3600)
except Exception:
    state.update(status='inference_failed_requires_review', error=traceback.format_exc())
    preserve_state()
    raise
preserve_state()
for config in jobs:''')
core = core.replace("        record['before'][str(size)]=native_eval(initial,label+'_before_'+str(size),size); preserve_state()",
                    "        record['before'][str(size)]=state['baseline_native1280']; preserve_state()")
core = core.replace("        record['status']='completed'; preserve_state()", "        record['status']='completed'; preserve_state()\n        print('NATIVE_BEST_FULL', label, json.dumps(record['after_best']['1280']), flush=True)")
core = core.replace('E3_AUGMENTATION_COMPLETE seven additional epochs at1280. Same validation1280 before/after; no classroom/test claim.',
                    'E3_ORIGINAL_CAMPAIGN_COMPLETE confidence/tile comparison and five independent7epoch trials. Unreviewed methodology only.')
evaluation = (ROOT/'src/classroom_ai/experiments/e3_improvement_eval.py').read_text(encoding='utf-8')
codes = [config, setup, ''.join(cells[3]['source']), ''.join(cells[4]['source']), evaluation+'\n\n'+core]
for index, code in enumerate(codes):
    ast.parse(code, filename=f'campaign_cell{index}')
source['cells'] = [dict(cell_type='markdown', metadata={}, source=['# E3 ORIGINAL: confidence, overlapping tiles and augmentation ablation\n',
    'E3 original best; five independent7epoch runs1280/B32/T4x2. Validation only; labels unchanged; no application promotion.\n'])]
source['cells'] += [dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code.splitlines(keepends=True)) for code in codes]
target = ROOT/'notebooks/kaggle/e3_original_improvement_20261014.ipynb'
target.write_text(json.dumps(source, ensure_ascii=False, indent=1), encoding='utf-8')
print(target)
print('SYNTAX_VERIFIED five code cells; E3original; seven epochs per training trial')
