"""Repeat three E3 recipes with independent seeds and paired random val audits."""
import ast
import json
from pathlib import Path
from classroom_ai.paths import PROJECT_ROOT as ROOT
def build():
    notebook = json.loads((ROOT / 'notebooks/kaggle/e3_original_improvement_20261014.ipynb').read_text(encoding='utf-8'))
    config = ''.join(notebook['cells'][1]['source'])
    config = config.replace('e3-original-confidence-tiles-ablation-20261014', 'e3-original-seeds123-2026-20261014')
    config = config.replace('E3 ORIGINAL CAMPAIGN', 'E3 SEED REPEATS')
    config += '\nTRAIN_SEEDS = (123, 2026)\nRANDOM_VAL_IMAGES = 500\n'
    config = config.replace('EPOCHS = 7', 'EPOCHS = 2').replace("(7,1280,32,'0,1')", "(2,1280,32,'0,1')")
    notebook['cells'][1]['source'] = config.splitlines(keepends=True)
    core = ''.join(notebook['cells'][5]['source'])
    core = core.replace("        ('GEOMETRY', dict(degrees=8., shear=1., perspective=.0001), []),\n", '')
    core = core.replace("        ('GRAY', {}, [A.ToGray(p=.15)]),\n", '')
    old = "        jobs.append(dict(base, **changes, name=source+'_'+group+'_1280_E7_B32', augmentations=serialized))"
    new = """        for train_seed in TRAIN_SEEDS:
            jobs.append(dict(base, **changes, seed=train_seed,
                             name=source+'_'+group+'_S'+str(train_seed)+'_1280_E7_B32',
                             augmentations=serialized))"""
    assert core.count(old) == 1
    core = core.replace(old, new).replace('assert len(jobs)==5', 'assert len(jobs)==6')
    core = core.replace('five independent runs,7epoch each', 'six independent runs; seeds123/2026;7epoch each')
    core = core.replace("print('CAMPAIGN_LIST'", "assert {j['seed'] for j in jobs} == {123,2026}\nassert len({j['name'] for j in jobs}) == 6\nprint('CAMPAIGN_LIST'")
    core = core.replace('def native_eval(weights, label, size=1280):', 'def native_eval(weights, label, size=1280, data_yaml=None):')
    core = core.replace("metrics = detector.val(data=str(DATA_YAML),", "metrics = detector.val(data=str(data_yaml or DATA_YAML),")
    core = core.replace("checkpoint_sha256=file_sha(weights))", "checkpoint_sha256=file_sha(weights), evaluation_yaml_sha256=file_sha(data_yaml or DATA_YAML))")
    start = core.index("state['status']='confidence_and_tiles'")
    end = core.index('for config in jobs:', start)
    core = core[:start] + '''# Audits draw only from val; never the held-out test500.
import random
random_val_yaml = {}
state['random_val_subsets'] = {}
state['baseline_random_val'] = {}
val_records = sorted((r for r in train_val if r['split']=='val'), key=lambda r:r['image_path'])
assert len(val_records)==2500 and RANDOM_VAL_IMAGES==500
for audit_seed in TRAIN_SEEDS:
    sample = random.Random(audit_seed).sample(val_records, RANDOM_VAL_IMAGES)
    image_paths = [str((DATA_DIR/r['image_path']).resolve()) for r in sample]
    assert len(set(image_paths)) == RANDOM_VAL_IMAGES
    assert all('/images/val/' in p.replace('\\\\','/') for p in image_paths)
    listing = OUT/('random_val_'+str(audit_seed)+'.txt')
    listing.write_text('\\n'.join(image_paths)+'\\n', encoding='utf-8')
    audit_yaml = OUT/('random_val_'+str(audit_seed)+'.yaml')
    audit_yaml.write_text(yaml.safe_dump(dict(path=str(DATA_DIR.resolve()),train='images/train',
                         val=str(listing.resolve()),nc=8,names=dict(enumerate(NAMES))),sort_keys=False),encoding='utf-8')
    random_val_yaml[audit_seed] = audit_yaml
    state['random_val_subsets'][str(audit_seed)] = dict(seed=audit_seed,images=500,
            boxes=sum(int(r['boxes']) for r in sample),image_paths=[r['image_path'] for r in sample],
            list_sha256=file_sha(listing),selection='uniform without replacement from fixed val2500 only')
    state['baseline_random_val'][str(audit_seed)] = native_eval(initials['E3_AdamW_lr1e4'],
                'original_random500_'+str(audit_seed),data_yaml=audit_yaml)
    print('RANDOM_VAL_BASELINE_FULL',audit_seed,json.dumps(state['baseline_random_val'][str(audit_seed)]),flush=True)
    preserve_state()
''' + core[end:]
    anchor = "        record['after_last_1280']=native_eval(last,label+'_after_last_1280',1280)"
    core = core.replace(anchor, anchor + '''
        record['random_val500'] = native_eval(best,label+'_random500',data_yaml=random_val_yaml[config['seed']])
        record['random_val_seed'] = config['seed']
        record['random_val_list_sha256'] = state['random_val_subsets'][str(config['seed'])]['list_sha256']
        print('RANDOM_VAL_BEST_FULL',label,json.dumps(record['random_val500']),flush=True)''')
    core = core.replace('E3_ORIGINAL_CAMPAIGN_COMPLETE confidence/tile comparison and five independent7epoch trials.',
                        'E3_SEED_REPEATS_COMPLETE six independent7epoch trials seeds123/2026; fullval2500 and paired randomval500.')
    core = core.replace('EPOCHS == 7', 'EPOCHS == 2').replace('epochs=7', 'epochs=2')
    core = core.replace("['epochs']==7", "['epochs']==2").replace("['epochs_completed']==7", "['epochs_completed']==2")
    core = core.replace('_E7_B32', '_E2_B32').replace('7epoch', '2epoch')
    notebook['cells'][5]['source'] = core.splitlines(keepends=True)
    notebook['cells'][0]['source'] = ['# E3 original: CONTROL / LIGHT_CONTRAST / INVERT, seeds123 and2026\n',
        'Six independent runs,2epoch each,1280,batch32,device0,1,T4x2; AdamWlr0.0001. Fullval2500 primary; paired random500 val audit. No test500 or application promotion.\n']
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']),filename=f'seed_repeat_cell{index}')
            cell['outputs']=[]
            cell['execution_count']=None
    target=ROOT/'notebooks/kaggle/e3_seed_repeats_123_2026_20261014.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return target
if __name__=='__main__':
    print(build())
    print('SYNTAX_VERIFIED six runs; two epochs; batch32; device0,1; two paired random val audits')
