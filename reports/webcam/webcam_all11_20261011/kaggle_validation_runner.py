# Standalone validation only. Run this cell, never the historical Run All.
import csv, hashlib, json, subprocess, sys, time, zipfile
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.165'],check=True)
import cv2, numpy as np, torch, ultralytics, yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.nms import TorchNMS
assert ultralytics.__version__=='8.4.165' and torch.cuda.is_available(), 'GPU required; validation only'
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
INPUT=Path('/kaggle/input')
OUT=Path('/kaggle/working/webcam_all11_20261011');OUT.mkdir(exist_ok=True)
import shutil
manifest_path=next(INPUT.rglob('all11_input_manifest.json'))
input_manifest=json.loads(manifest_path.read_text())
assert input_manifest['expected_models']==11 and input_manifest['expected_size_pairs']==44
checkpoints={};missing=[]
for label,meta in input_manifest['models'].items():
    candidates=list(INPUT.rglob(label+'_inference.pt.bin'))
    expected=meta['inference_sha256']
    verified=[p for p in candidates if expected and sha(p)==expected]
    if not verified:
        missing.append(label);continue
    assert len(verified)==1
    target=OUT/(label+'_inference.pt');shutil.copyfile(verified[0],target)
    checkpoints[label]=target
print('ALL11_CHECKPOINT_PREFLIGHT',json.dumps(dict(available=list(checkpoints),missing=missing,expected_models=11)),flush=True)
archives=list(INPUT.rglob('project8_v05_trainval_best_v04_20261008.zip.bin'))
assert len(archives)==1 and sha(archives[0])=='236972ca3e6142d977a29068cf217dc5312c8fb870c8094a98c4960386a0eb5b'
DATA=Path('/tmp/webcam_sizes_validation/project8_v05');DATA.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archives[0]) as archive:
    payload=archive.read('project8_v05/manifest.csv')
    assert hashlib.sha256(payload).hexdigest()=='22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
    rows=[r for r in csv.DictReader(payload.decode('utf-8-sig').splitlines()) if r['split']=='val']
    assert len(rows)==2500 and sum(int(r['boxes']) for r in rows)==8359
    for row in rows:
        for key,hashkey in [('image_path','image_sha256'),('label_path','label_sha256')]:
            relative=Path(row[key]);assert not relative.is_absolute() and '..' not in relative.parts
            target=DATA/relative;target.parent.mkdir(parents=True,exist_ok=True)
            content=archive.read('project8_v05/'+row[key]);assert hashlib.sha256(content).hexdigest()==row[hashkey]
            target.write_bytes(content)
NAMES={i:n for i,n in enumerate(['person','table','chair','laptop','cell phone','backpack','book','bottle'])}
datafile=OUT/'validation_data.yaml'
datafile.write_text(yaml.safe_dump(dict(path=str(DATA),train='images/val',val='images/val',names=NAMES)))
class CachedValidator(DetectionValidator):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.samples=[]
    def update_metrics(self,preds,batch):
        for index,pred in enumerate(preds):
            gt=self._prepare_batch(index,batch)
            det=torch.cat([pred['bboxes'],pred['conf'][:,None],pred['cls'][:,None]],1)
            self.samples.append(dict(im_name=Path(gt['im_file']).name,boxes=gt['bboxes'].cpu(),classes=gt['cls'].cpu(),pred=det.cpu()))
        super().update_metrics(preds,batch)
def nms(det,conf):
    det=det[det[:,4]>conf].float();chosen=[]
    for cls in det[:,5].unique():
        ids=(det[:,5]==cls).nonzero().flatten();chosen.append(ids[TorchNMS.nms(det[ids,:4],det[ids,4],.6)])
    if not chosen:return det
    ids=torch.cat(chosen);return det[ids[det[ids,4].argsort(descending=True)][:300]]
def replay(samples,extra):
    validator=DetectionValidator(args=dict(plots=False,conf=.001,max_det=300));validator.metrics.names=NAMES
    fixed_matcher=DetectionValidator(args=dict(plots=False));fixed_matcher.iouv=torch.linspace(.5,.95,10)
    tp=fp=fn=0
    for s in samples:
        det=nms(s['pred'],.001) if extra else s['pred'].float()
        pred=dict(bboxes=det[:,:4],conf=det[:,4],cls=det[:,5]);gt=dict(bboxes=s['boxes'],cls=s['classes'])
        classes=s['classes'].numpy()
        validator.metrics.update_stats(dict(**validator._process_batch(pred,gt),target_cls=classes,target_img=np.unique(classes),conf=det[:,4].numpy(),pred_cls=det[:,5].numpy(),im_name=s['im_name']))
        det=det[det[:,4]>.25];matched=fixed_matcher._process_batch(dict(bboxes=det[:,:4],conf=det[:,4],cls=det[:,5]),gt)
        t=int(matched['tp'][:,0].sum());tp+=t;fp+=len(det)-t;fn+=len(classes)-t
    validator.metrics.process(plot=False);m=validator.metrics.box
    p=tp/(tp+fp);r=tp/(tp+fn)
    return dict(mAP50=float(m.map50),mAP50_95=float(m.map),precision_evaluator=float(m.mp),recall_evaluator=float(m.mr),fixed_conf_025=dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,F1=2*p*r/(p+r)),per_class={NAMES[int(c)]:dict(AP50=float(m.ap50[i]),AP50_95=float(m.ap[i])) for i,c in enumerate(m.ap_class_index)})
state=dict(status='running',validation_images=2500,validation_boxes=8359,training_executed=False,test_used=False,model_passes=1,device=torch.cuda.get_device_name(0),sizes=[512,640,768,960],results={})
def save(): (OUT/'comparison.json').write_text(json.dumps(state,indent=2))
state['expected_models']=11;state['expected_size_pairs']=44;state['missing_checkpoints']=missing
state['input_manifest']=input_manifest
seed_path=next(INPUT.rglob('R04_existing_gpu_results.json'))
seed=json.loads(seed_path.read_text())
assert len(seed)==4 and all(v['cache_parity'] and v['checkpoint_sha256']==input_manifest['models']['R04']['inference_sha256'] for v in seed.values())
state['results']=seed
state['reused_pairs']=list(seed)
save()
for label,path in checkpoints.items():
    for size in state['sizes']:
        if label+'_'+str(size) in state['results']:continue
        model=YOLO(str(path));assert model.model.end2end and model.names==NAMES
        holders=[]
        def factory(**kwargs):
            validator=CachedValidator(**kwargs);holders.append(validator);return validator
        protocol=dict(data=str(datafile),split='val',imgsz=size,batch=8,device=0,workers=0,conf=.001,iou=.7,max_det=300,rect=True,quantize=16,nms=False,augment=False,plots=False)
        m=model.val(validator=factory,project=str(OUT/'val'),name=label+'_'+str(size),**protocol)
        samples=holders[0].samples;assert len(samples)==2500 and sum(len(s['classes']) for s in samples)==8359
        native=replay(samples,False)
        assert abs(native['mAP50']-float(m.box.map50))<1e-10 and abs(native['mAP50_95']-float(m.box.map))<1e-10
        state['results'][label+'_'+str(size)]=dict(checkpoint_sha256=sha(path),protocol=protocol,native=native,nms06=replay(samples,True),cache_parity=True)
        save();print('SIZE_COMPLETE',label,size,json.dumps(state['results'][label+'_'+str(size)]),flush=True)
        del model,holders,m,samples;torch.cuda.empty_cache()
state['completed_size_pairs']=len(state['results'])
state['status']='complete' if not missing and len(state['results'])==44 else 'partial_missing_checkpoints'
save()
print('ALL11_VALIDATION_RESULT',json.dumps(state),flush=True)
