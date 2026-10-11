# Standalone benchmark only: no training, no test split, only CUDA device 0.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv, hashlib, io, json, random, shutil, subprocess, sys, time, zipfile
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.165'], check=True)
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'T4' in torch.cuda.get_device_name(0)
torch.set_num_threads(2)
cv2.setNumThreads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
out = Path('/kaggle/working/r04_single_t4_latency_20261011')
out.mkdir(exist_ok=True)
expected = '2fedf0ef4cab22911ea9786b86cbb0ec37a0802012a8cb0982ec76d3c73c2105'
pt = next(Path('/kaggle/input').rglob('R04_inference.pt.bin'))
assert hashlib.sha256(pt.read_bytes()).hexdigest() == expected
shutil.copyfile(pt, out/'R04.pt')
archive = next(Path('/kaggle/input').rglob('project8_v05_trainval_best_v04_20261008.zip.bin'))
images = []
with zipfile.ZipFile(archive) as z:
    manifest = z.read('project8_v05/manifest.csv')
    assert hashlib.sha256(manifest).hexdigest() == '22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
    rows = [r for r in csv.DictReader(io.StringIO(manifest.decode('utf-8-sig'))) if r['split']=='val']
    assert len(rows)==2500
    for i in np.linspace(0,2499,40,dtype=int):
        r=rows[i]; payload=z.read('project8_v05/'+r['image_path'])
        assert hashlib.sha256(payload).hexdigest()==r['image_sha256']
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        assert image is not None
        images.append(image)
def extra_nms(result):
    det=result.boxes.data.float(); selected=[]
    for cls in det[:,5].unique():
        ids=(det[:,5]==cls).nonzero().flatten()
        selected.append(ids[TorchNMS.nms(det[ids,:4],det[ids,4],.6)])
    if selected:
        ids=torch.cat(selected)
        result.boxes.data=det[ids[det[ids,4].argsort(descending=True)][:300]]
    return result
modes={'cpu_fp32':('cpu',None),'t4_fp32':(0,None),'t4_fp16':(0,16)}
models={k:YOLO(str(out/'R04.pt')) for k in modes}
def run(key,image):
    device,quantize=modes[key]
    args=dict(imgsz=640,device=device,conf=.25,nms=False,augment=False,verbose=False,batch=1)
    if quantize is not None: args['quantize']=quantize
    result=models[key].predict(image,**args)[0]
    extra_nms(result).plot()
for key in modes:
    for image in images[:5]: run(key,image)
    parameter=next(models[key].predictor.model.model.parameters())
    assert parameter.device.type==('cpu' if key.startswith('cpu') else 'cuda')
    assert parameter.dtype==(torch.float16 if key.endswith('fp16') else torch.float32)
    print('WARMUP_DONE',key,str(parameter.device),str(parameter.dtype),flush=True)
samples={k:[] for k in modes}
jobs=[(k,i) for repeat in range(3) for k in modes for i in range(40)]
random.Random(20261011).shuffle(jobs)
torch.cuda.reset_peak_memory_stats(0)
for key,index in jobs:
    if not key.startswith('cpu'): torch.cuda.synchronize(0)
    start=time.perf_counter()
    run(key,images[index])
    if not key.startswith('cpu'): torch.cuda.synchronize(0)
    samples[key].append((time.perf_counter()-start)*1000)
state=dict(status='complete',gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count(),gpu_used=0,torch_version=torch.__version__,cpu_threads=2,imgsz=640,batch=1,forward_passes=1,conf=.25,nms_iou=.6,images=40,repeats=3,warmups=5,training_executed=False,test_used=False,checkpoint_sha256=expected,memory_budget_cap_applied=False,results={},note='Predict+class-aware NMS+plot, predecoded validation images. CUDA synchronized. Excludes capture, tracker, UI, image decode and network. One physical T4 used; not a simulation of RTX 5060 Laptop. Modes interleaved on same VM. CUDA peak memory includes both resident GPU model variants.')
for key,values in samples.items():
    state['results'][key]=dict(measurements=len(values),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)),over100ms_count=sum(x>100 for x in values),offline_serial_fps=1000/float(np.mean(values)))
state['cuda_peak_allocated_MiB']=torch.cuda.max_memory_allocated(0)/1024**2
state['cuda_peak_reserved_MiB']=torch.cuda.max_memory_reserved(0)/1024**2
(out/'comparison.json').write_text(json.dumps(state,indent=2))
print('R04_SINGLE_T4_COMPLETE',json.dumps(state),flush=True)
