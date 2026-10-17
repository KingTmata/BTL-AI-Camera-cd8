# Larger latency benchmark, 200 validation images x 10 repeats x 3 modes.
from collections import Counter
extended_images=[]
selected_rows=[rows[i] for i in np.linspace(0,2499,200,dtype=int)]
with zipfile.ZipFile(archive) as z:
    for r in selected_rows:
        payload=z.read('project8_v05/'+r['image_path'])
        assert hashlib.sha256(payload).hexdigest()==r['image_sha256']
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        assert image is not None
        extended_images.append(image)
for key in modes:
    for image in extended_images[:20]: run(key,image)
    torch.cuda.synchronize(0)
measurements={key:[] for key in modes}
rounds=[]
def summarize(x):
    return dict(measurements=len(x),p50_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),p99_ms=float(np.percentile(x,99)),mean_ms=float(np.mean(x)),max_ms=max(x),over100ms_count=sum(v>100 for v in x),over100ms_percent=100*sum(v>100 for v in x)/len(x),offline_serial_fps=1000/float(np.mean(x)))
start_extended=time.perf_counter()
for repeat in range(10):
    jobs=[(key,i) for key in modes for i in range(200)]
    random.Random(20261011+repeat).shuffle(jobs)
    one_round={key:[] for key in modes}
    for key,index in jobs:
        if not key.startswith('cpu'): torch.cuda.synchronize(0)
        t=time.perf_counter()
        run(key,extended_images[index])
        if not key.startswith('cpu'): torch.cuda.synchronize(0)
        latency=(time.perf_counter()-t)*1000
        measurements[key].append(dict(image_index=index,round=repeat+1,latency_ms=latency))
        one_round[key].append(latency)
    summary=dict(round=repeat+1,results={key:summarize(x) for key,x in one_round.items()})
    rounds.append(summary)
    print('EXTENDED_ROUND_COMPLETE',json.dumps(summary),flush=True)
    (out/'extended_samples.json').write_text(json.dumps(measurements))
extended=dict(status='complete',gpu=torch.cuda.get_device_name(0),visible_gpu_count=torch.cuda.device_count(),gpu_used=0,torch_version=torch.__version__,cpu_threads=2,imgsz=640,batch=1,forward_passes=1,conf=.25,nms_iou=.6,images=200,repeats=10,warmups=20,training_executed=False,test_used=False,checkpoint_sha256=expected,source_counts=dict(Counter(r['source_id'] for r in selected_rows)),results={key:summarize([x['latency_ms'] for x in records]) for key,records in measurements.items()},rounds=rounds,elapsed_seconds=time.perf_counter()-start_extended,cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(0)/1024**2,cuda_peak_reserved_MiB=torch.cuda.max_memory_reserved(0)/1024**2,note=state['note'])
(out/'extended.json').write_text(json.dumps(extended,indent=2))
print('R04_T4_EXTENDED_COMPLETE',json.dumps(extended),flush=True)
