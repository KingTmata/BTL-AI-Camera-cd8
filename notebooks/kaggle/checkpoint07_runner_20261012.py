"""Sequential exploratory training; execute after verified staging cells."""
import inspect
import time
import traceback
import os
import signal
import subprocess
from ultralytics.utils.torch_utils import EarlyStopping
assert START_TRAINING and ALLOW_UNREVIEWED_PILOT
assert EPOCHS == 10 and PATIENCE == 3
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
assert file_sha(initial) == BEST_SHA256
assert ultralytics.__version__ == '8.4.165'
CHECKPOINT_NAME = 'checkpoint 07 — YOLO26n Project8 v0.4, phiên Kaggle ngày 08/10/2026, best epoch 19'
print(CHECKPOINT_NAME, initial, BEST_SHA256, flush=True)
print('EXPLORATORY: unreviewed v0.5 labels; no classroom improvement claim; no test selection.', flush=True)
probe = YOLO(str(initial))
assert probe.model.yaml.get('scale') == 'n'
backbone_count = len(probe.model.yaml['backbone'])
assert 0 < backbone_count < len(probe.model.model)
backbone_layers = list(range(backbone_count))
print('Verified backbone:', [(i, type(probe.model.model[i]).__name__) for i in backbone_layers], flush=True)
del probe; gc.collect(); torch.cuda.empty_cache()
jobs = build_matrix(initial, DATA_YAML, OUT/'train', backbone_layers)
# Pin observed stopper semantics; strict positive-fitness ties must not reset patience.
stopper_source = inspect.getsource(EarlyStopping)
stopper = EarlyStopping(patience=3)
assert [stopper(i+1, score) for i, score in enumerate([.5, .5, .5, .5])] == [False, False, False, True]
zero_stopper = EarlyStopping(patience=3)
zero_behavior = [zero_stopper(i+1, 0.) for i in range(4)]
(OUT/'early_stopping_source.py').write_text(stopper_source, encoding='utf-8')
started = time.monotonic()
state = dict(status='preflight_pass', checkpoint_name=CHECKPOINT_NAME,
             initial_sha256=BEST_SHA256, manifest_sha256=MANIFEST_SHA256,
             full_visual_review_complete=bool(TRAIN_GATE),
             user_authorized_unreviewed_method_experiment=True,
             classroom_claim_allowed=False, test_used=False, jobs=jobs, runs={},
             environment=dict(ultralytics=ultralytics.__version__, torch=torch.__version__,
                              cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in (0, 1)]),
             early_stopper_zero_behavior=zero_behavior,
             evaluation_note='Native trainer AP1280 first; COCO640 and application evaluation are separate follow-up artifacts.')
def preserve_state():
    state['elapsed_seconds'] = round(time.monotonic()-started, 1)
    temporary = OUT/'status.json.partial'
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(OUT/'status.json')
def archive_trial(label):
    trial = OUT/'train'/label
    if not trial.exists():
        return
    bundle = Path(shutil.make_archive(str(OUT/label), 'zip', root_dir=trial))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    state['runs'][label]['archive'] = dict(path=str(bundle), sha256=file_sha(bundle), bytes=bundle.stat().st_size)
    preserve_state()
    print('ARCHIVE_VERIFIED', label, file_sha(bundle), flush=True)
def native_eval(weights, label):
    detector = YOLO(str(weights))
    metrics = detector.val(data=str(DATA_YAML), split='val', imgsz=1280,
                           batch=8, device=0, workers=2, conf=.001, iou=.7,
                           nms=False, max_det=300, augment=False, plots=True,
                           project=str(OUT/'evaluation'), name=label, exist_ok=False)
    result = dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                  per_class_AP50_95=dict(zip(NAMES, map(float, metrics.box.maps))),
                  protocol=dict(imgsz=1280, conf=.001, iou=.7, nms=False, split='val'),
                  checkpoint_sha256=file_sha(weights))
    del detector; gc.collect(); torch.cuda.empty_cache()
    return result
def train_ddp(config, record):
    """CLI spawns DDP; parent reads persisted epoch history, not child callbacks."""
    config_path = OUT/(config['name']+'_config.yaml')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    log_path = OUT/(config['name']+'_training.log')
    history_path = Path(config['project'])/config['name']/'results.csv'
    launched = time.monotonic()
    record['log'] = str(log_path)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([shutil.which('yolo'), 'detect', 'train', 'cfg='+str(config_path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        previous_epoch = 0
        while process.poll() is None:
            if history_path.is_file():
                with history_path.open(newline='') as stream:
                    history = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
                # Ignore a last row still being written; trainer writes metrics before weights.
                if history and history[-1].get('metrics/mAP50-95(B)'):
                    latest = history[-1]
                    epoch = int(latest['epoch'])
                    fitness = float(latest['metrics/mAP50-95(B)'])
                    if not math.isfinite(fitness) or epoch > config['epochs']:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        raise ValueError('Invalid fitness or epoch cap exceeded')
                    if epoch != previous_epoch:
                        record.update(status='running', epochs_completed=epoch, fitness=fitness,
                                      epoch_metrics=history, seconds=time.monotonic()-launched)
                        previous_epoch = epoch
                        preserve_state()
                        print('PROGRESS', config['name'], 'epoch', epoch, '/', config['epochs'],
                              'mAP50-95', fitness, flush=True)
            if time.monotonic()-launched >= 110*60 or time.monotonic()-started >= 9.5*3600:
                record['budget_stop'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record['status'] = 'budget_stop'
                preserve_state()
                return
            time.sleep(10)
        if process.returncode != 0:
            print(log_path.read_text(encoding='utf-8')[-8000:], flush=True)
            raise RuntimeError('DDP training failed; exit '+str(process.returncode))
    with history_path.open(newline='') as stream:
        history = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    record.update(epochs_completed=len(history), epoch_metrics=history, seconds=time.monotonic()-launched)
    assert 0 < len(history) <= config['epochs']
    print(log_path.read_text(encoding='utf-8')[-3500:], flush=True)
preserve_state()
state['status'] = 'baseline_validation'
preserve_state()
state['baseline_checkpoint07_native'] = native_eval(initial, 'checkpoint07_raw_native')
preserve_state()
# Smoke is separate and never becomes an initializer for E1–E6.
state['status'] = 'smoke_running'; preserve_state()
smoke_config = dict(jobs[0], name='SMOKE_checkpoint07', epochs=1, patience=3)
smoke_started = time.monotonic()
state['smoke'] = dict(status='starting', epochs_completed=0)
train_ddp(smoke_config, state['smoke'])
assert not state['smoke'].get('budget_stop')
state['smoke_seconds'] = time.monotonic()-smoke_started
state['estimated_training_seconds_max'] = state['smoke_seconds']*60
gc.collect(); torch.cuda.empty_cache()
preserve_state()
if state['estimated_training_seconds_max'] > 9.5*3600:
    state['status'] = 'budget_review_required'; preserve_state()
    raise RuntimeError('Measured estimate exceeds ten-hour budget with archive reserve; no matrix jobs launched.')
for config in jobs:
    label = config['name']
    if time.monotonic()-started >= 9.5*3600:
        state['status'] = 'budget_stop_before_next_job'; preserve_state()
        break
    assert file_sha(initial) == BEST_SHA256
    state['status'] = 'training'
    state['current_run'] = label
    state['runs'][label] = dict(status='starting', config=config,
                              initial_sha256=file_sha(initial), epochs_completed=0)
    preserve_state()
    run_started = time.monotonic()
    try:
        record = state['runs'][label]
        train_ddp(config, record)
        best = Path(config['project'])/label/'weights/best.pt'
        assert best.is_file() and record['epochs_completed'] <= 10
        record.update(status='budget_stop' if record.get('budget_stop') else
                      ('early_stop' if record['epochs_completed'] < 10 else 'completed'),
                      checkpoint=str(best), checkpoint_sha256=file_sha(best),
                      seconds=time.monotonic()-run_started)
        archive_trial(label)
        gc.collect(); torch.cuda.empty_cache()
        record['native_validation'] = native_eval(best, label+'_native')
        preserve_state()
        if record.get('budget_stop'):
            state['status'] = 'budget_stop'; preserve_state(); break
    except Exception:
        state['runs'][label].update(status='failed', error=traceback.format_exc())
        state['status'] = 'failed_requires_review'
        preserve_state(); archive_trial(label)
        raise
else:
    state['status'] = 'training_complete_evaluation_followup_required'
    preserve_state()
    print('MATRIX_TRAINING_COMPLETE: native val available; no classroom or independent-test claim.', flush=True)
