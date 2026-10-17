"""Measure all native-cache pairs/triple at six NMS thresholds on full val."""
import datetime
import itertools
import json
import shutil
import traceback
from pathlib import Path
import torch
import ultralytics
from classroom_ai.experiments.cache_ensemble import ensemble_cached_predictions
from classroom_ai.experiments.experiment_campaign import file_sha, rank_results, verified_bundle
from classroom_ai.experiments.validation_replay import replay_metrics
root = Path.cwd()
reports = root / 'reports/campaign_20261010'
first_run = reports / 'local_inference_20261009_180723'
second_run = reports / 'local_R02_inference_20261009_183043'
assert (reports / (second_run.name + '_R02.zip')).is_file(), 'R02 native/sweep not complete yet'
assert ultralytics.__version__ == '8.4.165'
torch.set_num_threads(2)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
output = reports / ('local_cache_ensemble_' + stamp)
output.mkdir(exist_ok=False)
shutil.copy2(Path(__file__), output / 'executed_source.py')
for relative in ('src/classroom_ai/experiments/cache_ensemble.py', 'src/classroom_ai/experiments/experiment_campaign.py',
                 'src/classroom_ai/inference/variants.py', 'src/classroom_ai/experiments/validation_replay.py',
                 'tests/unit/test_cache_ensemble.py'):
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative, destination)
first_state = json.loads((first_run / 'comparison.json').read_text())
second_state = json.loads((second_run / 'comparison.json').read_text())
assert first_state['protocol'] == second_state['protocol']
protocol = first_state['protocol']
manifest_sha = '22a0140ab5e4d454d7e0c46ab24145c6d52d8b949db67ab8142fdd3bb8bb9ccd'
caches, provenance = {}, {}
single_results = {}
for label, folder, source_state in [('BASELINE', first_run, first_state),
                                     ('R01', first_run, first_state), ('R02', second_run, second_state)]:
    cache_path = folder / (label + '_cache.pt')
    native = source_state['results'][label + '_native']
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    assert cache['manifest_sha256'] == native['manifest_sha256'] == manifest_sha
    assert cache['protocol'] == native['protocol'] == protocol
    assert cache['checkpoint_sha256'] == native['checkpoint_sha256'] == file_sha(native['checkpoint'])
    assert len(cache['samples']) == 2500
    assert sum(len(sample['target_cls']) for sample in cache['samples']) == 8359
    assert len(cache['names']) == 8
    for threshold, suffix in [(None, '_native'), (.6, '_nms_0.6')]:
        metric = replay_metrics(cache['samples'], cache['names'], iou=threshold)
        reference = source_state['results'][label + suffix]
        for key, value in [('mAP50', metric.box.map50), ('mAP50_95', metric.box.map),
                           ('precision', metric.box.mp), ('recall', metric.box.mr)]:
            assert abs(reference[key] - value) < 1e-10, (label, suffix, key)
    caches[label] = cache
    provenance[label] = dict(cache=str(cache_path), cache_sha256=file_sha(cache_path),
                             checkpoint=native['checkpoint'], checkpoint_sha256=cache['checkpoint_sha256'],
                             native_and_nms_06_parity=True)
    single_results.update({key: record for key, record in source_state['results'].items()
                           if key.startswith(label + '_')})
state = dict(status='running', validation_images=2500, validation_boxes=8359,
             manifest_sha256=manifest_sha, protocol=protocol, test_used=False,
             full_visual_review_complete=False, provenance=provenance, results={}, errors={},
             runtime=dict(ultralytics=ultralytics.__version__, torch=str(torch.__version__),
                          device='CPU', torch_threads=2),
             note='Actual cached native predictions, identical batch-canvas GT. CPU FP32 only; inference latency not measured.')
def preserve():
    (output / 'comparison.json').write_text(json.dumps(state, indent=2))
    digest = verified_bundle(output, [p for p in output.rglob('*') if p.is_file()],
                             reports / (output.name + '.zip'))
    print('CACHE_ENSEMBLE_VERIFIED_ARCHIVE', digest, flush=True)
preserve()
best_score = -1.
for size in (2, 3):
    for labels in itertools.combinations(caches, size):
        for threshold in (.3, .4, .5, .6, .7, .8):
            key = '+'.join(labels) + '_nms_' + str(threshold)
            try:
                samples = ensemble_cached_predictions([caches[label] for label in labels], iou=threshold)
                metrics = replay_metrics(samples, caches[labels[0]]['names'])
                assert sum(metrics.nt_per_class) == 8359
                record = dict(manifest_sha256=manifest_sha, protocol=protocol,
                              models=labels, extra_nms_iou=threshold, mAP50=float(metrics.box.map50),
                              mAP50_95=float(metrics.box.map), precision=float(metrics.box.mp),
                              recall=float(metrics.box.mr), per_class={metrics.names[int(cls)]:
                              dict(AP50=float(metrics.box.ap50[index]), AP50_95=float(metrics.box.ap[index]))
                              for index, cls in enumerate(metrics.box.ap_class_index)})
                state['results'][key] = record
                (output / (key + '.json')).write_text(json.dumps(record, indent=2))
                print('CACHE_ENSEMBLE_METRIC', key, json.dumps(record), flush=True)
                if record['mAP50'] > best_score:
                    best_score = record['mAP50']
                    state['best_ensemble'] = key
                    torch.save(dict(samples=samples, names=caches[labels[0]]['names'], record=record),
                               output / 'best_ensemble_predictions.pt')
                del samples, metrics
            except Exception:
                state['errors'][key] = traceback.format_exc()
                print('CACHE_ENSEMBLE_ERROR', key, state['errors'][key], flush=True)
            (output / 'comparison.json').write_text(json.dumps(state, indent=2))
        preserve()
state['status'] = 'complete_with_errors' if state['errors'] else 'complete'
state['ranking'] = rank_results(dict(single_results, **state['results']))
state['single_results'] = single_results
preserve()
print('CACHE_ENSEMBLE_DONE', output, state['best_ensemble'], best_score, flush=True)
