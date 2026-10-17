"""Replay class-aware NMS thresholds on cached native validation predictions.

Only run in the verified campaign kernel after the training batch completes.
Models, GT and the evaluator stay identical; NMS IoU is validation tuning.
"""
import datetime
import gc
import json
import traceback
from pathlib import Path
import torch
from ultralytics import YOLO
def run_nms_sweep(campaign_path, expected_manifest):
    campaign = Path(campaign_path)
    state = json.loads((campaign / 'comparison.json').read_text())
    prepared = json.loads((campaign / 'prepared.json').read_text())
    if state['status'] != 'batch_complete' or state['manifest_sha256'] != expected_manifest:
        raise ValueError('Expected completed, verified campaign')
    measurements = {name: run['measurements']['extra_nms_06']
                    for name, run in state['runs'].items() if run['status'] == 'complete'}
    selected = rank_results(measurements)[:3]
    output = campaign / ('nms_sweep_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    protocol = dict(state['protocol'], plots=False)
    report = dict(status='running', manifest_sha256=expected_manifest, protocol=state['protocol'],
                  selected=selected, results={}, errors={},
                  selection_note='NMS thresholds chosen on validation; no independent test claim.',
                  full_visual_review_complete=False)
    if 'NMS_SWEEP_SOURCE' in globals():
        (output / 'executed_source.py').write_text(NMS_SWEEP_SOURCE)
    def preserve(label):
        (output / 'comparison.json').write_text(json.dumps(report, indent=2))
        archive = campaign.parent / (output.name + '_' + label + '.zip')
        digest = verified_bundle(output, [path for path in output.rglob('*') if path.is_file()], archive)
        print('NMS_SWEEP_VERIFIED_ARCHIVE', archive, digest, flush=True)
    def metric_record(metrics):
        return dict(mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                    precision=float(metrics.box.mp), recall=float(metrics.box.mr))
    for label in selected:
        reference = measurements[label]
        checkpoint = reference['checkpoint']
        if file_sha(checkpoint) != reference['checkpoint_sha256']:
            raise ValueError('Selected checkpoint bytes differ')
        try:
            holder = []
            def factory(**kwargs):
                validator = CacheDetectionValidator(**kwargs)
                holder.append(validator)
                return validator
            model = YOLO(checkpoint)
            if [model.names[i] for i in range(8)] != prepared['classes'] or not model.model.end2end:
                raise ValueError('Ordered classes or native head differ')
            native = model.val(validator=factory, project=str(output / 'cache_evaluation'), name=label,
                               exist_ok=False, **protocol)
            if len(holder) != 1 or len(holder[0].cached_samples) != 2500:
                raise ValueError('Incomplete validation prediction cache')
            samples, names = holder[0].cached_samples, dict(model.names)
            if sum(len(sample['target_cls']) for sample in samples) != 8359:
                raise ValueError('Ground truth box count differs')
            cache = output / (label + '_cache.pt')
            torch.save(dict(samples=samples, names=names, checkpoint_sha256=reference['checkpoint_sha256'],
                            manifest_sha256=expected_manifest, protocol=state['protocol']), cache)
            record = dict(checkpoint_sha256=reference['checkpoint_sha256'], cache_sha256=file_sha(cache),
                          native=metric_record(native), replay={}, image_count=2500, target_boxes=8359)
            native_reference = state['runs'][label]['measurements']['native']
            replayed_native = metric_record(replay_metrics(samples, names, device='cuda:0'))
            replayed_nms = metric_record(replay_metrics(samples, names, iou=.6, device='cuda:0'))
            for key in replayed_native:
                if abs(record['native'][key] - native_reference[key]) > 1e-6:
                    raise ValueError('Fresh native/reference parity failed: ' + key)
                if abs(replayed_native[key] - record['native'][key]) > 1e-6:
                    raise ValueError('Cache/native parity failed: ' + key)
                if abs(replayed_nms[key] - reference[key]) > 1e-6:
                    raise ValueError('Cache/NMS0.6 parity failed: ' + key)
            record['parity'] = dict(native=True, extra_nms_06=True)
            del model, native, holder
            gc.collect()
            torch.cuda.empty_cache()
            for threshold in (.3, .4, .5, .6, .7, .8):
                result = replay_metrics(samples, names, iou=threshold, device='cuda:0')
                result_record = dict(metric_record(result), per_class={names[int(cls)]: dict(
                    AP50=float(result.box.ap50[j]), AP50_95=float(result.box.ap[j]))
                    for j, cls in enumerate(result.box.ap_class_index)})
                record['replay'][str(threshold)] = result_record
                print('NMS_SWEEP_METRIC', label, threshold, json.dumps(result_record), flush=True)
            report['results'][label] = record
            (output / (label + '.json')).write_text(json.dumps(record, indent=2))
            del samples
        except Exception:
            report['errors'][label] = traceback.format_exc()
            print('NMS_SWEEP_ERROR', label, report['errors'][label], flush=True)
        preserve(label)
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'complete_with_errors' if report['errors'] else 'complete'
    preserve('FINAL')
    print('NMS_SWEEP_DONE', json.dumps(report), flush=True)
    return report
