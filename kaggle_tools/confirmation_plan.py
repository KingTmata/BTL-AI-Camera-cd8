"""Budget-matched follow-up recipes; no training or model selection on test."""
from copy import deepcopy
from pathlib import Path

from kaggle_tools.experiment_campaign import rank_results


def build_confirmation_plan(state, output, group_controls=None):
    if state['status'] != 'batch_complete':
        raise ValueError('Wait for the main batch')
    complete = {name: run for name, run in state['runs'].items() if run['status'] == 'complete'}
    measured = {name: run['measurements']['native'] for name, run in complete.items()}
    ranked = rank_results(measured)
    for record in measured.values():
        if record['manifest_sha256'] != state['manifest_sha256'] or record['protocol'] != state['protocol']:
            raise ValueError('Full validation identity differs')
    nano = next((name for name in ranked if name.startswith('R') and not name.startswith('R12_')
                 and 'config' in complete[name]), None)
    if nano is None:
        raise ValueError('No measured nano configuration')
    winner = complete[nano]
    base = deepcopy(winner['config'])
    output = Path(output)
    jobs = []

    def add(label, model, epochs, requires=None, **changes):
        config = dict(deepcopy(base), project=(output / 'train').as_posix(), name=label,
                      model=Path(model).as_posix(), epochs=epochs, resume=False, exist_ok=False)
        config.update(changes)
        jobs.append(dict(label=label, requires=requires, config=config))

    # Repeated seeds reproduce the winning recipe from its original initializer.
    for seed in (43, 44):
        add('C_seed' + str(seed), base['model'], base['epochs'], seed=seed)
    add('C_longer_seed42', base['model'], max(8, 2 * base['epochs']), seed=42)

    teacher = next((name for name in ranked if name.startswith('R12_')), None)
    if group_controls is not None:
        controls = group_controls
        if (controls['manifest_sha256'] != state['manifest_sha256'] or
                controls['validation_images'] != 2500 or controls['validation_boxes'] != 8359 or
                controls['labels_changed'] or controls['holdout_roles_changed'] or
                controls['test_predictions_used']):
            raise ValueError('Group controls must preserve ground truth and holdout roles')
        expected_count = controls['original_train_images'] - controls['removed_train_images']
        if not 0 < controls['removed_train_images'] < controls['original_train_images']:
            raise ValueError('Invalid group removal budget')
        if (controls['subsets']['group_disjoint']['shared_holdout_groups'] != 0 or
                any(controls['subsets'][name]['train_images'] != expected_count
                    for name in ('group_disjoint', 'random_removal'))):
            raise ValueError('Known group overlap or unmatched random removal budget')
        if teacher is None:
            raise ValueError('No measured pretrained recipe for fresh group controls')
    if teacher is not None:
        # This control starts from COCO pretrained nano, matching the small recipe.
        config = dict(deepcopy(complete[teacher]['config']), model='yolo26n.pt',
                      project=(output / 'train').as_posix(), name='C_fresh_nano_control',
                      resume=False, exist_ok=False)
        jobs.append(dict(label='C_fresh_nano_control', requires=None, config=config,
                         initialization_kind='official_coco_pretrained'))
        if group_controls is not None:
            for label, subset in [('C_group_disjoint_nano', 'group_disjoint'),
                                  ('C_random_removal_nano', 'random_removal')]:
                selection = deepcopy(group_controls['subsets'][subset])
                control = dict(deepcopy(config), name=label, data=selection['data_yaml'])
                jobs.append(dict(label=label, requires=None, config=control,
                    initialization_kind='official_coco_pretrained', train_selection=selection))

    # Both arms reset optimizer at the same stage boundary. Only stage1 freeze differs.
    for stem, frozen in [('F_control', None), ('F_freeze', 10)]:
        add(stem + '_stage1', winner['checkpoint'], 2, freeze=frozen, seed=42)
        add(stem + '_stage2', output / 'train' / (stem + '_stage1') / 'weights/last.pt',
            4, requires=stem + '_stage1', freeze=None, seed=42, lr0=min(base['lr0'], .0001))

    teacher_better = teacher is not None and measured[teacher]['mAP50'] > measured[nano]['mAP50']
    if teacher_better:
        for label, source in [('D_student_control', None), ('D_distill', complete[teacher]['checkpoint'])]:
            add(label, winner['checkpoint'], 3, freeze=None, seed=42, batch=8,
                lr0=min(base['lr0'], .0001), distill_model=source)
    return dict(status='prepared_not_trained', winning_nano=nano, teacher=teacher,
                teacher_better_native=teacher_better, jobs=jobs,
                total_epochs=sum(job['config']['epochs'] for job in jobs),
                manifest_sha256=state['manifest_sha256'], protocol=state['protocol'],
                group_controls=deepcopy(group_controls),
                note='Seeds use original init; freeze/KD use selected nano. Teacher/student comparison is practical, not architecture-only. Longer schedule differs from short schedule.')
