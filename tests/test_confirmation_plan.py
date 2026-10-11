import copy
import unittest

from kaggle_tools.confirmation_plan import build_confirmation_plan


def campaign(teacher_score=.79):
    protocol = dict(data='/data/val.yaml', imgsz=1280, split='val')
    state = dict(status='batch_complete', manifest_sha256='gt', parent_sha256='initial',
                 protocol=protocol, runs={})
    for name, score in [('R04_class_weight_025', .77), ('R12_small_pretrained', teacher_score)]:
        checkpoint = '/train/' + name + '/weights/epoch3.pt'
        config = dict(model='/initial.pt', data='/data/train.yaml', epochs=4, batch=32,
                      lr0=.0005, seed=42, freeze=None, cls_pw=.25, optimizer='MuSGD')
        if name.startswith('R12_'):
            config.update(model='yolo26s.pt', epochs=10, batch=8, lr0=.001, cls_pw=0.,
                          warmup_epochs=1., cos_lr=True, lrf=.1)
        state['runs'][name] = dict(status='complete', checkpoint=checkpoint,
            checkpoint_sha256=name, config=config,
            measurements=dict(native=dict(mAP50=score, mAP50_95=.54, protocol=protocol,
                                          manifest_sha256='gt', checkpoint_sha256=name)))
    return state


class ConfirmationPlanTests(unittest.TestCase):
    def test_group_controls_use_fresh_nano_same_recipe_and_unchanged_evaluation(self):
        state = campaign()
        controls = dict(manifest_sha256='gt', original_train_images=12000,
            removed_train_images=255, validation_images=2500, validation_boxes=8359,
            labels_changed=False, holdout_roles_changed=False, test_predictions_used=False,
            subsets=dict(group_disjoint=dict(data_yaml='/controls/groups.yaml', train_images=11745,
                shared_holdout_groups=0), random_removal=dict(data_yaml='/controls/random.yaml',
                train_images=11745, shared_holdout_groups=17)))
        before = copy.deepcopy((state, controls))
        plan = build_confirmation_plan(state, '/followup', group_controls=controls)
        jobs = {job['label']: job for job in plan['jobs']}
        baseline = jobs['C_fresh_nano_control']['config']
        for label, subset in [('C_group_disjoint_nano', 'group_disjoint'),
                              ('C_random_removal_nano', 'random_removal')]:
            job = jobs[label]
            config = copy.deepcopy(job['config'])
            self.assertEqual(config.pop('data'), controls['subsets'][subset]['data_yaml'])
            config.pop('name')
            reference = {key: value for key, value in baseline.items() if key not in {'data', 'name'}}
            self.assertEqual(config, reference)
            self.assertEqual(config['model'], 'yolo26n.pt')
            self.assertEqual(job['initialization_kind'], 'official_coco_pretrained')
        self.assertEqual(plan['protocol'], state['protocol'])
        self.assertEqual((state, controls), before)

    def test_rejects_group_controls_with_changed_gt_or_remaining_known_group_overlap(self):
        controls = dict(manifest_sha256='gt', original_train_images=12000,
            removed_train_images=255, validation_images=2500, validation_boxes=8359,
            labels_changed=False, holdout_roles_changed=False, test_predictions_used=False,
            subsets=dict(group_disjoint=dict(data_yaml='/controls/groups.yaml', train_images=11745,
                shared_holdout_groups=0), random_removal=dict(data_yaml='/controls/random.yaml',
                train_images=11745, shared_holdout_groups=17)))
        cases = []
        for field, value in [('manifest_sha256', 'other'), ('validation_boxes', 100),
                             ('labels_changed', True), ('test_predictions_used', True)]:
            broken = copy.deepcopy(controls)
            broken[field] = value
            cases.append(broken)
        for subset, field, value in [('group_disjoint', 'shared_holdout_groups', 1),
                                      ('random_removal', 'train_images', 11000)]:
            broken = copy.deepcopy(controls)
            broken['subsets'][subset][field] = value
            cases.append(broken)
        for broken in cases:
            with self.assertRaises(ValueError):
                build_confirmation_plan(campaign(), '/followup', group_controls=broken)

    def test_fresh_nano_matches_small_pretrained_training_recipe(self):
        state = campaign()
        teacher = copy.deepcopy(state['runs']['R12_small_pretrained']['config'])
        plan = build_confirmation_plan(state, '/followup')
        job = next(job for job in plan['jobs'] if job['label'] == 'C_fresh_nano_control')
        actual = copy.deepcopy(job['config'])
        self.assertEqual(actual.pop('model'), 'yolo26n.pt')
        actual.pop('name')
        actual.pop('project')
        actual.pop('resume')
        actual.pop('exist_ok')
        teacher.pop('model')
        self.assertEqual(actual, teacher)

    def test_baseline_reference_cannot_supply_a_missing_training_recipe(self):
        state = campaign()
        reference = copy.deepcopy(state['runs']['R04_class_weight_025'])
        reference.pop('config')
        reference['measurements']['native']['mAP50'] = .90
        state['runs']['BASELINE'] = reference
        self.assertEqual(build_confirmation_plan(state, '/followup')['winning_nano'],
                         'R04_class_weight_025')

    def test_seed_repeats_use_original_init_and_preserve_winning_recipe(self):
        state = campaign()
        before = copy.deepcopy(state)
        plan = build_confirmation_plan(state, '/followup')
        for label, seed in [('C_seed43', 43), ('C_seed44', 44)]:
            cfg = next(job['config'] for job in plan['jobs'] if job['label'] == label)
            self.assertEqual((cfg['model'], cfg['seed'], cfg['epochs'], cfg['cls_pw']),
                             ('/initial.pt', seed, 4, .25))
        self.assertEqual(state, before)

    def test_freeze_pair_has_equal_steps_and_same_stage_two_schedule(self):
        plan = build_confirmation_plan(campaign(), '/followup')
        jobs = {job['label']: job for job in plan['jobs']}
        for suffix in ['stage1', 'stage2']:
            a, b = jobs['F_control_' + suffix], jobs['F_freeze_' + suffix]
            self.assertEqual(a['config']['epochs'], b['config']['epochs'])
            self.assertEqual(a['config']['batch'], b['config']['batch'])
            self.assertEqual(a['config']['lr0'], b['config']['lr0'])
        self.assertEqual(jobs['F_freeze_stage1']['config']['freeze'], 10)
        self.assertIsNone(jobs['F_freeze_stage2']['config']['freeze'])
        for stem in ['F_control', 'F_freeze']:
            self.assertEqual(jobs[stem + '_stage2']['requires'], stem + '_stage1')
            self.assertTrue(jobs[stem + '_stage2']['config']['model'].endswith(stem + '_stage1/weights/last.pt'))

    def test_distillation_is_conditional_and_has_matching_student_control(self):
        weak = build_confirmation_plan(campaign(.76), '/followup')
        self.assertFalse(any(job['label'].startswith('D_') for job in weak['jobs']))
        strong = build_confirmation_plan(campaign(.79), '/followup')
        jobs = {job['label']: job['config'] for job in strong['jobs']}
        control, kd = copy.deepcopy(jobs['D_student_control']), copy.deepcopy(jobs['D_distill'])
        self.assertIsNone(control.pop('distill_model'))
        self.assertTrue(kd.pop('distill_model').endswith('R12_small_pretrained/weights/epoch3.pt'))
        control.pop('name')
        kd.pop('name')
        self.assertEqual(control, kd)

    def test_rejects_incomplete_batch_or_mismatched_gt_and_protocol(self):
        for field, value in [('manifest_sha256', 'different'), ('protocol', dict(imgsz=640))]:
            state = campaign()
            state['runs']['R12_small_pretrained']['measurements']['native'][field] = value
            with self.assertRaises(ValueError):
                build_confirmation_plan(state, '/followup')
        state = campaign()
        state['status'] = 'running'
        with self.assertRaises(ValueError):
            build_confirmation_plan(state, '/followup')


if __name__ == '__main__':
    unittest.main()
