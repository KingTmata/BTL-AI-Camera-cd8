import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from kaggle_tools.experiment_campaign import file_sha
from kaggle_tools.group_controls import prepare_group_controls, select_group_disjoint_train_rows


def records():
    rows = []
    for index in range(24):
        rows.append(dict(image_id=f't{index:02}', split='train', source_id='source',
                         source_group_key=f'scene{index}', image_path=f'images/train/t{index:02}.jpg',
                         label_path=f'labels/train/t{index:02}.txt', boxes='1'))
    for split, scene in [('val', 'scene0'), ('test', 'scene1')]:
        rows.append(dict(image_id=split, split=split, source_id='source', source_group_key=scene,
                         image_path=f'images/{split}/{split}.jpg', label_path=f'labels/{split}/{split}.txt',
                         boxes='1'))
    return rows


class GroupControlTests(unittest.TestCase):
    def test_removes_only_training_members_of_either_holdout_group_and_keeps_records_immutable(self):
        rows = records()
        before = copy.deepcopy(rows)
        selected = select_group_disjoint_train_rows(rows)
        self.assertEqual([row['image_id'] for row in selected], [f't{i:02}' for i in range(2, 24)])
        self.assertEqual(rows, before)
        self.assertTrue(all(row['split'] == 'train' for row in selected))

    def test_rejects_missing_groups_duplicate_ids_unknown_roles_and_empty_result(self):
        cases = []
        for field, value in [('source_group_key', ''), ('split', 'validation')]:
            rows = records()
            rows[-1][field] = value
            cases.append(rows)
        rows = records()
        rows[-1]['image_id'] = 't00'
        cases.append(rows)
        rows = records()
        for row in rows:
            row['source_group_key'] = 'all-in-one-scene'
        cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):
                select_group_disjoint_train_rows(rows)

    def prepare_fixture(self, root):
        rows = records()
        for index, row in enumerate(rows):
            image, label = root / row['image_path'], root / row['label_path']
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f'fixture image {index}'.encode())
            label.write_text(f'{index % 8} 0.5 0.5 0.2 0.2\n')
            row.update(image_sha256=file_sha(image), label_sha256=file_sha(label))
        manifest = root / 'manifest.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest, rows

    def test_real_payload_lists_match_counts_keep_all_holdout_files_and_use_original_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.prepare_fixture(root)
            before = {row['label_path']: file_sha(root / row['label_path']) for row in rows}
            names = [f'class{i}' for i in range(8)]
            report = prepare_group_controls(root, manifest, root / 'controls', file_sha(manifest), names)
            clean, random = report['subsets']['group_disjoint'], report['subsets']['random_removal']
            self.assertEqual((clean['train_images'], random['train_images']), (22, 22))
            self.assertEqual(report['removed_train_images'], 2)
            self.assertEqual(clean['shared_holdout_groups'], 0)
            self.assertTrue(all(clean['class_boxes'][name] for name in names))
            config = yaml.safe_load(Path(clean['data_yaml']).read_text())
            self.assertEqual(Path(config['val']), root / 'images/val')
            self.assertNotIn('test', config)
            paths = Path(clean['train_list']).read_text().splitlines()
            self.assertFalse(any(path.endswith(('t00.jpg', 't01.jpg')) for path in paths))
            self.assertEqual(before, {row['label_path']: file_sha(root / row['label_path']) for row in rows})
            self.assertEqual(file_sha(manifest), report['manifest_sha256'])
            self.assertFalse(list((root / 'controls').glob('*.pt')))
            self.assertEqual(json.loads(json.dumps(report))['labels_changed'], False)

    def test_changed_training_payload_is_rejected_before_creating_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.prepare_fixture(root)
            (root / rows[2]['label_path']).write_text('0 0.1 0.1 0.3 0.3\n')
            with self.assertRaises(ValueError):
                prepare_group_controls(root, manifest, root / 'controls', file_sha(manifest),
                                       [f'class{i}' for i in range(8)])
            self.assertFalse((root / 'controls').exists())


if __name__ == '__main__':
    unittest.main()
