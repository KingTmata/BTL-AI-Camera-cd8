import unittest

from kaggle_tools.training_subsets import select_train_rows


def row(name, source='source_a', split='train'):
    return dict(image_id=name, source_id=source, split=split,
                image_path=f'images/{split}/{name}.jpg', label_path=f'labels/{split}/{name}.txt',
                image_sha256='a' * 64, label_sha256='b' * 64)


class TrainingSubsetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row('a'), row('b', 'source_b'), row('c', 'source_b'),
                     row('v', split='val'), row('t', split='test')]

    def test_excludes_only_train_sources_and_preserves_input_manifest(self):
        selected = select_train_rows(self.rows, excluded_sources=('source_b',))
        self.assertEqual([item['image_id'] for item in selected], ['a'])
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(self.rows[-2]['split'], 'val')

    def test_random_control_is_reproducible_same_count_and_train_only(self):
        selected = select_train_rows(self.rows, random_remove=2, seed=42)
        repeated = select_train_rows(list(reversed(self.rows)), random_remove=2, seed=42)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 1)
        self.assertTrue(all(item['split'] == 'train' for item in selected))

    def test_rejects_path_escape_or_mismatched_split_and_duplicate_ids(self):
        for path in ('../outside.jpg', 'C:/outside.jpg', 'images/val/a.jpg', 'images/train/../a.jpg'):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    select_train_rows([dict(row('a'), image_path=path)])
        with self.assertRaises(ValueError):
            select_train_rows([row('a'), row('a')])

    def test_rejects_unknown_exclusions_holdout_ids_and_empty_subsets(self):
        for kwargs in (dict(excluded_sources=('missing',)), dict(excluded_ids=('v',)),
                       dict(random_remove=3), dict(excluded_sources=('source_a', 'source_b'))):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    select_train_rows(self.rows, **kwargs)


if __name__ == '__main__':
    unittest.main()
