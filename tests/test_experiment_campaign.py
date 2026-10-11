import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kaggle_tools.experiment_campaign import select_epoch, rank_results, verified_bundle


class CampaignTests(unittest.TestCase):
    def test_selects_ap50_epoch_instead_of_ap5095_epoch(self):
        rows = [
            {'epoch': '1', 'metrics/mAP50(B)': '.78', 'metrics/mAP50-95(B)': '.51'},
            {'epoch': '2', 'metrics/mAP50(B)': '.76', 'metrics/mAP50-95(B)': '.55'},
        ]
        self.assertEqual(select_epoch(rows, 2), 1)

    def test_rejects_incomplete_duplicate_and_nonfinite_epoch_records(self):
        row = {'epoch': '1', 'metrics/mAP50(B)': '.78', 'metrics/mAP50-95(B)': '.51'}
        for rows, expected in [([row], 2), ([row, row], 2), ([dict(row, **{'metrics/mAP50(B)': 'nan'})], 1)]:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    select_epoch(rows, expected)

    def test_ranking_rejects_different_ground_truth_or_protocol(self):
        baseline = {'manifest_sha256': 'same', 'protocol': {'imgsz': 1280}, 'mAP50': .76}
        better = dict(baseline, mAP50=.79)
        self.assertEqual(rank_results({'base': baseline, 'new': better}), ['new', 'base'])
        for changed in [dict(better, manifest_sha256='different'), dict(better, protocol={'imgsz': 640})]:
            with self.assertRaises(ValueError):
                rank_results({'base': baseline, 'new': changed})

    def test_bundle_contains_verified_payload_and_rejects_external_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / 'run'
            root.mkdir()
            payload = root / 'selected.pt'
            payload.write_bytes(b'checkpoint bytes')
            result = root / 'metrics.json'
            result.write_text('{"mAP50": 0.78}', encoding='utf-8')
            destination = directory / 'result.zip'
            digest = verified_bundle(root, [payload, result], destination)
            self.assertEqual(digest, hashlib.sha256(destination.read_bytes()).hexdigest())
            with zipfile.ZipFile(destination) as archive:
                manifest = json.loads(archive.read('FILES.json'))
                self.assertEqual(set(manifest), {'selected.pt', 'metrics.json'})
                for name, metadata in manifest.items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), metadata['sha256'])
            outsider = directory / 'external.pt'
            outsider.write_bytes(b'outside')
            with self.assertRaises(ValueError):
                verified_bundle(root, [outsider], destination)
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
