"""Read-only source verification; write separate train lists, never edit labels.

Requires select_train_rows/file_sha embedded or imported by the caller. This only
prepares data; it does not train or mark any annotations reviewed.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml


def prepare_source_ablations(data_root, manifest_path, output, expected_manifest, names):
    data_root, manifest_path, output = Path(data_root), Path(manifest_path), Path(output)
    if file_sha(manifest_path) != expected_manifest:
        raise ValueError('Manifest bytes differ from verified v05 release')
    with manifest_path.open(encoding='utf-8-sig', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    train = select_train_rows(manifest)
    if len(train) != 12000 or sum(row['split'] == 'val' for row in manifest) != 2500:
        raise ValueError('Unexpected v05 split counts')
    counts = {}
    for row in train:
        image, label = data_root / row['image_path'], data_root / row['label_path']
        if not image.resolve().is_relative_to(data_root.resolve()) or not label.resolve().is_relative_to(data_root.resolve()):
            raise ValueError('Payload resolves outside the verified dataset')
        if file_sha(image) != row['image_sha256'] or file_sha(label) != row['label_sha256']:
            raise ValueError('Training payload changed: ' + row['image_id'])
        objects = Counter()
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError('Malformed detection label')
            cls, cx, cy, w, h = values
            if cls != int(cls) or not 0 <= cls < len(names) or not all(0 <= value <= 1 for value in (cx, cy, w, h)) or w <= 0 or h <= 0:
                raise ValueError('Invalid detection class or box')
            objects[int(cls)] += 1
        if sum(objects.values()) != int(row['boxes']):
            raise ValueError('Manifest box count differs')
        counts[row['image_id']] = objects
    source_counts = Counter(row['source_id'] for row in train)
    merged, celulares = source_counts['merged_3k_local'], source_counts['celulares']
    if (merged, celulares) != (2426, 772):
        raise ValueError('Source identities/counts differ from the inspected release')
    variations = {
        'D01_no_merged3k': dict(excluded_sources=('merged_3k_local',)),
        'D02_random_drop_2426': dict(random_remove=merged, seed=42),
        'D03_no_celulares': dict(excluded_sources=('celulares',)),
        'D04_random_drop_772': dict(random_remove=celulares, seed=42),
        'D05_no_both_sources': dict(excluded_sources=('merged_3k_local', 'celulares')),
        'D06_random_drop_3198': dict(random_remove=merged + celulares, seed=42),
    }
    selections = {name: select_train_rows(manifest, **config) for name, config in variations.items()}
    # A canonical release fingerprint excludes machine-specific absolute paths.
    holdout = [dict(image_id=row['image_id'], image_sha256=row['image_sha256'], label_sha256=row['label_sha256'])
               for row in sorted(manifest, key=lambda row: row['image_id']) if row['split'] == 'val']
    val_fingerprint = hashlib.sha256(json.dumps(holdout, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='prepared_not_trained', manifest_sha256=expected_manifest,
                  validation_fingerprint_sha256=val_fingerprint, validation_images=2500,
                  full_visual_review_complete=False, labels_changed=False,
                  note='Source removal is an exploratory ablation, not proof every source image is wrong. Random controls match image counts; sample counts/steps vary across pairs.', subsets={})
    for name, selected in selections.items():
        objects = sum((counts[row['image_id']] for row in selected), Counter())
        if any(objects[cls] == 0 for cls in range(len(names))):
            raise ValueError('Training subset loses a class')
        identity = [dict(image_id=row['image_id'], image_sha256=row['image_sha256'], label_sha256=row['label_sha256']) for row in selected]
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        train_list = output / (name + '_train.txt')
        train_list.write_text('\n'.join((data_root / row['image_path']).resolve().as_posix() for row in selected) + '\n', encoding='utf-8')
        config = dict(path=data_root.resolve().as_posix(), train=train_list.resolve().as_posix(),
                      val=(data_root / 'images/val').resolve().as_posix(), nc=len(names), names=dict(enumerate(names)))
        config_path = output / (name + '.yaml')
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        report['subsets'][name] = dict(train_images=len(selected), train_boxes=sum(objects.values()),
                                      class_boxes={names[cls]: objects[cls] for cls in range(len(names))},
                                      source_counts=dict(Counter(row['source_id'] for row in selected)),
                                      train_identity_sha256=identity_hash, train_list=str(train_list),
                                      train_list_sha256=file_sha(train_list), data_yaml=str(config_path),
                                      selection=variations[name])
        (output / (name + '_identity.json')).write_text(json.dumps(identity, indent=2))
        print('SOURCE_ABLATION_PREPARED', name, len(selected), sum(objects.values()), identity_hash, flush=True)
    (output / 'prepared.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
