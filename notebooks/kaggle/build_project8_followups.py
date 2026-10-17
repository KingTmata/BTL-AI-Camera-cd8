"""Rebuild portable Kaggle cells from current sources, logging provenance only to MD."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
CAMPAIGN = '/kaggle/working/BTL-AI-Camera-v05-campaign-20261010/batch_20261010_001917'
JOBS = {
    'inference_followup': dict(
        sources=('src/classroom_ai/experiments/experiment_campaign.py', 'src/classroom_ai/inference/variants.py',
                 'src/classroom_ai/experiments/validation_replay.py', 'src/classroom_ai/experiments/view_replay.py',
                 'notebooks/kaggle/project8_v05_inference_followup_20261010.py'),
        variable='INFERENCE_FOLLOWUP_SOURCE',
        entry=f"run_inference_followup('{CAMPAIGN}', DATA_YAML, MANIFEST_SHA256)"),
    'nms_sweep': dict(
        sources=('src/classroom_ai/experiments/experiment_campaign.py', 'src/classroom_ai/inference/variants.py',
                 'src/classroom_ai/experiments/validation_replay.py', 'notebooks/kaggle/project8_v05_nms_sweep_20261010.py'),
        variable='NMS_SWEEP_SOURCE',
        entry=f"run_nms_sweep('{CAMPAIGN}', MANIFEST_SHA256)"),
    'source_ablations_train': dict(
        sources=('src/classroom_ai/experiments/experiment_campaign.py', 'src/classroom_ai/inference/variants.py',
                 'src/classroom_ai/experiments/training_subsets.py',
                 'notebooks/kaggle/project8_v05_prepare_source_ablations_20261010.py',
                 'notebooks/kaggle/project8_v05_source_ablations_train_20261010.py'),
        variable='SOURCE_ABLATION_TRAIN_SOURCE',
        entry=f"run_source_ablations('{CAMPAIGN}', DATA_DIR, SOURCE_DATA / 'manifest.csv', DATA_YAML, MANIFEST_SHA256)"),
}
GUARDS = """
assert PREPARED['dataset_manifest_sha256'] == MANIFEST_SHA256
assert len([row for row in rows if row['split'] == 'val']) == 2500
assert not (DATA_DIR / 'images/test').exists()
assert ultralytics.__version__ == '8.4.165'
"""
def digest(payload):
    return hashlib.sha256(payload).hexdigest()
def build_followups(root):
    root = Path(root).resolve()
    log = root / 'reports/training/project8_v05_log.md'
    if not log.is_file():
        raise FileNotFoundError('Existing campaign Markdown log is required')
    packages, records = {}, {}
    # Validate every source before writing any regenerated cell.
    for name, config in JOBS.items():
        pieces, source_hashes = [], {}
        for relative in config['sources']:
            path = root / relative
            payload = path.read_bytes()
            source_hashes[relative] = digest(payload)
            source = payload.decode('utf-8').replace('\r\n', '\n')
            source = source.replace('from classroom_ai.inference.variants import merge_detections\n', '')
            source = source.replace('from classroom_ai.inference.variants import EndToEndViews, restore_boxes\n', '')
            source = source.replace('from classroom_ai.experiments.validation_replay import CacheDetectionValidator\n', '')
            pieces.append(source)
        combined = '\n\n'.join(pieces) + GUARDS + config['entry'] + '\n'
        tree = ast.parse(combined)
        if any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('classroom_ai.experiments')
               for node in ast.walk(tree)):
            raise ValueError('Embedded source still depends on a local classroom_ai.experiments import')
        payload = combined.encode('utf-8')
        packed = base64.b64encode(zlib.compress(payload, 9)).decode('ascii')
        assert zlib.decompress(base64.b64decode(packed)) == payload
        variable = config['variable']
        source_hash = digest(payload)
        cell = (f'# {name}: run only after the verified main batch completes.\n'
                'import base64, hashlib, zlib\n'
                f"{variable} = zlib.decompress(base64.b64decode('{packed}')).decode('utf-8')\n"
                f"assert hashlib.sha256({variable}.encode('utf-8')).hexdigest() == '{source_hash}'\n"
                f"exec(compile({variable}, '{name}_20261010.py', 'exec'))\n")
        ast.parse(cell)
        stem = f'notebooks/kaggle/project8_v05_{name}'
        cell_path, notebook_path = stem + '_cell_20261010.py', stem + '_20261010.ipynb'
        notebook = dict(nbformat=4, nbformat_minor=5,
            metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
            cells=[dict(cell_type='markdown', id=name + '-notes', metadata={}, source=[
                f'# Project8 v05 — {name}\n',
                'Prepared locally; GPU execution is not claimed. Paste the code cell into the verified campaign kernel after its main batch completes. Dataset identity, complete batch state and runtime preflights remain enforced. Test predictions are not used.\n']),
                dict(cell_type='code', id=name + '-run', metadata={}, execution_count=None,
                     outputs=[], source=cell.splitlines(keepends=True))])
        notebook_text = json.dumps(notebook, ensure_ascii=False, indent=1)
        packages[name] = (cell_path, cell, notebook_path, notebook_text)
        records[name] = dict(execution_status='prepared_not_GPU_executed',
            compilation='PASS', compression_roundtrip='PASS', sources=source_hashes,
            embedded_source_sha256=source_hash, embedded_source_bytes=len(payload),
            cell=cell_path, cell_sha256=digest(cell.encode()),
            notebook=notebook_path, notebook_sha256=digest(notebook_text.encode()))
    for cell_path, cell, notebook_path, notebook_text in packages.values():
        (root / cell_path).write_text(cell, encoding='utf-8', newline='\n')
        (root / notebook_path).write_text(notebook_text, encoding='utf-8', newline='\n')
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n\n### GPU follow-up cells — rebuilt without deleted archive dependencies\n\n```json\n'
                     + json.dumps(records, indent=2) + '\n```\n')
    return records
if __name__ == '__main__':
    records = build_followups(Path(__file__).resolve().parents[2])
    print(json.dumps({name: dict(cell=value['cell'], cell_sha256=value['cell_sha256'],
        embedded_source_sha256=value['embedded_source_sha256']) for name, value in records.items()}, indent=2))
