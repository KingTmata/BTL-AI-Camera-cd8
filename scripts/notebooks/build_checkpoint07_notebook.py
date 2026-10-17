"""Reuse checksum-verified archive staging, replacing the historical campaign."""
import copy
import json
from pathlib import Path
from classroom_ai.paths import PROJECT_ROOT as root
source = json.loads((root/'notebooks/kaggle/project8_v05_campaign_20261010.ipynb').read_text(encoding='utf-8'))
cells = copy.deepcopy(source['cells'][:5])
header = '''# Checkpoint 07 — YOLO26n Project8 v0.4, Kaggle 08/10/2026, best epoch 19
Six new sequential method experiments, max 10 epochs each, patience=3.
Initializer SHA256: d6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb.
07 is the Streamlit dropdown index, not campaign R07 v0.5.
User authorized unreviewed existing v0.5 data on 12/10/2026.
No classroom-domain or independent-test claims. No deployment promotion.
'''
cells[0]['source'] = header.splitlines(True)
config = ''.join(cells[1]['source'])
config = config[:config.index('BASE_AUG =')]
config = config.replace('# 1. Ba pilot don gian: moi luot 2 epoch, cung BEST v0.5', '# 1. Six method experiments; user authorized launch')
config = config.replace('START_TRAINING = False', 'START_TRAINING = True')
config = config.replace('EPOCHS = 2', 'EPOCHS = 10').replace('PATIENCE = 0', 'PATIENCE = 3')
config = config.replace('CLOSE_MOSAIC = 0', 'CLOSE_MOSAIC = 2')
config = config.replace('BTL-AI-Camera-v05-campaign-20261010', 'checkpoint07-v04-method-matrix-20261012')
config = config.replace('92adfeb2516861a6958e8adc4a15cf764faa3956078da1ebc81db1fbb7014e31',
                        'd6fa68a79abd7eebfc91724c683d2cfd0c2ace6f6623a019fbef44729bf47adb')
cells[1]['source'] = config.splitlines(True)
stage = ''.join(cells[3]['source']).replace('BEST v0.5', 'checkpoint 07 Project8 v0.4 best epoch 19')
cells[3]['source'] = stage.splitlines(True)
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []; cell['execution_count'] = None
runner = (root/'src/classroom_ai/experiments/checkpoint07_matrix.py').read_text(encoding='utf-8') + '\n' + (
    root/'notebooks/kaggle/checkpoint07_runner_20261012.py').read_text(encoding='utf-8')
cells.append(dict(cell_type='code', metadata={}, source=runner.splitlines(True), outputs=[], execution_count=None))
for index, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'checkpoint07_cell_{index}', 'exec')
notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), cells=cells)
destination = root/'notebooks/kaggle/classroom07_matrix_e10_p3.ipynb'
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print(destination)
