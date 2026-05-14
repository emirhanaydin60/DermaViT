import glob
import torch
from pathlib import Path

pths = sorted([p for p in glob.glob('results/**/*.pth', recursive=True)])
if not pths:
    print('No pth files found')
    raise SystemExit(0)

for p in pths:
    print('FILE:', p)
    try:
        ckpt = torch.load(p, map_location='cpu')
        print('  type:', type(ckpt))
        if isinstance(ckpt, dict):
            print('  keys:', list(ckpt.keys()))
            if 'model_state_dict' in ckpt:
                print('    model_state_dict keys sample:', list(ckpt['model_state_dict'].keys())[:10])
        else:
            print('  not dict; obj repr:', repr(ckpt)[:200])
    except Exception as e:
        print('  load error:', e)
