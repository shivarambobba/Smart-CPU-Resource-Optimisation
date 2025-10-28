#!/usr/bin/env python3
"""Quick model sanity check: load regressor checkpoint and run a forward + embed."""
import json
import traceback
from pathlib import Path
import importlib.util
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_EVALS = Path(ROOT) / 'experiments' / 'run_evals.py'


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location('run_evals_mod', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out = {'ok': False, 'errors': []}
    try:
        mod = load_module_from_path(RUN_EVALS)
        SmallRegressor = getattr(mod, 'SmallRegressor')
        make_synthetic_sequences = getattr(mod, 'make_synthetic_sequences')

        model = SmallRegressor()
        ckpt_path = Path(ROOT) / 'models' / 'transformer_lstm.pth'
        out['ckpt_path'] = str(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        ck = torch.load(str(ckpt_path), map_location='cpu')
        # ck may be a state_dict or a saved model object
        try:
            if isinstance(ck, dict) and ('state_dict' in ck):
                sd = ck['state_dict']
            elif isinstance(ck, dict) and any(k.startswith('input_proj') or k.startswith('transformer') for k in ck.keys()):
                sd = ck
            else:
                sd = None

            if sd is not None:
                model.load_state_dict(sd)
                out['load_method'] = 'state_dict'
            else:
                # try assigning the loaded object as model
                try:
                    model = ck
                    out['load_method'] = 'model_object'
                except Exception:
                    raise RuntimeError('Unknown checkpoint format and failed to load')
        except Exception as e:
            out['errors'].append('load_state_error: ' + repr(e))
            # continue and try to use ck directly

        model.eval()

        X, Y = make_synthetic_sequences(n=3, seq_len=10, seed=42)
        sample = X[0:1]
        with torch.no_grad():
            inp = torch.from_numpy(sample.astype('float32'))
            pred = model(inp)
        out['prediction'] = {
            'value': float(pred.detach().cpu().numpy().ravel()[0]),
            'shape': list(pred.shape),
        }

        # test embed
        emb = None
        try:
            e = model.embed(X[0])
            emb = np.array(e).astype(float)
            out['embed'] = {'shape': list(emb.shape), 'first_vals': emb.ravel()[:5].tolist()}
        except Exception as e:
            out['embed_error'] = repr(e)

        out['ok'] = True
    except Exception as e:
        out['errors'].append(traceback.format_exc())

    # write report
    (Path(ROOT) / 'experiments').mkdir(exist_ok=True)
    with open(Path(ROOT) / 'experiments' / 'model_check.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', Path(ROOT) / 'experiments' / 'model_check.json')


if __name__ == '__main__':
    main()
