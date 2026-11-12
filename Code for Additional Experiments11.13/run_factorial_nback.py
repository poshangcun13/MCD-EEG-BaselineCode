"""Locked 0-back -> {0,2}-back x {recording 1,2} identity benchmark.

Five blocked outer folds; independent source-only validation block. Every fold
trains ONE model and evaluates its unchanged checkpoint in all four cells.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REST = HERE
from experiment_paths import NB_DATA, OUTPUT_ROOT
sys.path.insert(0, str(REST))
import run_resting_benchmark as R
import run_pretrained_benchmark as P

OUT = OUTPUT_ROOT / 'factorial_nback'
NAMES = ['LRMD', 'FBCNet', 'EEGNet', 'EEGViT', 'EEGConformer',
         'EEGPT_pretrained', 'BrainprintNet', 'ATCNet']
CELLS = {'T0S0': (0, 1), 'T1S0': (2, 1),
         'T0S1': (0, 2), 'T1S1': (2, 2)}
Y = np.repeat(np.arange(10), 120)


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def folds():
    """Contiguous blocks of 24 segment indices in every identity.

    The outer test block rotates 1..5. Validation is the next block cyclically;
    the three remaining blocks train. Block indices synchronize all four cells.
    Array order is documented, not assumed to establish true chronological time.
    """
    block = np.tile(np.arange(120) // 24, 10)
    result = []
    for f in range(5):
        test = np.flatnonzero(block == f)
        val = np.flatnonzero(block == (f + 1) % 5)
        train = np.flatnonzero((block != f) & (block != (f + 1) % 5))
        assert len(train) == 720 and len(val) == len(test) == 240
        assert not set(train) & set(val) and not set(train) & set(test)
        assert not set(val) & set(test)
        for ix, n in [(train, 72), (val, 24), (test, 24)]:
            assert np.array_equal(np.bincount(Y[ix]), np.full(10, n))
        result.append((train, val, test))
    assert np.array_equal(np.sort(np.concatenate([f[2] for f in result])), np.arange(1200))
    return result


def protocol():
    return {
        'version': 1,
        'prediction_target': '10-class closed-set participant identity; not N-back load decoding',
        'source': 'recording 1, 0-back only',
        'cells': {c: {'load': v[0], 'recording': v[1]} for c, v in CELLS.items()},
        'acquisition': {'participants': 10, 'channels': 60, 'sampling_rate_hz': 400,
                        'segment_seconds': 2, 'segments_per_identity_cell': 120,
                        'recording_interval_hours': 12},
        'identity_mapping': 'Author-confirmed common participant-axis convention; no external person-ID field in N-back MAT',
        'outer_folds': '5 consecutive blocks of 24 array-order segments per participant; fold f tests block f',
        'inner_validation': 'next source 0-back recording-1 block cyclically; remaining three blocks train',
        'samples_per_fold': {'train': 720, 'validation': 240, 'test_per_cell': 240},
        'test_alignment': 'same held-out participant/segment-axis indices in every cell; index correspondence is not stimulus matching',
        'oof': 'each of 1200 samples per cell tested once by its corresponding outer-fold model',
        'reporting': 'mean +/- sample SD over 5 outer-fold metrics; disjoint segment blocks, same participants; not 5 independent subject cohorts',
        'selection': 'minimum source-only validation cross entropy; delta 1e-5; no test evaluation during training',
        'fitted_normalization': 'training samples only; validation and all test cells transform only',
        'models': NAMES,
        'local_models': 'seven existing local implementations, randomly initialized; no architecture retuning',
        'eegpt': 'publicly pretrained official EEGPT frozen target encoder + source-trained linear probe; NOT prior local EEGPT-named network',
        'eegpt_checkpoint_sha256': '9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0',
        'eegpt_encoder_weights_sha256': P.sha(P.WEIGHTS) if P.WEIGHTS.exists() else None,
        'eegpt_official_commit': 'ea7282f698c406cd423ccbedbe9960853761144c',
        'eegpt_features': 'frozen official encoder; locally generated cache; 58 channels, 400->256 Hz, segment CAR/demean/RMS',
        'eegpt_readout': '2048->16 per patch then 128->10; dropout .5, maxnorm 1/.25; FP32',
        'optimizer': 'AdamW; weight_decay .01; label_smoothing .1; gradient clip 5',
        'learning_rates': {n: {'EEGViT': 3e-4, 'EEGPT_pretrained': 5e-4}.get(n, 1e-3) for n in NAMES},
        'batch_sizes': {n: 8 if n == 'BrainprintNet' else 32 for n in NAMES},
        'epochs_max': 80, 'epochs_min': 20, 'early_stopping_patience': 15,
        'seeds': [4201, 4202, 4203, 4204, 4205],
        'precision': 'local models CUDA BF16 autocast; EEGPT probe FP32',
        'metrics': 'Acc; macro Recall/F1; macro identity-wise interpolated one-vs-rest EER',
        'hyperparameter_search': 'none; fixed before evaluation',
        'limits': ['session changes include cap reapplication, 12-hour elapsed time and other acquisition variation',
                   'no matched no-removal control; no claim of isolated cap-induced effect',
                   'MAT does not contain original stimulus IDs, timestamps, or overlap metadata; array-order blocks are not verified runs'],
        'source_hashes': {str(p): P.sha(p) for p in [Path(__file__), REST / 'resting_models.py',
                                                  REST / 'run_resting_benchmark.py',
                                                  HERE / 'run_pretrained_benchmark.py']},
        'input_hashes': {f'{load}-BACK-{session}.mat': P.sha(NB_DATA / f'{load}-BACK-{session}.mat')
                         for load, session in CELLS.values()},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
    }


def prepare(config):
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    all_hashes = set()
    audits = []
    for cell, (load, session) in CELLS.items():
        path = NB_DATA / f'{load}-BACK-{session}.mat'
        target = cache / f'{cell}_raw.npy'
        if not target.exists():
            a = sio.loadmat(path, variable_names=['X'])['X']
            assert a.shape == (10, 60, 800, 120)
            a = a.transpose(0, 3, 1, 2).reshape(1200, 60, 800).astype('float32')
            assert np.isfinite(a).all()
            np.save(target, a)
            del a
        x = np.load(target, mmap_mode='r')
        assert x.shape == (1200, 60, 800) and np.isfinite(x).all()
        hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in x]
        assert len(set(hashes)) == 1200, f'Within-cell duplicate: {cell}'
        assert not all_hashes & set(hashes), f'Cross-cell duplicate: {cell}'
        all_hashes.update(hashes)
        pd.DataFrame({'cell': cell, 'sample_index': np.arange(1200), 'subject_id': Y + 1,
                      'segment_in_subject': np.tile(np.arange(1, 121), 10),
                      'array_block': np.tile(np.arange(120) // 24 + 1, 10),
                      'sha256_float32': hashes}).to_csv(OUT / f'{cell}_samples.csv', index=False)
        audits.append({'cell': cell, 'input': str(path), 'raw_cache_sha256': P.sha(target),
                       'input_sha256': config['input_hashes'][path.name], 'shape': list(x.shape)})
        print('AUDIT', cell, '1200 unique finite samples', flush=True)
    dump(OUT / 'data_audit.json', {'cells': audits, 'within_and_cross_cell_exact_duplicates': 0,
                                  'sample_count': 4800, 'sha_precision': 'model input float32'})
    membership = []
    for fold, (tr, va, te) in enumerate(folds(), 1):
        for cell in CELLS:
            roles = [('train', tr), ('validation', va), ('test', te)] if cell == 'T0S0' else [('test', te)]
            for role, ix in roles:
                membership.extend({'fold': fold, 'cell': cell, 'role': role,
                                   'sample_index': int(i), 'subject_id': int(Y[i] + 1),
                                   'segment_in_subject': int(i % 120 + 1)} for i in ix)
    pd.DataFrame(membership).to_csv(OUT / 'fold_membership.csv', index=False)


def representations(name, config):
    result = {}
    if name == 'EEGPT_pretrained':
        # Generate the same frozen-encoder representations from the supplied data;
        # the release does not depend on unpublished feature caches or Table 11 runs.
        _, used, indices = P.channels()
        for cell in CELLS:
            folder = OUT / 'cache' / 'EEGPT_pretrained' / cell
            folder.mkdir(parents=True, exist_ok=True)
            raw = np.load(OUT / f'cache/{cell}_raw.npy', mmap_mode='r')
            a = P.features(raw, folder, 'input', indices, used)
            assert a.shape == (1200, 8, 2048) and np.isfinite(a).all()
            result[cell] = a
        return result
    for cell in CELLS:
        raw = np.load(OUT / f'cache/{cell}_raw.npy', mmap_mode='r')
        if name not in ['FBCNet', 'BrainprintNet']:
            result[cell] = raw[:, None]
            continue
        p = OUT / f'cache/{name}_{cell}.npy'
        if not p.exists():
            partial = p.with_suffix('.partial.npy')
            output = None
            for start in range(0, 1200, 64):
                part = R.representation(raw[start:start + 64], name)
                if output is None:
                    output = np.lib.format.open_memmap(partial, mode='w+', dtype='float32',
                                                     shape=(1200, *part.shape[1:]))
                output[start:start + len(part)] = part
            output.flush()
            del output
            partial.replace(p)
            print('REPRESENTATION', name, cell, flush=True)
        a = np.load(p, mmap_mode='r')
        assert np.isfinite(a).all()
        result[cell] = a
    return result


def probabilities(model, x, name, batch):
    if name != 'EEGPT_pretrained':
        return R.scores(model, x, batch)
    # Pure evaluation of the projected checkpoint: no max-norm writes on test data.
    model.eval()
    rows = []
    with torch.inference_mode():
        for b in x.split(128):
            h = nn.functional.linear(b, model.probe1.weight, model.probe1.bias).flatten(1)
            logits = nn.functional.linear(h, model.probe2.weight, model.probe2.bias)
            rows.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(rows)


def validation_probabilities(model, x, name, batch):
    # Match the previously fixed public-probe validation protocol.
    return P.predict(model, x) if name == 'EEGPT_pretrained' else R.scores(model, x, batch)


def train(name, arrays, config, selected_folds):
    folder = OUT / 'models' / name
    folder.mkdir(parents=True, exist_ok=True)
    for fold, (tr, va, te) in enumerate(folds(), 1):
        if fold not in selected_folds:
            continue
        complete = folder / f'fold{fold}_complete.json'
        if complete.exists():
            print('SKIP', name, fold, flush=True)
            continue
        start = time.time()
        R.seed(4200 + fold)
        source = arrays['T0S0']
        axes = (0, 1) if name == 'EEGPT_pretrained' else (0, 3)
        mean = source[tr].mean(axis=axes, keepdims=True, dtype=np.float64).astype('float32')
        std = source[tr].std(axis=axes, keepdims=True, dtype=np.float64).astype('float32').clip(1e-6)
        x = torch.as_tensor((source[tr] - mean) / std, device='cuda')
        v = torch.as_tensor((source[va] - mean) / std, device='cuda')
        labels = torch.as_tensor(Y[tr], device='cuda')
        model = P.LinearProbe(8, 10).cuda() if name == 'EEGPT_pretrained' else R.build(name).cuda()
        model.eval()
        with torch.no_grad():
            model(x[:2])
        model.train()
        parameters = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rates'][name], weight_decay=.01)
        batch = config['batch_sizes'][name]
        indices = torch.arange(len(tr), device='cuda')
        best_loss = float('inf')
        best_epoch = 0
        state = None
        history = []
        print('START', name, fold, 'parameters', parameters, 'train/val/test', 720, 240, 240, flush=True)
        for epoch in range(1, 81):
            model.train()
            total = 0.
            for b in indices[torch.randperm(len(indices), device='cuda')].split(batch):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=name != 'EEGPT_pretrained'):
                    loss = nn.functional.cross_entropy(model(x[b]), labels[b], label_smoothing=.1)
                assert torch.isfinite(loss), f'Nonfinite loss: {name}/{fold}/{epoch}'
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                total += loss.item() * len(b)
            p = validation_probabilities(model, v, name, batch)
            vl = float(-np.log(p[np.arange(len(va)), Y[va]].clip(1e-12)).mean())
            history.append({'epoch': epoch, 'train_loss': total / len(tr),
                            'validation_loss': vl, 'validation_accuracy': float((p.argmax(1) == Y[va]).mean())})
            if vl < best_loss - 1e-5:
                best_loss, best_epoch = vl, epoch
                state = {k: a.detach().cpu().clone() for k, a in model.state_dict().items()}
            if epoch % 10 == 0:
                print('EPOCH', name, fold, epoch, 'validation_loss', round(vl, 5),
                      'best', best_epoch, 'seconds', round(time.time() - start), flush=True)
            if epoch >= 20 and epoch - best_epoch >= 15:
                break
        assert state is not None
        model.load_state_dict(state)
        model.eval()
        pd.DataFrame(history).to_csv(folder / f'fold{fold}_history.csv', index=False)
        np.savez(folder / f'fold{fold}_normalization.npz', mean=mean, std=std)
        checkpoint = folder / f'fold{fold}_model.pt'
        torch.save({'method': name, 'state_dict': state, 'best_epoch': best_epoch,
                    'seed': 4200 + fold, 'train_indices': tr, 'validation_indices': va,
                    'test_indices': te, 'trainable_parameters': parameters,
                    'encoder_weights_sha256': config['eegpt_encoder_weights_sha256'] if name == 'EEGPT_pretrained' else None}, checkpoint)
        # All four cells are first accessed for scoring after checkpoint selection.
        del x, v, labels, opt
        torch.cuda.empty_cache()
        metrics = {}
        for cell in CELLS:
            test = torch.as_tensor((arrays[cell][te] - mean) / std, device='cuda')
            p = probabilities(model, test, name, batch)
            assert p.shape == (240, 10) and np.isfinite(p).all()
            assert np.allclose(p.sum(1), 1, atol=1e-5)
            metrics[cell] = R.metrics(Y[te], p)
            pred = pd.DataFrame({'cell': cell, 'fold': fold, 'sample_index': te,
                                 'subject_id': Y[te] + 1, 'segment_in_subject': te % 120 + 1,
                                 'predicted_subject_id': p.argmax(1) + 1})
            for c in range(10):
                pred[f'p_subject_{c + 1}'] = p[:, c]
            pred.to_csv(folder / f'fold{fold}_{cell}_predictions.csv', index=False)
            del test
        assert all(torch.equal(a.detach().cpu(), state[k]) for k, a in model.state_dict().items()), 'Test scoring mutated model state'
        dump(complete, {'method': name, 'fold': fold, 'metrics': metrics,
                        'checkpoint_sha256': P.sha(checkpoint), 'best_epoch': best_epoch,
                        'best_validation_loss': best_loss, 'epochs_run': epoch,
                        'trainable_parameters': parameters, 'seconds': time.time() - start,
                        'same_unchanged_checkpoint_for_all_cells': True})
        print('DONE', name, fold, {c: round(m['Acc'], 4) for c, m in metrics.items()},
              'seconds', round(time.time() - start), flush=True)
        del model, state
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=NAMES, default=NAMES)
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 6)))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert all(1 <= f <= 5 for f in args.folds)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)
    config = protocol()
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == config, 'Locked protocol changed; do not mix experiments'
    else:
        dump(path, config)
        snapshot = OUT / 'code_snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in config['source_hashes']:
            shutil.copy2(source, snapshot / Path(source).name)
    prepare(config)
    if args.prepare_only:
        print('PREPARATION COMPLETE', flush=True)
        return
    for name in args.models:
        arrays = representations(name, config)
        train(name, arrays, config, args.folds)
        del arrays
        gc.collect()
    print('REQUESTED FACTORIAL RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
