"""Table 8: reproduce all eight reported cognitive-stage methods from MAT inputs.

Usage: python run_cognitive_benchmark.py [--models EEGNet-lite EEGNet-SE ...]
Set MCD_EEG_DATA_ROOT before starting Python; see experiment_paths.py.
Each model uses the same five participant-disjoint folds, mixed-cohort training,
and separate hearing/deaf test reporting. Existing complete models are reused.
"""
import argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import neural_stage as N
import run_eight_benchmark as B
from stage_classification import dump
from experiment_paths import OUTPUT_ROOT

MODELS = ['EEGNet-lite', 'EEGNet-SE', 'MS-EEGNet-SF', 'ShallowLogVar',
          'EEG-TCN', 'EEGConformer-lite', 'FBCNet-lite', 'EEGNet-lite-Ens2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA is required for the reported training configuration.'
    torch.set_num_threads(2)
    N.OUT.mkdir(parents=True, exist_ok=True)
    B.OUT.mkdir(parents=True, exist_ok=True)
    signals = N.OUT / 'signals.npy'
    if signals.exists() and (N.OUT / 'sample_index.csv').exists():
        N.check_sources()
        raw = np.load(signals)
    else:
        raw = N.extract()
    meta = pd.read_csv(N.OUT / 'sample_index.csv')
    assert raw.shape == (len(meta), 60, 400)
    dump(B.OUT / 'plan.json', {'models': MODELS, 'train_population': 'combined',
                              'test_reporting': 'hearing/deaf subsets of held-out participants',
                              'outer_folds': 5, 'outer_seed': 42,
                              'max_epochs': 50, 'learning_rates': [.001, .0003],
                              'validation_subject_fraction': .25,
                              'selection': 'inner validation accuracy, then refit outer training subjects'})
    if any(name in args.models for name in ['EEGNet-lite', 'EEGNet-lite-Ens2']):
        source = N.OUT / 'combined'
        if not (source / 'oof_predictions.csv').exists():
            N.run(raw, meta, 'combined', 50)
        dest = B.OUT / 'EEGNet-lite'
        dest.mkdir(exist_ok=True)
        for name in ['oof_predictions.csv', 'fold_subjects.csv']:
            pd.read_csv(source / name).to_csv(dest / name, index=False)
    others = [name for name in args.models if name not in ['EEGNet-lite', 'EEGNet-lite-Ens2']]
    if others:
        x = torch.tensor(raw[:, None], device='cuda')
        y = torch.tensor(meta.stage.to_numpy(), dtype=torch.long, device='cuda')
        for name in others:
            B.run(name, x, y, meta, SimpleNamespace(epochs=50))
        del x, y
        torch.cuda.empty_cache()
    if 'EEGNet-lite-Ens2' in args.models:
        from run_stage_ensemble import run
        run()
    from benchmark_metrics import collect
    collect()


if __name__ == '__main__':
    main()
