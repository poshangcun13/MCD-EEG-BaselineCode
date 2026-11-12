"""Table 8: the reported two-member EEGNet-lite probability ensemble.

Run after neural_stage.py, or use run_cognitive_benchmark.py. Only the second
member is trained here; the first is the saved EEGNet-lite model. The selected
learning rate/epoch count comes from training-side validation of the first
member. No third ensemble member or pre-existing local cache is required.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from neural_stage import StageNet, train, infer
from stage_classification import splits, dump
from experiment_paths import OUTPUT_ROOT

ROOT = OUTPUT_ROOT / 'cognitive_stage'


def run():
    source = ROOT / 'neural' / 'combined'
    dest = ROOT / 'eight_model_results' / 'EEGNet-lite-Ens2'
    dest.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / 'neural' / 'sample_index.csv')
    base = pd.read_csv(source / 'oof_predictions.csv')
    pd.testing.assert_frame_equal(base[meta.columns], meta)
    config = pd.read_csv(source / 'fold_metrics.csv')
    x = torch.tensor(np.load(ROOT / 'neural' / 'signals.npy')[:, None], device='cuda')
    y = torch.tensor(meta.stage.to_numpy(), dtype=torch.long, device='cuda')
    probability = np.full(len(meta), np.nan)
    for fold, (tr, te) in enumerate(splits(meta, 5, 42), 1):
        conf = config.loc[config.fold == fold].iloc[0]
        assert np.array_equal(np.flatnonzero(base.fold.to_numpy() == fold), te)
        checkpoint_path = dest / f'fold{fold}_seed{1300 + fold}.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert checkpoint['epochs'] == int(conf.selected_epochs)
            assert checkpoint['lr'] == float(conf.learning_rate)
            assert checkpoint['train_subjects'] == sorted(meta.subject.iloc[tr].unique())
            model = StageNet().cuda()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model, _, _, _ = train(x, y, tr, None, float(conf.learning_rate),
                                    int(conf.selected_epochs), 1300 + fold, False)
            torch.save({'state_dict': model.cpu().state_dict(),
                        'epochs': int(conf.selected_epochs),
                        'lr': float(conf.learning_rate), 'seed': 1300 + fold,
                        'train_subjects': sorted(meta.subject.iloc[tr].unique())}, checkpoint_path)
            model.cuda()
        probability[te] = (expit(base.decision_score.iloc[te].to_numpy())
                           + expit(infer(model, x[te]))) / 2
        del model
    assert np.isfinite(probability).all()
    meta.assign(fold=base.fold, prediction=(probability >= .5).astype(int),
                decision_score=probability).to_csv(dest / 'oof_predictions.csv', index=False)
    pd.read_csv(source / 'fold_subjects.csv').to_csv(dest / 'fold_subjects.csv', index=False)
    dump(dest / 'provenance.json', {'members': 2, 'seeds': '300+fold and 1300+fold',
                                  'weights': [.5, .5], 'threshold': .5,
                                  'selection': 'training-side validation of the first member only'})
    print('Two-member ensemble complete', flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(2)
    run()
