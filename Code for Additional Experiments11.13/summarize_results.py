"""Summarize generated Table 6, 9, and 11 metrics without plotting or training.

Usage: python summarize_results.py [--output summary.csv]
Table 8 is summarized by benchmark_metrics.py.
"""
import argparse
import json
import numpy as np
import pandas as pd
from experiment_paths import OUTPUT_ROOT


def collect():
    rows = []
    settings = [('Table6', OUTPUT_ROOT / 'resting_state', 'resting_state'),
                ('Table6', OUTPUT_ROOT / 'nback', 'nback'),
                ('Table11', OUTPUT_ROOT / 'pretrained_eegpt', 'pretrained')]
    for table, root, kind in settings:
        for first in sorted(root.glob('*')):
            if not first.is_dir():
                continue
            folders = sorted(first.glob('*')) if kind == 'nback' else [first]
            for folder in folders:
                if not folder.is_dir():
                    continue
                files = [folder / f'fold{fold}_metrics.json' for fold in range(1, 6)]
                if not all(p.exists() for p in files):
                    continue
                values = [json.loads(p.read_text(encoding='utf-8')) for p in files]
                row = {'table': table, 'condition': first.name if kind == 'nback' else
                       ('resting_state' if kind == 'resting_state' else first.name),
                       'model': 'EEGPT_pretrained' if kind == 'pretrained' else folder.name}
                for metric in ['Acc', 'EER', 'EER_macro', 'EER_pooled', 'Recall', 'F1']:
                    if all(metric in v for v in values):
                        a = [v[metric] for v in values]
                        row[metric + '_mean'] = float(np.mean(a))
                        row[metric + '_sd'] = float(np.std(a, ddof=1))
                rows.append(row)
    for folder in sorted((OUTPUT_ROOT / 'factorial_nback' / 'models').glob('*')):
        files = [folder / f'fold{fold}_complete.json' for fold in range(1, 6)]
        if not all(p.exists() for p in files):
            continue
        values = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        for cell in ['T0S0', 'T1S0', 'T0S1', 'T1S1']:
            row = {'table': 'Table9', 'condition': cell, 'model': folder.name}
            for metric in ['Acc', 'EER', 'Recall', 'F1']:
                a = [v['metrics'][cell][metric] for v in values]
                row[metric + '_mean'] = float(np.mean(a))
                row[metric + '_sd'] = float(np.std(a, ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional CSV destination.')
    args = parser.parse_args()
    table = collect()
    if args.output:
        table.to_csv(args.output, index=False)
    print(table.to_string(index=False) if len(table) else 'No complete five-fold results found.')
