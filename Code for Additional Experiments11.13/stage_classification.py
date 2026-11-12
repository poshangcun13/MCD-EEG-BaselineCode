"""Shared participant folds and metrics for the reported cognitive-stage experiment."""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def splits(meta, n, seed):
    """Stratify distinct subjects by population; then expand to sample indices."""
    subjects = meta[['subject', 'population']].drop_duplicates().reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    result = []
    for tr, te in cv.split(subjects.subject, subjects.population):
        train = np.flatnonzero(meta.subject.isin(subjects.subject.iloc[tr]))
        test = np.flatnonzero(meta.subject.isin(subjects.subject.iloc[te]))
        assert not set(meta.subject.iloc[train]) & set(meta.subject.iloc[test])
        assert set(meta.stage.iloc[train]) == set(meta.stage.iloc[test]) == {0, 1}
        result.append((train, test))
    return result

def metrics(y, pred, score):
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'roc_auc': float(roc_auc_score(y, score)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
