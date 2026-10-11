"""Evaluation helpers used by the reproducible external-validation script."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score,
)


def discrimination_metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "AUPRC_baseline": float(np.mean(y)),
    }


def threshold_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
        "predicted_positive_fraction": float(np.mean(pred)),
        "mean_probability_true_positive": float(np.mean(p[y == 1])),
        "mean_probability_true_negative": float(np.mean(p[y == 0])),
        "median_probability_true_positive": float(np.median(p[y == 1])),
        "median_probability_true_negative": float(np.median(p[y == 0])),
    }


def bootstrap_discrimination_ci(y, p, n_boot=2000, seed=42):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        pp = p[idx]
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp))
    return {
        "valid_bootstraps": len(aucs),
        "AUROC_mean": float(np.mean(aucs)),
        "AUROC_2.5%": float(np.percentile(aucs, 2.5)),
        "AUROC_97.5%": float(np.percentile(aucs, 97.5)),
        "AUPRC_mean": float(np.mean(aps)),
        "AUPRC_2.5%": float(np.percentile(aps, 2.5)),
        "AUPRC_97.5%": float(np.percentile(aps, 97.5)),
    }
