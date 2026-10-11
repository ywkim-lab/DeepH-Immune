#!/usr/bin/env python3
"""Reproduce validation of the 5 supplied fold models on ImmuScope external2 (N=154)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from deeph_immune.io import load_dataset_bundle
from deeph_immune.metrics import bootstrap_discrimination_ci, discrimination_metrics, threshold_metrics
from deeph_immune.model import build_deeph_immune_v2_shortcut_resistant
from deeph_immune.paths import default_external_csv, default_external_feature_dir, default_model_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=default_external_csv())
    p.add_argument("--feature-dir", type=Path, default=default_external_feature_dir())
    p.add_argument("--model-dir", type=Path, default=default_model_dir())
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "external_validation")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_models(model_dir):
    models = list(Path(model_dir).glob("DeepH_Immune_Immuscope_7092_shortcut_resistant_fold*.keras"))
    models.sort(key=lambda p: int(re.search(r"fold(\d+)", p.name).group(1)))
    if len(models) != 5:
        raise ValueError(f"Expected exactly 5 fold checkpoints in {model_dir}; found {len(models)}")
    return models


def main():
    args = parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)

    f = args.feature_dir
    bundle = load_dataset_bundle(
        args.csv,
        f / "ESM2_peptide_DeepH_Immune_CEDAR_EXternal_309.npy",
        f / "ESM2_mhc_DeepH_Immune_CEDAR_EXternal_309.npy",
        f / "Labels_DeepH_Immune_CEDAR_EXternal_309.npy",
        f / "Sample_ids_DeepH_Immune_CEDAR_EXternal_309.npy",
    )
    y = bundle.labels
    print(f"External/test cohort: N={len(y)}, positive={(y==1).sum()}, negative={(y==0).sum()}, prevalence={y.mean():.4f}")

    fold_predictions, rows = [], []
    for fold, checkpoint in enumerate(find_models(args.model_dir), start=1):
        # Rebuild the exact architecture and load only learned weights. This avoids
        # deserialization problems across Keras versions while preserving predictions.
        tf.keras.backend.clear_session()
        model, _ = build_deeph_immune_v2_shortcut_resistant()
        model.load_weights(checkpoint)
        p = model.predict(bundle.model_inputs, batch_size=args.batch_size, verbose=0).reshape(-1)
        fold_predictions.append(p)
        rows.append({"Model": f"Fold {fold}", "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)})
        print(f"Fold {fold}: AUROC={rows[-1]['AUROC']:.4f}, AUPRC={rows[-1]['AUPRC']:.4f}")
        del model

    fold_predictions = np.column_stack(fold_predictions)
    ensemble = fold_predictions.mean(axis=1)
    disc = discrimination_metrics(y, ensemble)
    cls = threshold_metrics(y, ensemble, args.threshold)
    ci = bootstrap_discrimination_ci(y, ensemble, args.bootstrap, args.seed)
    rows.append({"Model": "5-fold ensemble", "AUROC": disc["AUROC"], "AUPRC": disc["AUPRC"]})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out / "per_fold_and_ensemble_metrics.csv", index=False)

    pred_df = bundle.dataframe.copy()
    for j in range(5):
        pred_df[f"fold{j+1}_probability"] = fold_predictions[:, j]
    pred_df["ensemble_probability"] = ensemble
    pred_df["ensemble_prediction"] = (ensemble >= args.threshold).astype(int)
    pred_df.to_csv(out / "external2_154_predictions.csv", index=False)

    summary = {
        "dataset": args.csv.name,
        "n": int(len(y)), "positive": int((y==1).sum()), "negative": int((y==0).sum()),
        "discrimination": disc, "classification_at_threshold": cls,
        "bootstrap_95CI": ci,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ROC
    fpr, tpr, _ = roc_curve(y, ensemble)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, lw=2, label=f"5-fold ensemble (AUROC={disc['AUROC']:.3f})")
    ax.plot([0,1],[0,1], "--", lw=1, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ImmuScope external2 validation (N=154)")
    ax.legend(loc="lower right", frameon=False); fig.tight_layout()
    fig.savefig(figdir / "ROC_ensemble.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y, ensemble)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(recall, precision, lw=2, label=f"AUPRC={disc['AUPRC']:.3f}")
    ax.axhline(y.mean(), ls="--", lw=1, label=f"Baseline={y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="ImmuScope external2 validation (N=154)")
    ax.legend(loc="lower left", frameon=False); fig.tight_layout()
    fig.savefig(figdir / "PR_ensemble.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    print()
    print("5-fold ensemble")
    print(f"AUROC={disc['AUROC']:.4f} | AUPRC={disc['AUPRC']:.4f} | baseline={disc['AUPRC_baseline']:.4f}")
    print(f"Threshold={args.threshold:.2f} | sensitivity={cls['sensitivity']:.4f} | specificity={cls['specificity']:.4f} | MCC={cls['MCC']:.4f}")
    print("Saved outputs to:", out)


if __name__ == "__main__":
    main()
