#!/usr/bin/env python3
"""Reproduce 5-fold training for the DeepH-Immune 7,092 allele-balanced experiment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from deeph_immune.io import load_dataset_bundle
from deeph_immune.model import build_deeph_immune_v2_shortcut_resistant
from deeph_immune.paths import default_training_csv, default_training_feature_dir
from deeph_immune.training_utils import (
    allele_label_weights, build_splits, combine_weights, length_label_weights,
)


def categorical_prior_oof(df, y, splits, column):
    """Diagnostic only: how well can one metadata field predict the label?"""
    pred = np.zeros(len(df), dtype=np.float32)
    for tr, va in splits:
        train = df.iloc[tr]
        global_mean = float(train["Immunogenicity"].mean())
        mapping = train.groupby(column)["Immunogenicity"].mean()
        pred[va] = df.iloc[va][column].map(mapping).fillna(global_mean).to_numpy(dtype=np.float32)
    return {
        "feature": column,
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=default_training_csv())
    p.add_argument("--feature-dir", type=Path, default=default_training_feature_dir())
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "training_reproduction")
    p.add_argument("--split-mode", choices=["stratified_length", "peptide_group", "allele_group"], default="stratified_length")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-length-weights", action="store_true")
    p.add_argument("--no-allele-weights", action="store_true")
    p.add_argument("--single-class-allele-weight", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    model_out = out / "models"; model_out.mkdir(exist_ok=True)

    f = args.feature_dir
    bundle = load_dataset_bundle(
        args.csv,
        f / "ESM2_peptide_training_Immuscope_7092.npy",
        f / "ESM2_mhc_training_Immuscope_7092.npy",
        f / "Labels_training_Immuscope_7092.npy",
        f / "Sample_ids_training_Immuscope_7092.npy",
    )
    df, y = bundle.dataframe, bundle.labels
    lengths = df["Peptide_length"].to_numpy(dtype=int)
    splits = build_splits(df, y, args.split_mode, args.folds, args.seed)

    tf.keras.utils.set_random_seed(args.seed)
    print(f"Samples: {len(df):,} | labels: {Counter(y)} | alleles: {df['MHC_allele'].nunique()}")
    print("Split mode:", args.split_mode)

    # These baselines are intentionally simple. If allele identity alone performs
    # well, that is a warning that the model could exploit label prevalence by allele.
    prior_rows = []
    for col in ["MHC_allele", "Species", "Source"]:
        if col in df.columns:
            row = categorical_prior_oof(df, y, splits, col)
            prior_rows.append(row)
            print(f"{col}-only prior: AUROC={row['auroc']:.4f}, AUPRC={row['auprc']:.4f}")
    pd.DataFrame(prior_rows).to_csv(out / "shortcut_prior_diagnostics.csv", index=False)
    pd.crosstab(df["MHC_allele"], df["Immunogenicity"]).to_csv(out / "allele_label_distribution.csv")

    fold_rows, pred_rows = [], []
    Xall = bundle.model_inputs

    for fold, (tr, va) in enumerate(splits, start=1):
        print()
        print(f"========== Fold {fold}/{args.folds} ==========")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + fold)
        model, _ = build_deeph_immune_v2_shortcut_resistant()

        Xtr = [x[tr] for x in Xall]; Xva = [x[va] for x in Xall]
        ytr, yva = y[tr], y[va]

        length_w = None if args.no_length_weights else length_label_weights(ytr, lengths[tr])
        allele_w = None if args.no_allele_weights else allele_label_weights(
            df.iloc[tr]["MHC_allele"].to_numpy(), ytr,
            single_class_weight=args.single_class_allele_weight,
        )
        sample_weight = combine_weights(length_w, allele_w)

        checkpoint = model_out / f"DeepH_Immune_Immuscope_7092_shortcut_resistant_fold{fold}.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=12, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auprc", mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_auprc", mode="max", save_best_only=True, verbose=1),
        ]

        history = model.fit(
            Xtr, ytr, validation_data=(Xva, yva), sample_weight=sample_weight,
            epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
            callbacks=callbacks, verbose=2,
        )
        p = model.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
        auroc = roc_auc_score(yva, p); auprc = average_precision_score(yva, p)
        best_epoch = int(np.argmax(history.history["val_auprc"]) + 1)
        print(f"Fold {fold}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, best_epoch={best_epoch}")

        fold_rows.append({"fold": fold, "n_train": len(tr), "n_val": len(va), "auroc": auroc, "auprc": auprc, "best_epoch": best_epoch, "model_path": str(checkpoint)})
        for li, gi in enumerate(va):
            r = df.iloc[gi]
            pred_rows.append({
                "sample_id": r["sample_id"], "fold": fold, "Species": r["Species"],
                "Source": r.get("Source", ""), "MHC_allele": r["MHC_allele"], "Peptide": r["Peptide"],
                "Peptide_length": int(r["Peptide_length"]), "y_true": int(yva[li]), "y_score": float(p[li]),
            })
        pd.DataFrame(history.history).to_csv(out / f"history_fold{fold}.csv", index=False)

    fold_df = pd.DataFrame(fold_rows); pred_df = pd.DataFrame(pred_rows)
    fold_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    pred_df.to_csv(out / "cv_out_of_fold_predictions.csv", index=False)
    summary = {
        "n": len(df), "split_mode": args.split_mode, "folds": args.folds,
        "length_aware_weights": not args.no_length_weights,
        "allele_label_weights": not args.no_allele_weights,
        "mean_auroc": float(fold_df.auroc.mean()), "std_auroc": float(fold_df.auroc.std(ddof=0)),
        "mean_auprc": float(fold_df.auprc.mean()), "std_auprc": float(fold_df.auprc.std(ddof=0)),
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("Final CV summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
