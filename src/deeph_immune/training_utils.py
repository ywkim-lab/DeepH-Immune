"""Cross-validation and sample-weight utilities from the 7,092-sample experiment."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def length_bin(lengths):
    lengths = np.asarray(lengths, dtype=int)
    out = np.empty(lengths.shape, dtype=object)
    out[lengths <= 11] = "9-11"
    out[(lengths >= 12) & (lengths <= 15)] = "12-15"
    out[(lengths >= 16) & (lengths <= 20)] = "16-20"
    out[lengths >= 21] = "21-25"
    return out


def make_strata(y, lengths):
    bins = length_bin(lengths)
    return np.asarray([f"y{int(label)}_{b}" for label, b in zip(y, bins)])


def length_label_weights(y_train, lengths_train, min_w=0.5, max_w=3.0):
    strata = make_strata(y_train, lengths_train)
    counts = Counter(strata)
    median_count = float(np.median(list(counts.values())))
    w = np.asarray([np.sqrt(median_count / counts[s]) for s in strata], dtype=np.float32)
    w = np.clip(w, min_w, max_w)
    return w / w.mean()


def allele_label_weights(alleles, y, single_class_weight=0.25, min_w=0.25, max_w=3.0):
    """Balance label contribution inside an allele without synthesizing samples."""
    tmp = pd.DataFrame({"allele": np.asarray(alleles).astype(str), "y": np.asarray(y).astype(int)})
    counts = tmp.groupby(["allele", "y"]).size().unstack(fill_value=0)
    weights = np.ones(len(tmp), dtype=np.float32)
    for allele, idx in tmp.groupby("allele").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        n0 = int(counts.loc[allele].get(0, 0)); n1 = int(counts.loc[allele].get(1, 0))
        if n0 > 0 and n1 > 0:
            total = n0 + n1
            w0 = np.sqrt(total / (2.0 * n0)); w1 = np.sqrt(total / (2.0 * n1))
            local_y = tmp.loc[idx, "y"].to_numpy()
            weights[idx] = np.where(local_y == 0, w0, w1)
        else:
            weights[idx] = single_class_weight
    weights = np.clip(weights, min_w, max_w)
    return weights / weights.mean()


def combine_weights(length_w, allele_w, min_w=0.25, max_w=4.0):
    if length_w is None and allele_w is None:
        return None
    if length_w is None:
        w = allele_w.copy()
    elif allele_w is None:
        w = length_w.copy()
    else:
        w = length_w * allele_w
    w = np.clip(w, min_w, max_w)
    return (w / w.mean()).astype(np.float32)


def build_splits(df, y, split_mode="stratified_length", folds=5, seed=42):
    strata = make_strata(y, df["Peptide_length"].to_numpy(dtype=int))
    if split_mode == "stratified_length":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(df)), strata))
    if split_mode == "peptide_group":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(df)), strata, groups=df["Peptide"].astype(str)))
    if split_mode == "allele_group":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(df)), strata, groups=df["MHC_allele"].astype(str)))
    raise ValueError(f"Unknown split mode: {split_mode}")
