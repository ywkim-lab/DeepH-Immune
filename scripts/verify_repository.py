#!/usr/bin/env python3
"""Fast integrity check for a freshly cloned DeepH-Immune repository."""
from pathlib import Path
import sys, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def need(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    train_csv = need(ROOT / "data/training/ImmuScope_7092_trainable_fullMHC281_allelebalanced.csv")
    ext_csv = need(ROOT / "data/external/ImmuScope_external2_154_trainable_fullMHC281.csv")
    train = pd.read_csv(train_csv, low_memory=False)
    ext = pd.read_csv(ext_csv, low_memory=False)

    assert len(train) == 7092
    assert train["Immunogenicity"].value_counts().to_dict() == {0: 3546, 1: 3546}
    assert train["MHC_allele"].nunique() == 46
    assert len(ext) == 154
    assert int((ext.Immunogenicity == 1).sum()) == 202
    assert int((ext.Immunogenicity == 0).sum()) == 107
    assert train["MHC_aligned_281"].astype(str).str.len().eq(281).all()
    assert ext["MHC_aligned_281"].astype(str).str.len().eq(281).all()

    checks = [
        (ROOT/"data/training/features/ESM2_peptide_training_Immuscope_7092.npy", (7092,1280)),
        (ROOT/"data/training/features/ESM2_mhc_training_Immuscope_7092.npy", (7092,1280)),
        (ROOT/"data/training/features/Labels_training_Immuscope_7092.npy", (7092,)),
        (ROOT/"data/training/features/Sample_ids_training_Immuscope_7092.npy", (7092,)),
        (ROOT/"data/external/features/ESM2_peptide_DeepH_Immune_CEDAR_EXternal_309.npy", (309,1280)),
        (ROOT/"data/external/features/ESM2_mhc_DeepH_Immune_CEDAR_EXternal_309.npy", (309,1280)),
        (ROOT/"data/external/features/Labels_DeepH_Immune_CEDAR_EXternal_309.npy", (309,)),
        (ROOT/"data/external/features/Sample_ids_DeepH_Immune_CEDAR_EXternal_309.npy", (309,)),
    ]
    for path, shape in checks:
        need(path)
        arr = np.load(path, allow_pickle=True)
        assert arr.shape == shape, (path, arr.shape, shape)

    # Row-order alignment is critical for ESM features.
    assert np.array_equal(
        train.sample_id.astype(str).to_numpy(),
        np.load(checks[3][0], allow_pickle=True).astype(str),
    )
    assert np.array_equal(
        ext.sample_id.astype(str).to_numpy(),
        np.load(checks[7][0], allow_pickle=True).astype(str),
    )

    models = sorted((ROOT/"models/pretrained").glob("*.keras"))
    assert len(models) == 5
    for p in models:
        assert zipfile.is_zipfile(p), f"Invalid Keras archive: {p}"
        with zipfile.ZipFile(p) as z:
            assert "model.weights.h5" in z.namelist()

    print("Repository verification: PASS")
    print("  training: 7092 (3546 positive / 3546 negative), 46 alleles")
    print("  external2: 309 (202 positive / 107 negative)")
    print("  frozen ESM-2 arrays: aligned")
    print("  pretrained fold checkpoints: 5")


if __name__ == "__main__":
    main()
