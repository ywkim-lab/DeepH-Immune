from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_counts_and_alignment():
    train = pd.read_csv(ROOT / "data/training/ImmuScope_7092_trainable_fullMHC281_allelebalanced.csv")
    ext = pd.read_csv(ROOT / "data/external/DeepH_Immune_CEDAR_External_validation_mapped_full_MHC281_309.csv")
    assert len(train) == 7092
    assert int((train.Immunogenicity == 1).sum()) == 3546
    assert int((train.Immunogenicity == 0).sum()) == 3546
    assert train.MHC_allele.nunique() == 46
    assert len(ext) == 309
    assert int((ext.Immunogenicity == 1).sum()) == 202
    assert int((ext.Immunogenicity == 0).sum()) == 107
    ids = np.load(ROOT / "data/training/features/Sample_ids_training_Immuscope_7092.npy", allow_pickle=True).astype(str)
    assert np.array_equal(train.sample_id.astype(str).to_numpy(), ids)
