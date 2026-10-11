"""Data loading and alignment checks for DeepH-Immune."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .model import (
    PEP_MAX_LEN,
    MHC_ALIGNED_LEN,
    ESM_DIM,
    tokenize_peptide,
    tokenize_mhc_aligned,
)


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    peptide_tokens: np.ndarray
    mhc_tokens: np.ndarray
    esm_peptide: np.ndarray
    esm_mhc: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray

    @property
    def model_inputs(self):
        return [self.peptide_tokens, self.mhc_tokens, self.esm_peptide, self.esm_mhc]


def _required_columns() -> set[str]:
    return {
        "sample_id", "Species", "MHC_allele", "Peptide", "Immunogenicity",
        "Peptide_length", "MHC_aligned_281", "MHC_ungapped",
    }


def load_dataset_bundle(
    csv_path: Path,
    peptide_esm_path: Path,
    mhc_esm_path: Path,
    label_path: Path,
    sample_id_path: Path,
) -> DatasetBundle:
    """Load one CSV + its frozen ESM-2 arrays and verify row alignment.

    The sample-ID check is deliberately strict. A prediction should never be run
    with embeddings whose row order differs from the CSV row order.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    missing = _required_columns().difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

    df["sample_id"] = df["sample_id"].astype(str)
    df["Peptide"] = df["Peptide"].astype(str).str.upper().str.strip()
    df["MHC_aligned_281"] = df["MHC_aligned_281"].astype(str)
    df["Immunogenicity"] = pd.to_numeric(df["Immunogenicity"]).astype(np.int32)

    pep_tokens = np.asarray([tokenize_peptide(x) for x in df["Peptide"]], dtype=np.int32)
    mhc_tokens = np.asarray([tokenize_mhc_aligned(x) for x in df["MHC_aligned_281"]], dtype=np.int32)

    esm_pep = np.load(peptide_esm_path).astype(np.float32)
    esm_mhc = np.load(mhc_esm_path).astype(np.float32)
    labels = np.load(label_path).astype(np.int32)
    sample_ids = np.load(sample_id_path, allow_pickle=True).astype(str)

    n = len(df)
    expected = {
        "peptide tokens": (pep_tokens.shape, (n, PEP_MAX_LEN)),
        "MHC tokens": (mhc_tokens.shape, (n, MHC_ALIGNED_LEN)),
        "peptide ESM": (esm_pep.shape, (n, ESM_DIM)),
        "MHC ESM": (esm_mhc.shape, (n, ESM_DIM)),
        "labels": (labels.shape, (n,)),
        "sample IDs": (sample_ids.shape, (n,)),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise ValueError(f"{name} shape mismatch: got {got}, expected {want}")

    y_csv = df["Immunogenicity"].to_numpy(dtype=np.int32)
    if not np.array_equal(y_csv, labels):
        raise ValueError("CSV labels and ESM label array are not aligned")
    if not np.array_equal(df["sample_id"].to_numpy(), sample_ids):
        raise ValueError("CSV sample_id order and ESM sample-ID array are not aligned")

    return DatasetBundle(df, pep_tokens, mhc_tokens, esm_pep, esm_mhc, labels, sample_ids)
