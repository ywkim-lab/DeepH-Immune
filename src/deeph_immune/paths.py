"""Repository path helpers.

All command-line programs resolve files relative to the cloned repository instead
of relying on machine-specific absolute paths.
"""
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_training_csv() -> Path:
    return repository_root() / "data" / "training" / "ImmuScope_7092_trainable_fullMHC281_allelebalanced.csv"


def default_training_feature_dir() -> Path:
    return repository_root() / "data" / "training" / "features"


def default_external_csv() -> Path:
    return repository_root() / "data" / "external" / "DeepH_Immune_CEDAR_External_validation_mapped_full_MHC281_309.csv"


def default_external_feature_dir() -> Path:
    return repository_root() / "data" / "external" / "features"


def default_model_dir() -> Path:
    return repository_root() / "models" / "pretrained"
