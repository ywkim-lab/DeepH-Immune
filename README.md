# DeepH-Immune

**MHC-II immunogenicity prediction via residue-level peptide–MHC interaction learning and protein language models**

DeepH-Immune is a deep learning framework for predicting peptide–MHC class II (pMHC-II) immunogenicity. The framework combines compact convolutional transformer (CCT) sequence encoders, peptide-to-MHC cross-attention, candidate binding-core modeling, and frozen ESM-2 protein language model features.




## Highlights

- **Sequence-interaction branch:** CCT representations for peptide and MHC-II, followed by peptide-to-MHC cross-attention and pooled sequence features.
- **Protein-language-model branch:** Residue-mean-pooled, 1,280-dimensional ESM-2 embeddings for peptide and MHC-II sequences.
- **Interaction-aware fusion:** Feature products and absolute differences complement learned peptide representations.
- **Reproducibility:** Five-fold Keras checkpoints, training and evaluation scripts, environment specifications, and alignment-verification utilities.
- **Interpretability:** Attention and binding-core analyses; attention scores should **not** be interpreted as direct evidence of physical contacts or causality.

## Requirements

- Python and a compatible TensorFlow/Keras environment (see `requirements_tensorflow.txt` and `environment_tensorflow.yml`).
- A separate ESM-2 environment when generating embeddings (see `requirements_esm2.txt` and `environment_esm2.yml`).
- Sufficient memory and computing resources for ESM-2 feature extraction or model training.

The supplied checkpoints were saved using **Keras 3.12.0**; validation was executed using **TensorFlow 2.20.0**. Match these versions where possible to minimize model-loading differences.

## Installation

```bash
git clone https://github.com/ywkim-lab/DeepH-Immune.git
cd DeepH-Immune

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements_tensorflow.txt
pip install -e .
```

## External validation


```bash
python scripts/verify_repository.py
python external_validation/run_external_validation.py
```


## Repository structure

```text
DeepH-Immune/
├── data/
│   ├── training/                   # Training CSV/metadata; large training ESM-2 arrays excluded
│   ├── external/                   # External-validation inputs (check available arrays)
│   ├── metadata.json
│   └── README.md
├── models/
│   └── pretrained/                 # Five pretrained .keras checkpoints
├── src/deeph_immune/
│   ├── model.py                    # Model architecture
│   ├── io.py                       # Data and feature alignment utilities
│   ├── metrics.py
│   ├── paths.py
│   └── training_utils.py
├── training/
│   └── train_cv.py                 # Five-fold training script
├── external_validation/
│   └── run_external_validation.py  # External evaluation script
├── esm2/
│   └── extract_esm2.py            # Frozen feature extraction
├── notebooks/
│   ├── 01_Training_Reproduction.ipynb
│   ├── 02_External_Validation.ipynb
│   └── 03_BranchI_Interpretability.ipynb
├── scripts/
│   └── verify_repository.py
├── docs/
├── tests/
├── outputs/                        # Generated results (typically Git-ignored)
├── requirements_tensorflow.txt
├── requirements_esm2.txt
├── environment_tensorflow.yml
├── environment_esm2.yml
├── pyproject.toml
├── Makefile
├── MANIFEST.sha256
└── README.md
```

> The tree documents the intended project organization; check the repository for the files included in the current release.

## Model architecture

### Branch I — residue-level peptide–MHC-II interactions

### Branch II — frozen ESM-2 representations

This branch uses [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D). The peptide and ungapped MHC-II sequences are represented by residue-mean-pooled, **1,280-dimensional** frozen embeddings.



For new data, create a separate environment for ESM-2 extraction:

```bash
conda env create -f environment_esm2.yml
conda activate deeph-immune-esm2
python esm2/extract_esm2.py --help
```

Check the extractor's arguments and input/output paths in `esm2/extract_esm2.py`, then generate both peptide and MHC arrays. Verify their alignment before training:

```bash
python scripts/verify_repository.py
```

After the required training features have been generated and placed at the expected paths, run:

```bash
python training/train_cv.py --epochs 80 --batch-size 64
```

Training results may vary with software versions, random seeds, hardware, and data preprocessing.

## Interpretation and limitations

DeepH-Immune predicts functional immunogenicity from peptide–MHC-II sequence information and protein-language-model features; predictions do not directly measure T-cell receptor binding or guarantee an experimental T-cell response. Attention visualizations represent **learned model dependencies**, not proof of physical residue contacts or causal mechanisms. Because training uses immunogenicity rather than direct binding-affinity labels, the resulting motifs should be described as **model-inferred, immunogenicity-associated core motifs** unless independently validated.


## Citation

If you use this repository in published research, please cite the corresponding DeepH-Immune manuscript.

## License and contact

Please consult the repository for its current license and issue tracker. Questions about reproducing the methods or using the code can be submitted through [GitHub Issues](https://github.com/ywkim-lab/DeepH-Immune/issues).


