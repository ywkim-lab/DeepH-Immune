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

The pretrained checkpoints can be used for inference **without retraining**. You do not need to download the ESM-2 model if compatible, precomputed **external-validation** embeddings are available locally.

Before running validation, confirm the presence of:

1. The external-validation CSV and corresponding ESM-2 peptide and MHC feature arrays.
2. All required pretrained checkpoints under `models/pretrained/`.
3. Matching row order and identifiers between the CSV and frozen features.

```bash
python scripts/verify_repository.py
python external_validation/run_external_validation.py
```

The evaluation workflow writes outputs to `outputs/external_validation/`, including prediction tables, metrics, and ROC/precision–recall figures. Depending on the inputs and script version, output filenames may include:

```text
outputs/external_validation/
├── external2_154_predictions.csv
├── per_fold_and_ensemble_metrics.csv
├── summary.json
└── figures/
    ├── ROC_ensemble.png
    └── PR_ensemble.png
```

**Note:** The evaluation script cannot run directly if its expected external ESM-2 arrays are also missing. In that case, regenerate those features using the same encoder and preprocessing procedure before validation.

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

The sequence branch uses:

- A **peptide CCT encoder** with a 64-dimensional latent representation and two transformer blocks.
- An **MHC-II CCT encoder** with a 64-dimensional latent representation and one transformer block.
- **Four-head peptide → MHC cross-attention**, using peptide queries and MHC keys/values.
- Cross-attention pooling, learned candidate 9-mer core pooling, and peptide pooling.

The shortcut-resistant configuration does not pass an independent standalone `mhc_pool` feature directly to the classifier.

### Branch II — frozen ESM-2 representations

This branch uses [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D). The peptide and ungapped MHC-II sequences are represented by residue-mean-pooled, **1,280-dimensional** frozen embeddings. Learned bottlenecks and L2 normalization support interaction-oriented fusion of peptide features, peptide × MHC products, and absolute peptide–MHC feature differences. A standalone MHC ESM vector is not directly concatenated to the final classifier.

### Fusion and prediction

The sequence and ESM-2 interaction features are combined by **gated fusion** and classified using dense layers of **128 → 64 → sigmoid**, yielding a predicted immunogenicity score.

## Retraining and ESM-2 feature generation

**The original training ESM-2 feature arrays are not distributed with this repository.** In particular, the following generated feature files must be recreated before retraining:

- `ESM2_mhc_training_Immuscope_7092`
- `ESM2_peptide_training_Immuscope_7092`

Their exact filename extensions and output locations should match the paths expected by the training pipeline. Preserve the training CSV's original sequence ordering and sample IDs: **the embedding-array rows must align exactly with the corresponding CSV rows**.

Create a separate environment for ESM-2 extraction:

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

## Reproducibility notes

- Retain allele, peptide, label, and sample-ID mappings during preprocessing.
- Do not reorder input CSV rows without regenerating the aligned embedding arrays.
- Check both training/test separation and overlap with external-validation observations.
- Use the provided checkpoints for model evaluation; retraining requires generating the missing training features.

  > **Repository data notice:** The large frozen ESM-2 feature arrays for the **7,092-sample training set** (`ESM2_mhc_training_Immuscope_7092` and `ESM2_peptide_training_Immuscope_7092`) are **not hosted in this repository** because of their file sizes. To retrain the model, users must generate the corresponding features locally. The pretrained five-fold Keras checkpoints are provided separately under `models/pretrained/`. External validation without regeneration is possible **only when the required external-validation input files and frozen ESM-2 features are present**.

## Citation

If you use this repository in published research, please cite the corresponding DeepH-Immune manuscript. Add a complete bibliographic reference and DOI here when available.

## License and contact

Please consult the repository for its current license and issue tracker. Questions about reproducing the methods or using the code can be submitted through [GitHub Issues](https://github.com/ywkim-lab/DeepH-Immune/issues).


