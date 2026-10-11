# DeepH-Immune

**Reproducible MHC class II peptide immunogenicity modelling with a shortcut-resistant CCT/cross-attention branch and frozen ESM-2 interaction features.**



## Experiment packaged here

| Dataset | Total | Positive | Negative | MHC sequence alleles |
|---|---:|---:|---:|---:|
| Allele-balanced training | **7,092** | **3,546** | **3,546** | **46** |
| Collaborator-provided external2/test | **154** | **98** | **56** | **14** |

Peptide lengths are 12–21 aa in these final files. Peptides are tokenized to a maximum length of 25; the aligned MHC-II sequence input is 281 positions. Frozen ESM-2 embeddings are 1280-dimensional.

## Clone → validate immediately

The precomputed ESM-2 arrays and five checkpoints are included, so **you do not need to download ESM-2 or retrain the model to reproduce external validation**.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> DeepH-Immune
cd DeepH-Immune

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_tensorflow.txt
pip install -e .

python scripts/verify_repository.py
python external_validation/run_external_validation.py
```

Outputs are written to:

```text
outputs/external_validation/
├── external2_154_predictions.csv
├── per_fold_and_ensemble_metrics.csv
├── summary.json
└── figures/
    ├── ROC_ensemble.png
    └── PR_ensemble.png
```

The supplied checkpoints were saved with Keras 3.12.0 and the executed validation notebook used TensorFlow 2.20.0. The supplied executed experiment reported an ensemble **AUROC ≈ 0.6642** and **AUPRC ≈ 0.7640** on external2. See [`docs/REFERENCE_RESULTS.md`](docs/REFERENCE_RESULTS.md) for fold-wise values and fixed-threshold metrics.

## Repository layout

```text
DeepH-Immune/
├── data/
│   ├── training/                  # 7,092 CSV + frozen training ESM-2 arrays
│   ├── external/                  # 154 CSV + frozen external ESM-2 arrays
│   ├── metadata.json
│   └── README.md
├── models/
│   └── pretrained/                # Fold 1–5 .keras checkpoints
├── src/deeph_immune/
│   ├── model.py                   # canonical serializable architecture
│   ├── io.py                      # strict CSV/ESM row-alignment checks
│   ├── metrics.py
│   ├── paths.py
│   └── training_utils.py
├── training/
│   └── train_cv.py                # complete 5-fold training pipeline
├── external_validation/
│   └── run_external_validation.py # one-command evaluation
├── esm2/
│   └── extract_esm2.py            # optional ESM-2 regeneration
├── notebooks/
│   ├── 01_Training_Reproduction.ipynb
│   ├── 02_External_Validation.ipynb
│   └── 03_BranchI_Interpretability.ipynb
├── scripts/
│   └── verify_repository.py
├── docs/
├── tests/
├── outputs/                       # generated results (Git-ignored)
├── requirements_tensorflow.txt
├── requirements_esm2.txt
├── environment_tensorflow.yml
├── environment_esm2.yml
├── pyproject.toml
├── Makefile
├── MANIFEST.sha256
└── README.md
```

## Model overview

### Branch I: residue-level sequence interaction

- peptide CCT encoder: 64-dimensional latent representation, two transformer blocks;
- MHC-II CCT encoder: 64-dimensional latent representation, one transformer block;
- **peptide → MHC cross-attention** (peptide query, MHC key/value; four heads);
- cross-attention pooling + learned 9-mer candidate-core pooling + peptide pooling.

The shortcut-resistant version intentionally avoids sending a direct standalone `mhc_pool` vector to the classifier.

### Branch II: frozen ESM-2 interaction features

The experiment uses `facebook/esm2_t33_650M_UR50D`. Peptide and ungapped MHC-II sequences are residue-mean-pooled to 1280-dimensional frozen features. After learned bottlenecks and L2 normalization, the fusion uses peptide representation, peptide×MHC product, and absolute difference; a standalone MHC ESM vector is not directly concatenated into the classifier.

### Fusion

Sequence and ESM interaction features are combined by gated fusion, then classified by dense layers (128 → 64 → sigmoid).

## Reproduce 5-fold training

The pre-trained checkpoints are included. To retrain from scratch using the supplied frozen features:

```bash
python training/train_cv.py --epochs 80 --batch-size 64
```

Defaults reproduce the experiment configuration:

- 5 folds;
- seed 42;
- `stratified_length` splitting;
- early stopping on validation AUPRC (patience 12);
- ReduceLROnPlateau on validation AUPRC;
- length-label sample weighting;
- allele-label weighting (approximately neutral after exact within-allele balancing, but retained for fidelity to the training workflow).

Retraining writes to `outputs/training_reproduction/` and does not overwrite `models/pretrained/`.

## Publication-quality Branch-I analysis

Open:

```bash
jupyter lab notebooks/03_BranchI_Interpretability.ipynb
```

<!-- The notebook uses **out-of-fold interpretation**: each training observation is interpreted by the fold model for which that observation was held out. 


Attention is treated as a learned dependency, not as proof of physical contact or causality. Because the model is trained on immunogenicity labels rather than direct binding-affinity labels, discovered motifs should be described as **model-inferred / immunogenicity-associated core motifs** unless independently validated.

## Optional: regenerate ESM-2 features

This is **not required** for a clone-and-run workflow. If you want to regenerate the frozen embeddings, use a separate environment:

```bash
conda env create -f environment_esm2.yml
conda activate deeph-immune-esm2
```

See [`esm2/README.md`](esm2/README.md). ESM-2 receives true peptide sequences and `MHC_ungapped`; the 281-column alignment gaps are never supplied to ESM-2.

## Integrity and reproducibility

Run:

```bash
python scripts/verify_repository.py
```

This verifies dataset counts, class balance, MHC alignment length, feature shapes, CSV↔ESM sample-ID order, and all five Keras checkpoint archives. SHA-256 hashes for distributed files are recorded in `MANIFEST.sha256`.

## Notes for GitHub

All distributed individual files are below GitHub's 100 MB per-file hard limit. The frozen `.npy` arrays and `.keras` checkpoints are intentionally tracked so a clone can run immediately. If the repository grows substantially later, moving large binary assets to a GitHub Release or Git LFS would be reasonable.

## Documentation

- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md)
- [`docs/REFERENCE_RESULTS.md`](docs/REFERENCE_RESULTS.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/ORIGINAL_FILE_MAP.md`](docs/ORIGINAL_FILE_MAP.md)
- [`docs/Task2_Experiment_Record.md`](docs/Task2_Experiment_Record.md)

---

**Experiment:** DeepH-Immune V2 shortcut-resistant, ImmuScope allele-balanced 7,092 → external2 154  
**Primary model input:** peptide sequence + aligned MHC-II sequence + frozen ESM-2 peptide/MHC features  
**Cross-validation:** 5-fold
