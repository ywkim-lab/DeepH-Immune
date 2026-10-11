# Optional ESM-2 regeneration

The precomputed ESM-2 arrays are already included. Regeneration is optional and requires the separate PyTorch/Transformers environment.

Training features:

```bash
python esm2/extract_esm2.py \
  --csv data/training/ImmuScope_7092_trainable_fullMHC281_allelebalanced.csv \
  --output-dir /tmp/training_esm \
  --name training_Immuscope_7092
```

External features:

```bash
python esm2/extract_esm2.py \
  --csv data/external/DeepH_Immune_CEDAR_External_validation_mapped_full_MHC281_309.csv \ # correct the name of the file if needed
  --output-dir /tmp/external_esm \
  --name Immuscope_external2_309
```

ESM-2 receives the true peptide sequence and `MHC_ungapped`; alignment gaps are never sent to the protein language model.
