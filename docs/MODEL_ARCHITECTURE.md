# Model architecture

The canonical architecture is implemented in `src/deeph_immune/model.py`.

## Branch I — sequence interaction branch

1. Peptide tokens (maximum length 25) → CCT encoder (64-dimensional latent representation; two transformer blocks).
2. Aligned MHC-II tokens (281 positions) → CCT encoder (64-dimensional latent representation; one transformer block).
3. **Peptide-query → MHC-key/value cross-attention** with four heads.
4. Peptide-conditioned cross-attention pooling, learned candidate 9-mer core pooling, and peptide pooling.

A direct standalone MHC pooled vector is deliberately not concatenated into the classifier in the shortcut-resistant architecture.

## Branch II — frozen ESM-2 feature branch

Precomputed 1280-dimensional ESM-2 mean-pooled embeddings are projected to 128 dimensions. Peptide and MHC representations are L2-normalized and combined using peptide representation, element-wise product, and absolute difference. A standalone MHC ESM vector is not passed directly to the classifier.

## Fusion and classifier

The sequence and ESM interaction features are combined with gated fusion, followed by dense layers (128 → 64 → sigmoid).

