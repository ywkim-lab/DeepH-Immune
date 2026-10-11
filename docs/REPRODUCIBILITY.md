# Reproducibility notes

The supplied trained checkpoints were saved with **Keras 3.12.0**, and the executed external-validation notebook used **TensorFlow 2.20.0** on Python 3.11 with NVIDIA GPUs. The GitHub-ready TensorFlow requirements therefore pin TensorFlow 2.20.0 and Keras 3.12.0.

External validation rebuilds the architecture from `src/deeph_immune/model.py` and calls `model.load_weights()` on each `.keras` archive. This avoids depending on historical optimizer serialization and is deliberately more robust than calling `load_model()` on the complete checkpoint.

Deterministic settings used by the training workflow:

- random seed: 42;
- five folds;
- default split: stratified by label × peptide-length bin;
- early stopping monitor: validation AUPRC;
- early-stopping patience: 12;
- learning-rate reduction patience: 5;
- initial Adam learning rate: 3e-4.

GPU kernels can introduce small numerical differences across CUDA/cuDNN hardware, so results should be compared to the reference values with reasonable floating-point tolerance rather than bit-for-bit prediction identity.
