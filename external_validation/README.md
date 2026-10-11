# External validation

After installing the TensorFlow environment, run:

```bash
python external_validation/run_external_validation.py
```

The command loads the distributed external2 CSV, checks ESM row alignment, rebuilds the model architecture, loads all five checkpoints, computes fold-wise and ensemble AUROC/AUPRC, fixed-threshold metrics, bootstrap confidence intervals, predictions, and ROC/PR figures.
