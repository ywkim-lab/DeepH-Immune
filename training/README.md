# Training

Run from the repository root:

```bash
python training/train_cv.py --epochs 80 --batch-size 64
```

The default reproduces the original 5-fold stratified-by-label/peptide-length workflow with seed 42. Results are written to `outputs/training_reproduction/` and do not overwrite the distributed pretrained checkpoints.
