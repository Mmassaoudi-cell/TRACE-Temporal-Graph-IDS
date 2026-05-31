# TRACE Reproducible Architecture

This folder contains the latest efficient TRACE implementation used for the
NSL-KDD revision experiments. It intentionally excludes figure generation,
table generation, notebooks, and intermediate sweep scripts.

## Architecture

TRACE is designed for flow datasets that do not publish explicit source and
destination endpoint topology.

1. Preprocess numerical and categorical flow features using transformations
   fitted on training data only.
2. Derive surrogate graph statistics from train-fitted percentile node bins.
3. Concatenate flow features and surrogate graph features.
4. Split training data into model-training and validation folds.
5. Apply Borderline-SMOTE rare-class augmentation to the model-training fold
   only.
6. Train a class-balanced, regularized XGBoost multiclass classifier.
7. Calibrate class-specific thresholds on the held-out validation fold.

The default configuration is the efficient TRACE variant selected by the
targeted NSL-KDD sweep:

```text
surrogate nodes       16
rare-class target     12000 samples/class
XGBoost estimators    1100
maximum tree depth    5
learning rate         0.03
subsample             0.90
column subsample      0.86
validation split      0.20
random seed           42
```

## Requirements

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost
```

## Reproduce NSL-KDD Evaluation

From the repository root:

```bash
python github/trace_model.py \
  --train Data/KDD/kdd_train.csv \
  --test Data/KDD/kdd_test.csv
```

On Windows PowerShell:

```powershell
python github/trace_model.py `
  --train Data/KDD/kdd_train.csv `
  --test Data/KDD/kdd_test.csv
```

The script prints accuracy, macro precision, macro recall, and macro F1.

Verified local output for the leakage-safe packaged implementation:

```text
accuracy:        0.928628
macro_precision: 0.962838
macro_recall:    0.670747
macro_f1:        0.726920
```

## Notes

- Test labels are used only for final evaluation.
- Preprocessing, surrogate-graph statistics, augmentation, and threshold
  calibration are fitted without test-label access.
- The implementation uses a validation split inside the training file for
  threshold calibration.
- The packaged metrics are intentionally conservative: surrogate-node bins and
  frequency statistics are learned from the training fold only and reused for
  validation/test inference.
- This compact release represents the final TRACE configuration only. Earlier
  diagnostic variants are intentionally excluded.
