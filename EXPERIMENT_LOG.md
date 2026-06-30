# Experiment Log

This file tracks all experiments and results for the TriConsistencyNet research project.

---

## Experiment: Baseline_v1 (EfficientNetV2-S Full Run)
- **Date**: 2026-07-01
- **Git Commit**: `05ca6e8`
- **Dataset**: FaceForensics++ (c23)
- **Backbone**: `tf_efficientnetv2_s`
- **Batch Size**: 128 (on DGX A100)
- **Learning Rate**: 1e-4
- **Epochs**: 30 (Full Run)
- **Best Validation Accuracy**: `96.76% (Epoch 24)`
- **Test Accuracy**: `95.85%`
- **Test Precision**: `97.29%`
- **Test Recall**: `97.94%`
- **Test F1-score**: `97.61%`
- **Test ROC-AUC**: `0.9426`
- **Confusion Matrix**:
  ```text
  [[11920  2528]
   [ 1909 90621]]
  ```
- **Notes**: Baseline model successfully trained to 30 epochs and frozen. Evaluated on the unseen test split of 106,978 faces. Showing strong convergence and solid generalization. Ready as comparison benchmark.
