# Experiment Log

This file tracks all experiments and results for the TriConsistencyNet research project.

---

## Experiment: Baseline_v1 (EfficientNetV2-S Sanity Check)
- **Date**: 2026-06-30
- **Git Commit**: `[Pending full-scale run]`
- **Dataset**: FaceForensics++ (c23)
- **Backbone**: `tf_efficientnetv2_s`
- **Batch Size**: 128 (on DGX A100) / 32 (local)
- **Learning Rate**: 1e-4
- **Epochs**: 2 (Sanity Run) / 30 (Full Run)
- **Best Validation Accuracy**: `[Pending]`
- **Best F1-Score**: `[Pending]`
- **Notes**: Local dry-run verification completed successfully. The pipeline runs, computes weights, loads data, executes forward/backward passes, and checkpoints models with 0 crashes. Ready for DGX sanity run.
