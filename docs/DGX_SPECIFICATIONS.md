# College NVIDIA DGX AI Research Infrastructure Specifications

This document stores the specifications of the DGX environment to guide architecture design, batch size limits, and training loop parameters.

## Hardware Summary

- **Server Model**: NVIDIA DGX A100 (8x GPUs total, 320 GB total GPU VRAM)
- **Allotted GPU**: 1x NVIDIA A100-SXM4 (40 GB VRAM, SXM4 high-bandwidth form factor)
- **CPU**: AMD EPYC 7742 (2 Sockets, 64 physical cores per socket, 128 total physical cores, 256 total hardware threads)
- **System Memory (RAM)**: 1.0 TiB (1024 GB)
- **Storage**: ~1.8 TB SSD volume mounted directly into workspace

## Software & Drivers

- **Operating System**: Ubuntu 22.04 LTS (running inside Kubeflow Kubernetes managed container)
- **Python Version**: 3.10
- **CUDA Toolkit Version**: 12.4
- **cuDNN Version**: 9.x
- **Container Environment**: JupyterLab running inside Docker managed by Kubeflow

## Optimization Decisions

1. **Frame Extraction**: Parallelized using `multiprocessing.Pool` targeting up to 256 CPU workers to utilize the AMD EPYC threads.
2. **Face Extraction (RetinaFace)**: Parallelized using `multiprocessing.Pool` (recommended 4-16 workers) with TensorFlow GPU Memory Growth (`tf.config.experimental.set_memory_growth`) enabled on worker startup to prevent VRAM allocation conflicts.
3. **Training Batch Sizes**: With 40 GB of A100 VRAM, batch sizes for EfficientNetV2-S and TriConsistencyNet can be scaled high (e.g. 128-256) to maximize GPU utilization.
