# Deployment

## ONNX Export
The PyTorch model is first exported to ONNX (Open Neural Network Exchange).
Script: `scripts/export_onnx.py`
- We use opset 14 to ensure compatibility with modern TensorRT versions.
- Constant folding is enabled.

## TensorRT Optimization
TensorRT provides extreme optimization for NVIDIA GPUs.
Script: `deployment/tensorrt/build_engine.py`
- Parses the ONNX model.
- Applies layer fusion and kernel auto-tuning.
- Generates a serialized `.engine` file.
- **FP16** mode is enabled to leverage Tensor Cores on the Jetson Nano for faster inference without significant accuracy drop.

## Jetson Nano Application
Script: `deployment/jetson/app.py`
- Uses PyCUDA and the TensorRT Python API.
- Captures frames.
- Allocates page-locked memory for fast GPU transfers.
- Runs asynchronous inference.
- Decodes the mask efficiently.

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->

<!-- revised by Tung Le -->
