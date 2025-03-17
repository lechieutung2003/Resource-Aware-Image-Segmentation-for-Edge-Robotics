# Architecture

The system uses a Teacher-Student Knowledge Distillation architecture.

## Teacher Model
- **Architecture**: DeepLabV3
- **Backbone**: ResNet-50
- **Purpose**: Provides high-quality segmentation masks and soft probability targets (logits).
- **Pros**: High mIoU, strong feature representation.
- **Cons**: High latency, high memory footprint, too slow for Jetson Nano edge deployment.

## Student Model
- **Architecture**: Custom Segmentation Model
- **Backbone**: MobileNetV3-Small
- **Head**: Simple Conv-BN-ReLU-Conv projection head.
- **Purpose**: Learns to mimic the teacher's output while maintaining a fraction of the computational cost.
- **Pros**: Extremely low latency, low parameter count, perfectly suited for TensorRT on Jetson Nano.
- **Cons**: Lower baseline accuracy (mitigated by Knowledge Distillation).

## Deployment Pipeline
PyTorch (Training) -> ONNX (Export) -> TensorRT (Optimization) -> Jetson Nano (Inference)

<!-- update -->

<!-- update -->

<!-- update -->

<!-- update -->

<!-- revised by Tung Le -->

<!-- revised by Tung Le -->
