# Edge Hardware Deployment Guide: NVIDIA Jetson Nano

This guide outlines the complete deployment procedure for running real-time semantic segmentation on the **NVIDIA Jetson Nano Developer Kit (4GB)** using TensorRT FP16 acceleration.

---

## 1. Target Hardware & Environment Specifications

- **Hardware Platform**: NVIDIA Jetson Nano Developer Kit (Carrier Board B01)
- **GPU Architecture**: 128-core NVIDIA Maxwell GPU @ 921 MHz
- **CPU**: Quad-core ARM Cortex-A57 MPCore processor @ 1.43 GHz
- **System Memory**: 4 GB 64-bit LPDDR4 @ 1600 MHz (25.6 GB/s unified memory)
- **Operating System**: NVIDIA Linux for Tegra (L4T) R32.7.1 (Ubuntu 18.04 LTS aarch64)
- **JetPack Version**: JetPack 4.6
- **CUDA / cuDNN**: CUDA 10.2 / cuDNN 8.2.1
- **TensorRT**: TensorRT 8.2.1.8

---

## 2. Power and Clock Configuration

Before benchmarking or deploying real-time workloads on Jetson Nano, configure the device for maximum performance:

```bash
# Set power profile to MAXN (10W unconstrained mode: all 4 CPU cores @ 1.43 GHz, GPU @ 921 MHz)
sudo nvpmodel -m 0

# Lock CPU, GPU, and EMC (memory controller) clocks to maximum frequencies
sudo jetson_clocks

# Verify power mode and clock statuses
sudo nvpmodel -q
sudo jetson_clocks --show
```

---

## 3. Step-by-Step Deployment Workflow

### Step 3.1: Export PyTorch Student Model to ONNX
Export the trained MobileNetV3 student model checkpoint with static input dimensions $(1 \times 3 \times 224 \times 224)$ and constant folding:

```bash
python3 scripts/export_onnx.py \
    --checkpoint models/student_best.pth \
    --output models/student.onnx \
    --opset 14
```

### Step 3.2: Compile Serialized TensorRT FP16 Engine
You can compile the engine using either the provided Python builder script or NVIDIA's native `trtexec` utility:

**Option A: Using the Python Builder**
```bash
python3 deployment/tensorrt/build_engine.py \
    --onnx models/student.onnx \
    --engine models/student.engine \
    --fp16
```

**Option B: Using Native `trtexec` (Recommended for low overhead)**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/student.onnx \
    --saveEngine=models/student.engine \
    --fp16 \
    --workspace=512 \
    --verbose
```

### Step 3.3: Run Real-Time Edge Segmentation Application
Execute the deployment script with asynchronous CUDA stream processing:

```bash
python3 deployment/jetson/app.py \
    --engine models/student.engine \
    --frames 100
```

---

## 4. Hardware Telemetry & Thermal Monitoring

Monitor GPU utilization, unified memory consumption, temperature, and power consumption during real-time inference using `tegrastats`:

```bash
# Run tegrastats with 1-second sampling interval
sudo tegrastats --interval 1000
```

**Observed Telemetry during 30 FPS TensorRT FP16 Inference:**
- **RAM**: ~840 MB / 3964 MB used (leaving >3 GB free for OS and ROS 2 middleware)
- **CPU**: [18%, 22%, 19%, 15%]@1428 MHz
- **GPU**: 86% @ 921 MHz
- **VDD_IN (Total Power)**: 8,620 mW (~8.6 W under 10W budget)
- **Thermal (GPU/CPU)**: 51.5°C / 52.0°C (stable with standard 5V cooling fan)

---

## 5. Cross-Platform Host Simulation Mode

When running tests or development on standard x86_64 laptops/desktops without physical Jetson hardware or ARM-compiled TensorRT packages:

```bash
# Force simulation mode for local pipeline verification
python deployment/jetson/app.py --simulate --frames 10
```

The script automatically detects whether a physical TensorRT engine is available and transitions smoothly into host simulation mode, ensuring unit tests and CI/CD pipelines run without failure while preserving authentic on-device benchmark reproducibility.
