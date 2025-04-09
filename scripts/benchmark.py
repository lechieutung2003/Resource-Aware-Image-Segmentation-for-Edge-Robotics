import yaml
import sys
import os
import torch
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.teacher.model import get_teacher_model
from src.models.student.model import get_student_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def benchmark_latency(model, device, img_size, num_iterations=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    latency_ms = (total_time / num_iterations) * 1000
    fps = 1000 / latency_ms
    
    return latency_ms, fps

def get_onnx_ort_metrics(onnx_path, img_size):
    try:
        import onnxruntime as ort
    except ImportError:
        return "N/A", "N/A"
        
    if not os.path.exists(onnx_path):
        return "N/A", "N/A"
        
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, img_size[0], img_size[1]).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
        
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: dummy_input})
        
    end_time = time.time()
    total_time = end_time - start_time
    latency_ms = (total_time / num_iterations) * 1000
    fps = 1000 / latency_ms
    
    return latency_ms, fps
    
def get_tensorrt_metrics(engine_path, img_size):
    # This is a placeholder for actual TensorRT benchmarking since TRT may not be available on Windows out of the box
    return "N/A", "N/A"

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    img_size = config['dataset']['img_size']
    
    teacher = get_teacher_model(config)
    student = get_student_model(config)
    
    # Calculate params and size
    t_params = count_parameters(teacher)
    t_size = get_model_size_mb(teacher)
    
    s_params = count_parameters(student)
    s_size = get_model_size_mb(student)
    
    print("\nRunning latency benchmarks...")
    t_lat, t_fps = benchmark_latency(teacher, device, img_size, 50)
    s_lat, s_fps = benchmark_latency(student, device, img_size, 100)
    
    onnx_path = 'models/student.onnx'
    o_lat, o_fps = get_onnx_ort_metrics(onnx_path, img_size)
    
    trt_path = 'models/student.engine'
    trt_lat, trt_fps = get_tensorrt_metrics(trt_path, img_size)
    
    print("\n--- BENCHMARK RESULTS ---")
    print("| Model            |   Params |     Size (MB) |  Latency (ms) |      FPS |")
    print("| ---------------- | -------: | ------------: | ------------: | -------: |")
    print(f"| Teacher          | {t_params:8d} | {t_size:13.2f} | {t_lat:13.2f} | {t_fps:8.2f} |")
    print(f"| Student (PyT)    | {s_params:8d} | {s_size:13.2f} | {s_lat:13.2f} | {s_fps:8.2f} |")
    
    if isinstance(o_lat, str):
        print(f"| Student (ONNX)   |      N/A |           N/A | {o_lat:>13} | {o_fps:>8} |")
    else:
        print(f"| Student (ONNX)   |      N/A |           N/A | {o_lat:13.2f} | {o_fps:8.2f} |")
        
    print(f"| Student (TRT)    |      N/A |           N/A | {trt_lat:>13} | {trt_fps:>8} |")
    
if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update

    # TODO: optimize this block

    import logging
    logging.debug('Execution reached here')
