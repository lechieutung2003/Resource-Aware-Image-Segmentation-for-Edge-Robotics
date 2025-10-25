"""
TensorRT Engine Builder for Edge Robotics Segmentation.
Parses ONNX computational graphs, applies layer fusion, and compiles optimized
FP16 / INT8 serialized execution plans for NVIDIA Jetson platforms.
"""

import os
import sys
import numpy as np
import time
import argparse

def build_engine(onnx_path, engine_path, fp16=True, int8=False, calibrator=None):
    """
    Compile serialized TensorRT engine from an ONNX model.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] TensorRT is not installed. Cannot build engine on host.")
        return False
        
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Enable FP16 half precision for Jetson Maxwell GPU
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 precision enabled.")
        
    # Enable INT8 quantization if calibrator is provided
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            config.int8_calibrator = calibrator
            print("[INFO] INT8 precision enabled with Entropy Calibrator II.")

    print(f"[INFO] Parsing ONNX model: {onnx_path}")
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return False

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse ONNX model.")
            for error in range(parser.num_errors):
                print(f"       {parser.get_error(error)}")
            return False
            
    print("[INFO] Compiling TensorRT execution plan (layer fusion & kernel auto-tuning)...")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    end_time = time.time()
    
    if serialized_engine is None:
        print("[ERROR] Failed to build serialized TensorRT engine.")
        return False
        
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"[SUCCESS] Engine compiled in {end_time - start_time:.2f} seconds.")
    print(f"[SUCCESS] Saved serialized plan to: {engine_path}")
    return True

def main():
    parser = argparse.ArgumentParser("Build TensorRT engine for edge deployment")
    parser.add_argument('--onnx', type=str, default='models/student.onnx',
                        help="Path to input ONNX model")
    parser.add_argument('--engine', type=str, default='models/student.engine',
                        help="Path to output serialized TensorRT engine")
    parser.add_argument('--fp16', action='store_true', default=True,
                        help="Enable FP16 precision mode (default: True)")
    parser.add_argument('--int8', action='store_true', default=False,
                        help="Enable INT8 precision mode")
    args = parser.parse_args()
    
    build_engine(args.onnx, args.engine, fp16=args.fp16, int8=args.int8)

if __name__ == "__main__":
    main()
