import os
import sys
import numpy as np
import time
import argparse

def build_engine(onnx_path, engine_path, fp16=True):
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available. Cannot build engine.")
        return False
        
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Allow FP16 if requested
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
            
    print("Building TensorRT engine. This may take a while...")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    end_time = time.time()
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return False
        
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"Engine built successfully in {end_time - start_time:.2f} seconds.")
    print(f"Saved to: {engine_path}")
    return True

def main():
    parser = argparse.ArgumentParser("Build TensorRT engine")
    parser.add_argument('--onnx', type=str, default='models/student.onnx')
    parser.add_argument('--engine', type=str, default='models/student.engine')
    parser.add_argument('--fp16', action='store_true', default=True)
    args = parser.parse_args()
    
    build_engine(args.onnx, args.engine, args.fp16)

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update

    # TODO: optimize this block
