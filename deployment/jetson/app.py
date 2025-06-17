import cv2
import numpy as np
import time
import argparse

def get_engine(engine_path):
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available.")
        return None
        
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, namespace="")
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    except FileNotFoundError:
        print(f"Engine file not found: {engine_path}")
        return None

def main():
    parser = argparse.ArgumentParser("Jetson Nano Deployment Simulation")
    parser.add_argument('--engine', type=str, default='models/student.engine')
    parser.add_argument('--image', type=str, help="Path to input image. Uses dummy if not provided.")
    args = parser.parse_args()
    
    # Initialize Engine (if TRT available)
    engine = get_engine(args.engine)
    
    # If we don't have an engine, we simulate for testing
    if engine is None:
        print("Running in simulation mode (no TRT engine found or TRT not installed).")
        print("Measurements below are simulated.")
        
        # Simulate video capture
        for i in range(10):
            t0 = time.time()
            
            # Simulate capture
            time.sleep(0.033) # ~30fps camera
            t1 = time.time()
            
            # Simulate preprocess
            img = np.random.randn(224, 224, 3)
            time.sleep(0.005)
            t2 = time.time()
            
            # Simulate inference
            time.sleep(0.015)
            t3 = time.time()
            
            # Simulate postprocess
            mask = np.zeros((224, 224), dtype=np.uint8)
            time.sleep(0.005)
            t4 = time.time()
            
            print(f"Frame {i}: Capture={1000*(t1-t0):.1f}ms, Pre={1000*(t2-t1):.1f}ms, "
                  f"Infer={1000*(t3-t2):.1f}ms, Post={1000*(t4-t3):.1f}ms | "
                  f"Total={1000*(t4-t0):.1f}ms, FPS={1/(t4-t0):.1f}")
        return

    # Real TRT deployment logic
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
            
    print("Engine loaded and buffers allocated.")
    
    # For now just running dummy data through TRT to measure inference
    for i in range(10):
        t0 = time.time()
        
        # Capture
        time.sleep(0.033) 
        t1 = time.time()
        
        # Preprocess
        dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
        np.copyto(inputs[0]['host'], dummy_img.ravel())
        t2 = time.time()
        
        # Inference
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
        t3 = time.time()
        
        # Postprocess
        out = outputs[0]['host'].reshape((1, 21, 224, 224))
        mask = np.argmax(out, axis=1)
        t4 = time.time()
        
        print(f"Frame {i}: Capture={1000*(t1-t0):.1f}ms, Pre={1000*(t2-t1):.1f}ms, "
              f"Infer={1000*(t3-t2):.1f}ms, Post={1000*(t4-t3):.1f}ms | "
              f"Total={1000*(t4-t0):.1f}ms, FPS={1/(t4-t0):.1f}")
              
if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
