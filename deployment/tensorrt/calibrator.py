"""
TensorRT INT8 Entropy Calibrator for Post-Training Quantization (PTQ).
Minimizes KL-Divergence between FP32 and INT8 activation histograms on edge robotics data.
"""

import os
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None

if TRT_AVAILABLE:
    class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """
        INT8 Entropy Calibrator II for NVIDIA TensorRT.
        Feeds representative batches of input images to calculate optimal dynamic range
        and quantization scale factors for activation tensors.
        """
        def __init__(self, data_loader, cache_file="models/calibration.cache",
                     batch_size=1, input_shape=(1, 3, 224, 224)):
            super().__init__()
            self.data_loader = data_loader
            self.data_iter = iter(data_loader) if data_loader is not None else None
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.shape = input_shape
            self.device_input = cuda.mem_alloc(int(np.prod(self.shape) * 4))  # 4 bytes for float32

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            try:
                if self.data_iter is None:
                    return None
                batch = next(self.data_iter)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                data = np.ascontiguousarray(batch.cpu().numpy().astype(np.float32))
                cuda.memcpy_htod(self.device_input, data)
                return [int(self.device_input)]
            except (StopIteration, Exception):
                return None

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
else:
    class Int8EntropyCalibrator:
        """Fallback stub when TensorRT / PyCUDA are not installed on host machine."""
        def __init__(self, data_loader=None, cache_file="models/calibration.cache", **kwargs):
            self.data_loader = data_loader
            self.cache_file = cache_file
            print("[INFO] Int8EntropyCalibrator instantiated in host fallback mode (TRT/PyCUDA not installed).")
