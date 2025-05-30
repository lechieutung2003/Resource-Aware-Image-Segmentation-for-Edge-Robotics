import yaml
import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.student.model import get_student_model

def export_to_onnx(model, dummy_input, onnx_path, opset_version=14, dynamic_axes=False):
    model.eval()
    
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }

    # Extract the actual tensor from the dict output for ONNX export if necessary,
    # or export the model directly if it handles dicts properly.
    # Often dict outputs are problematic for ONNX, so we wrap it.
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)['out']
            
    wrapped_model = ONNXWrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict
    )
    print(f"Model successfully exported to {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Export student model to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='models/student.onnx')
    args = parser.parse_args()
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu') # Export on CPU
    model = get_student_model(config)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Exiting.")
        return
        
    model.to(device)
    
    # Create dummy input based on config
    img_size = config['dataset']['img_size']
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)
    
    opset = config['export'].get('opset', 14)
    dynamic_axes = config['export'].get('dynamic_axes', False)
    
    export_to_onnx(model, dummy_input, args.output, opset, dynamic_axes)
    
    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully via onnx.checker.")
        
        # Verify numerical consistency
        ort_session = ort.InferenceSession(args.output)
        
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        pytorch_outs = model(dummy_input)['out'].detach().numpy()
        
        diff = np.max(np.abs(ort_outs[0] - pytorch_outs))
        print(f"Maximum absolute difference between PyTorch and ONNX outputs: {diff:.6f}")
        
    except ImportError:
        print("ONNX or ONNXRuntime not installed, skipping numerical verification.")

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update

    # TODO: optimize this block
