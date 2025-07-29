import os
import sys
import argparse
import torch
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from tools.infer.predict_det import TextDetector
import tools.infer.pytorchocr_utility as utility

def main(args):    
    print("Initializing TextDetector to load the model...")
    text_detector = TextDetector(args)
    model = text_detector.net
    model.eval()
    print("Model loaded successfully.")

    dummy_input = torch.randn(1, 3, args.height, args.width)
    if args.use_gpu:
        dummy_input = dummy_input.cuda()
        model = model.cuda()
    
    print(f"Using dummy input shape: (1, 3, {args.height}, {args.width})")

    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Xuất mô hình ONNX Float32 ---
    try:
        fp32_save_path = os.path.join(args.save_dir, "model_fp32.onnx")
        print(f"\nExporting FP32 model to: {fp32_save_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            fp32_save_path,
            export_params=True,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['maps'],
            dynamic_axes={
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'maps': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print("FP32 ONNX model exported successfully.")
    except Exception as e:
        print(f"Error during FP32 export: {e}")

    # --- Xuất mô hình ONNX Float16 ---
    try:
        fp16_save_path = os.path.join(args.save_dir, "model_fp16.onnx")
        print(f"\nConverting model to FP16 and exporting to: {fp16_save_path}")
        
        model_fp16 = model.half()
        dummy_input_fp16 = dummy_input.half()
        
        torch.onnx.export(
            model_fp16,
            dummy_input_fp16,
            fp16_save_path,
            export_params=True,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['maps'],
            dynamic_axes={
                'image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'maps': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print("FP16 ONNX model exported successfully.")
    except Exception as e:
        print(f"Error during FP16 export: {e}")


def parse_export_args():
  
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--det_model_path", type=str, required=True, help="Path to the PyTorch detection model (.pth).")
    parser.add_argument("--det_yaml_path", type=str, required=True, help="Path to the model's YAML config file.")
    
    parser.add_argument("--save_dir", type=str, default="./onnx_models", help="Directory to save the exported ONNX models.")
    parser.add_argument("--height", type=int, default=736, help="The height of the dummy input for ONNX export.")
    parser.add_argument("--width", type=int, default=1280, help="The width of the dummy input for ONNX export.")

    parser.add_argument("--use_gpu", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--use_dilation", type=bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default='slow')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_export_args()
    main(args)