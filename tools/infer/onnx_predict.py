import os
import sys
import argparse
import time
import json
import cv2
import numpy as np
import onnxruntime as ort

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from pytorchocr.utils.utility import get_image_file_list, check_and_read
from pytorchocr.data import create_operators, transform
from pytorchocr.postprocess import build_post_process
import tools.infer.pytorchocr_utility as utility


class ONNXTextDetector:
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        
        # Initialize preprocessing
        self._setup_preprocessing()
        
        # Initialize postprocessing
        self._setup_postprocessing()
        
        # Load ONNX model
        self._load_onnx_model()
        
    def _setup_preprocessing(self):
        """Setup preprocessing pipeline similar to TextDetector"""
        # Check if model has fixed input size
        input_shape = None
        if hasattr(self.args, 'onnx_model_path') and os.path.exists(self.args.onnx_model_path):
            try:
                import onnx
                onnx_model = onnx.load(self.args.onnx_model_path)
                input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
                if len(input_shape) == 4:
                    h_dim = input_shape[2].dim_value if input_shape[2].dim_value > 0 else None
                    w_dim = input_shape[3].dim_value if input_shape[3].dim_value > 0 else None
                    if h_dim and w_dim:
                        self.fixed_input_size = (h_dim, w_dim)
                        print(f"Detected fixed input size: {self.fixed_input_size}")
                    else:
                        self.fixed_input_size = None
                        print("Detected dynamic input size")
            except:
                self.fixed_input_size = None
        else:
            self.fixed_input_size = None
        
        # Setup preprocessing based on whether we have fixed size or not
        if self.fixed_input_size:
            # For fixed size models, resize to exact dimensions
            pre_process_list = [{
                'DetResizeForTest': {
                    'image_shape': self.fixed_input_size,  # Force exact size
                    'keep_ratio': False
                }
            }, {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }, {
                'ToCHWImage': None
            }, {
                'KeepKeys': {
                    'keep_keys': ['image', 'shape']
                }
            }]
        else:
            # For dynamic size models, use limit-based resizing
            pre_process_list = [{
                'DetResizeForTest': {
                    'limit_side_len': self.args.det_limit_side_len,
                    'limit_type': self.args.det_limit_type,
                }
            }, {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }, {
                'ToCHWImage': None
            }, {
                'KeepKeys': {
                    'keep_keys': ['image', 'shape']
                }
            }]
        
        # Adjust normalization for DB++
        if self.det_algorithm == "DB++":
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean': [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': self.args.det_limit_side_len
                }
            }
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            
        self.preprocess_op = create_operators(pre_process_list)
        
    def _setup_postprocessing(self):
        """Setup postprocessing based on detection algorithm"""
        postprocess_params = {}
        
        if self.det_algorithm in ["DB", "DB++"]:
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = self.args.det_db_thresh
            postprocess_params["box_thresh"] = self.args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = self.args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = self.args.use_dilation
            postprocess_params["score_mode"] = self.args.det_db_score_mode
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = self.args.det_east_score_thresh
            postprocess_params["cover_thresh"] = self.args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = self.args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = self.args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = self.args.det_sast_nms_thresh
            self.det_sast_polygon = self.args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = self.args.det_pse_thresh
            postprocess_params["box_thresh"] = self.args.det_pse_box_thresh
            postprocess_params["min_area"] = self.args.det_pse_min_area
            postprocess_params["box_type"] = self.args.det_pse_box_type
            postprocess_params["scale"] = self.args.det_pse_scale
            self.det_pse_box_type = self.args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = self.args.scales
            postprocess_params["alpha"] = self.args.alpha
            postprocess_params["beta"] = self.args.beta
            postprocess_params["fourier_degree"] = self.args.fourier_degree
            postprocess_params["box_type"] = self.args.det_fce_box_type
        else:
            raise ValueError(f"Unknown det_algorithm: {self.det_algorithm}")
            
        self.postprocess_op = build_post_process(postprocess_params)
        
    def _load_onnx_model(self):
        """Load ONNX model with appropriate providers"""
        # Setup ONNX Runtime providers
        providers = []
        if self.args.use_gpu and ort.get_device() == 'GPU':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Load model
        print(f"Loading ONNX model from: {self.args.onnx_model_path}")
        print(f"Using providers: {providers}")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.args.onnx_model_path, 
            sess_options=session_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded successfully!")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")
        print(f"Output shape: {self.session.get_outputs()[0].shape}")
        
    def order_points_clockwise(self, pts):
        """Order points in clockwise direction"""
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect
    
    def clip_det_res(self, points, img_height, img_width):
        """Clip detection results to image boundaries"""
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    
    def filter_tag_det_res(self, dt_boxes, image_shape):
        """Filter and clean detection results"""
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        """Filter detection results with only clipping"""
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def predict(self, img):
        """Predict text regions in image"""
        ori_im = img.copy()
        data = {'image': img}
        
        st = time.time()
        
        # Preprocessing
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
            
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        
        # Convert to appropriate dtype for FP16 models
        if self.args.use_fp16:
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)
        
        # Check if we need to resize input to match expected dimensions
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and input_shape[2] != 'height' and input_shape[3] != 'width':
            # Fixed input size model
            expected_h, expected_w = input_shape[2], input_shape[3]
            current_h, current_w = img.shape[2], img.shape[3]
            
            if current_h != expected_h or current_w != expected_w:
                print(f"Resizing input from ({current_h}, {current_w}) to ({expected_h}, {expected_w})")
                # Resize the input image to match expected dimensions
                img_resized = np.zeros((1, 3, expected_h, expected_w), dtype=img.dtype)
                for c in range(3):
                    img_resized[0, c] = cv2.resize(img[0, c], (expected_w, expected_h))
                img = img_resized
                
                # Update shape_list accordingly
                scale_h = expected_h / current_h
                scale_w = expected_w / current_w
                shape_list[0] = [expected_h, expected_w, scale_h, scale_w]
        
        # ONNX inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: img})
        except Exception as e:
            print(f"ONNX inference error: {e}")
            print(f"Input shape: {img.shape}")
            print(f"Expected input shape: {self.session.get_inputs()[0].shape}")
            raise e
        
        # Parse outputs based on algorithm
        preds = {}
        if self.det_algorithm == "EAST":
            # For EAST, we need to handle multiple outputs
            preds['f_geo'] = outputs[0] if len(outputs) > 1 else outputs[0][:, :2]
            preds['f_score'] = outputs[1] if len(outputs) > 1 else outputs[0][:, 2:]
        elif self.det_algorithm == 'SAST':
            # For SAST, handle multiple outputs
            if len(outputs) == 1:
                # If single output, split channels
                output = outputs[0]
                preds['f_border'] = output[:, :2]
                preds['f_score'] = output[:, 2:3]
                preds['f_tco'] = output[:, 3:5]
                preds['f_tvo'] = output[:, 5:7]
            else:
                preds['f_border'] = outputs[0]
                preds['f_score'] = outputs[1]
                preds['f_tco'] = outputs[2]
                preds['f_tvo'] = outputs[3]
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs[0]
        elif self.det_algorithm == 'FCE':
            for i, output in enumerate(outputs):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError(f"Algorithm {self.det_algorithm} not implemented")
        
        # Postprocessing
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        
        # Filter results
        if (self.det_algorithm == "SAST" and 
            hasattr(self, 'det_sast_polygon') and self.det_sast_polygon) or \
           (self.det_algorithm in ["PSE", "FCE"] and 
            hasattr(self.postprocess_op, 'box_type') and self.postprocess_op.box_type == 'poly'):
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        
        et = time.time()
        return dt_boxes, et - st
    
    def __call__(self, img, use_slice=False):
        """Main inference method with optional image slicing"""
        MIN_BOUND_DISTANCE = 50
        dt_boxes = np.zeros((0, 4, 2), dtype=np.float32)
        elapse = 0
        
        # Handle tall images (height >> width)
        if (img.shape[0] / img.shape[1] > 2 and 
            img.shape[0] > self.args.det_limit_side_len and use_slice):
            
            start_h = 0
            end_h = 0
            while end_h <= img.shape[0]:
                end_h = start_h + img.shape[1] * 3 // 4
                subimg = img[start_h:end_h, :]
                if len(subimg) == 0:
                    break
                    
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_h
                
                if (len(sub_dt_boxes) == 0 or 
                    img.shape[1] - max([x[-1][1] for x in sub_dt_boxes]) > MIN_BOUND_DISTANCE):
                    start_h = end_h
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 1])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    bottom_line = (0 if len(sub_dt_boxes) <= 1 
                                 else int(np.max(sub_dt_boxes[:-1, 2, 1])))
                    if bottom_line > 0:
                        start_h += bottom_line
                        sub_dt_boxes = sub_dt_boxes[sub_dt_boxes[:, 2, 1] <= bottom_line]
                    else:
                        start_h = end_h
                        
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array([0, offset], dtype=np.float32)
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([0, offset], dtype=np.float32),
                            axis=0
                        )
                elapse += sub_elapse
                
        # Handle wide images (width >> height)
        elif (img.shape[1] / img.shape[0] > 3 and 
              img.shape[1] > self.args.det_limit_side_len * 3 and use_slice):
            
            start_w = 0
            end_w = 0
            while end_w <= img.shape[1]:
                end_w = start_w + img.shape[0] * 3 // 4
                subimg = img[:, start_w:end_w]
                if len(subimg) == 0:
                    break
                    
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_w
                
                if (len(sub_dt_boxes) == 0 or 
                    img.shape[0] - max([x[-1][0] for x in sub_dt_boxes]) > MIN_BOUND_DISTANCE):
                    start_w = end_w
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 0])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    right_line = (0 if len(sub_dt_boxes) <= 1 
                                else int(np.max(sub_dt_boxes[:-1, 1, 0])))
                    if right_line > 0:
                        start_w += right_line
                        sub_dt_boxes = sub_dt_boxes[sub_dt_boxes[:, 1, 0] <= right_line]
                    else:
                        start_w = end_w
                        
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array([offset, 0], dtype=np.float32)
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([offset, 0], dtype=np.float32),
                            axis=0
                        )
                elapse += sub_elapse
        else:
            # Process normally
            dt_boxes, elapse = self.predict(img)
            
        return dt_boxes, elapse


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX Text Detection Inference')
    
    # Model paths
    parser.add_argument("--onnx_model_path", type=str, required=True,
                       help="Path to ONNX model file (.onnx)")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Path to input image or directory")
    parser.add_argument("--draw_img_save_dir", type=str, default="./inference_results",
                       help="Directory to save visualization results")
    
    # Model configuration
    parser.add_argument("--det_algorithm", type=str, default='DB++',
                       choices=['DB', 'DB++', 'EAST', 'SAST', 'PSE', 'FCE'],
                       help="Detection algorithm")
    parser.add_argument("--use_fp16", action='store_true',
                       help="Use FP16 model")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Use GPU for inference")
    parser.add_argument("--use_slice", action='store_true',
                       help="Use image slicing for large images")
    
    # Detection parameters
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    
    # DB/DB++ parameters
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--use_dilation", action='store_true')
    parser.add_argument("--det_db_score_mode", type=str, default='slow')
    
    # EAST parameters
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)
    
    # SAST parameters
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", action='store_true')
    
    # PSE parameters
    parser.add_argument("--det_pse_thresh", type=float, default=0.0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=int, default=16)
    parser.add_argument("--det_pse_box_type", type=str, default='quad')
    parser.add_argument("--det_pse_scale", type=int, default=1)
    
    # FCE parameters
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)
    parser.add_argument("--det_fce_box_type", type=str, default='poly')
    
    # Other parameters
    parser.add_argument("--page_num", type=int, default=0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check ONNX Runtime installation
    try:
        import onnxruntime as ort
        print(f"ONNXRuntime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
    except ImportError:
        print("Please install onnxruntime: pip install onnxruntime or onnxruntime-gpu")
        return
    
    # Create output directory
    os.makedirs(args.draw_img_save_dir, exist_ok=True)
    
    # Initialize detector
    print("Initializing ONNX Text Detector...")
    text_detector = ONNXTextDetector(args)
    
    # Get image list
    image_file_list = get_image_file_list(args.image_dir)
    total_time = 0
    count = 0
    save_results = []
    
    print(f"Found {len(image_file_list)} images to process")
    print(f"Model type: {'FP16' if args.use_fp16 else 'FP32'}")
    print(f"Using GPU: {args.use_gpu}")
    print(f"Algorithm: {args.det_algorithm}")
    
    for idx, image_file in enumerate(image_file_list):
        print(f"\nProcessing {idx+1}/{len(image_file_list)}: {image_file}")
        
        # Load image
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                print(f"Error loading image: {image_file}")
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        
        # Process each image/page
        for index, img in enumerate(imgs):
            st = time.time()
            dt_boxes, _ = text_detector(img, use_slice=args.use_slice)
            elapse = time.time() - st
            total_time += elapse
            count += 1
            
            # Save results
            if len(imgs) > 1:
                save_pred = (f"{os.path.basename(image_file)}_{index}\t" +
                           f"{json.dumps([x.tolist() for x in dt_boxes])}\n")
                print(f"  Page {index}: Found {len(dt_boxes)} text regions in {elapse:.3f}s")
            else:
                save_pred = (f"{os.path.basename(image_file)}\t" +
                           f"{json.dumps([x.tolist() for x in dt_boxes])}\n")
                print(f"  Found {len(dt_boxes)} text regions in {elapse:.3f}s")
            
            save_results.append(save_pred)
            
            # Draw and save visualization
            src_im = utility.draw_text_det_res(dt_boxes, img)
            
            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace(".pdf", f"_{index}.png")
            else:
                save_file = image_file
                
            img_path = os.path.join(
                args.draw_img_save_dir, 
                f"det_res_{os.path.basename(save_file)}"
            )
            cv2.imwrite(img_path, src_im)
            print(f"  Visualization saved: {img_path}")
    
    # Save detection results
    results_file = os.path.join(args.draw_img_save_dir, "det_results.txt")
    with open(results_file, "w") as f:
        f.writelines(save_results)
    
    print(f"\n{'='*50}")
    print(f"Inference completed!")
    print(f"Total images processed: {count}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per image: {total_time/count:.3f}s")
    print(f"Results saved to: {args.draw_img_save_dir}")
    print(f"Detection results: {results_file}")


if __name__ == "__main__":
    main()
