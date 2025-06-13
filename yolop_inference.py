import torch
import cv2
import numpy as np
import time
from pathlib import Path
import random

from utils.utils import (
    time_synchronized, select_device, scale_coords, xyxy2xywh,
    non_max_suppression, split_for_trace_model,
    plot_one_box, letterbox
)
from utils.Lane import Lane
class YOLOPModel:
    def __init__(self, weights_path='yolopv2.pt', device='cuda:0', img_size=640, conf_thres=0.3, iou_thres=0.45):
        self.device = select_device(device)
        self.model = torch.jit.load(weights_path, map_location=self.device).to(self.device)
        self.model.eval()
        self.img_size = img_size
        self.half = self.device.type != 'cpu'
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Precompute normalization tensors
        self.mean_float = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1).float()
        self.std_float = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1).float()
        self.mean_half = self.mean_float.half()
        self.std_half = self.std_float.half()
        if self.half:
            self.model.half()
        dummy = torch.zeros(1, 3, img_size, img_size).to(self.device)
        self.model(dummy.half() if self.half else dummy)
        print("[INFO] YOLOPv2 Model loaded and warmed up on:", next(self.model.parameters()).device)

    def infer(self, frame_bgr: np.ndarray):
        try:
            img0 = frame_bgr.copy()
            output_img = img0.copy()

            # Preprocessing
            print(f"[DEBUG] Input image: shape={img0.shape}, min={img0.min()}, max={img0.max()}")
            img = letterbox(img0, self.img_size, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            if self.half:
                img = img.half()
                mean = self.mean_half
                std = self.std_half
            else:
                img = img.float()
                mean = self.mean_float
                std = self.std_float
            img = img / 255.0
            img = (img - mean) / std
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.model(img)
            t2 = time_synchronized()
            inference_time = t2 - t1
            print(f"[DEBUG] YOLOPv2 lane output: shape={ll.shape}, min={ll.min().item()}, max={ll.max().item()}")

            # Post-processing
            pred = split_for_trace_model(pred, anchor_grid)
            detections = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

            lane_detector = Lane(ll)
            lane_detector.detect_lanes()
            lane_detector.detect_road()

            # Process detections
            det = detections[0]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], output_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, output_img, line_thickness=3)

            # Overlay lane and road visuals
            output_img = lane_detector.show_detected_lanes(output_img)
            output_img = lane_detector.show_roads(output_img)

            return output_img, det, lane_detector, inference_time
        except Exception as e:
            print(f"[⚠️ YOLOPv2 inference error: {e}")
            return img0, None, None, 0.0