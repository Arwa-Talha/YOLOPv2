import cv2
import numpy as np
import torch
from scipy.ndimage import label

class Lane:
    def __init__(self, lane_line, threshold=0.3, save_debug=False):
        """
        Lane detection and road zone filling from lane segmentation mask.
        
        Args:
            lane_line (torch.Tensor): model output tensor (B, C, H, W)
            threshold (float): sigmoid activation threshold for lane mask
            save_debug (bool): whether to save debug lane mask image
        """
        self.is_right = False
        self.threshold = threshold
        self.save_debug = save_debug
        self.lane_mask = self.interpolate(lane_line)
        self.detected_lanes = self.detect_lanes()
        self.roads = np.zeros_like(self.lane_mask, dtype=np.uint8)
        self.detect_road()

        if self.save_debug:
            cv2.imwrite('D:/map_grad/test/debug_lane_mask.png', self.lane_mask * 255)

    def interpolate(self, lane_line):
        try:
            if not isinstance(lane_line, torch.Tensor) or lane_line.ndim != 4:
                print(f"[⚠️ Lane Error] Invalid lane_line shape: {lane_line.shape if isinstance(lane_line, torch.Tensor) else type(lane_line)}")
                return np.zeros((720, 1280), dtype=np.uint8)
            if torch.isnan(lane_line).any():
                print("[⚠️ Lane Error] lane_line contains NaN values")
                return np.zeros((720, 1280), dtype=np.uint8)

            print(f"[DEBUG] lane_line shape: {lane_line.shape}, min={lane_line.min().item()}, max={lane_line.max().item()}")

            # Interpolate segmentation to full resolution
            ll_seg_mask = torch.nn.functional.interpolate(
                lane_line, scale_factor=2, mode='bilinear', align_corners=False
            )
            
            # Convert safely to NumPy
            ll_seg_mask_np = torch.sigmoid(ll_seg_mask).detach().cpu().numpy()

            # Assume batch=1, channel=1
            lane_mask = ll_seg_mask_np[0, 0, :, :]

            # Apply threshold
            lane_mask = (lane_mask > self.threshold).astype(np.uint8)

            print(f"[DEBUG] lane_mask shape: {lane_mask.shape}, min: {lane_mask.min()}, max: {lane_mask.max()}")
            return lane_mask

        except Exception as e:
            print(f"[⚠️ Lane Error] Interpolation failed: {e}")
            # Return fallback mask with same size as lane_line if possible
            fallback_shape = (720, 1280)
            if isinstance(lane_line, torch.Tensor) and lane_line.ndim == 4:
                fallback_shape = (
                    lane_line.shape[2] * 2,
                    lane_line.shape[3] * 2
                )
            return np.zeros(fallback_shape, dtype=np.uint8)

    def detect_lanes(self):
        h = self.lane_mask.shape[0]
        self.lane_mask[:int(h * 0.5), :] = 0  # Mask upper half
        self.labeled_lanes, num_lanes = label(self.lane_mask)
        print(f"[DEBUG] Detected {num_lanes} connected components")
        H, W = self.labeled_lanes.shape
        center_col = W // 2
        start_row = int(h * 0.5)

        self.left_lanes = self.find_line_labels(start_row, center_col - 1, -1, -1)
        self.right_lanes = self.find_line_labels(start_row, center_col + 1, W, 1)

        if len(self.right_lanes) >= 1:
            self.is_right = True

        self.lanes_idx = list(set(self.left_lanes + self.right_lanes))
        self.lanes_idx.sort()
        print(f"[DEBUG] Detected lane indices: {self.lanes_idx}")

        detected_lanes = np.zeros_like(self.labeled_lanes, dtype=np.uint8)
        for label_id in self.lanes_idx:
            detected_lanes[self.labeled_lanes == label_id] = label_id
        return detected_lanes

    def find_line_labels(self, start_row, start, end, step):
        found = set()
        for col in range(start, end, step):
            labels = set(self.labeled_lanes[start_row:, col]) - {0}
            for lbl in labels:
                if lbl not in found:
                    found.add(lbl)
                    if len(found) >= 4:  # Max 4 lanes per side
                        return list(found)
        return list(found)

    def show_detected_lanes(self, img):
        blue_color = np.array([0, 0, 255], dtype=np.uint8)
        lane_mask = np.zeros_like(img, dtype=np.uint8)
        if self.detected_lanes.shape[:2] != img.shape[:2]:
            self.detected_lanes = cv2.resize(
                self.detected_lanes, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        lane_mask[self.detected_lanes != 0] = blue_color
        output_img = img.astype(np.float32) + lane_mask.astype(np.float32) * 0.5
        return output_img.astype(np.uint8)

    def detect_road(self):
        if len(self.lanes_idx) < 2:
            print(f"[⚠️ Lane Warning] Only {len(self.lanes_idx)} lanes detected, skipping road region filling.")
            return
        print(f"[DEBUG] Filling road with {len(self.lanes_idx)} lanes")
        for i in range(len(self.lanes_idx) - 1):
            l1 = self.lanes_idx[i]
            l2 = self.lanes_idx[i + 1]
            lane1_dict = self.rowwise_col_stat_for_loop(self.detected_lanes, l1, 'max')
            lane2_dict = self.rowwise_col_stat_for_loop(self.detected_lanes, l2, 'min')
            zone_idx = i + 1
            self.fill_between(lane1_dict, lane2_dict, zone_idx)

    def fill_between(self, dict1, dict2, idx):
        for row in dict1:
            if row in dict2:
                col1 = dict1[row]
                col2 = dict2[row]
                cmin, cmax = min(col1, col2), max(col1, col2)
                self.roads[row, cmin:cmax] = idx

    def rowwise_col_stat_for_loop(self, label_mask, target_label, stat='min'):
        coords = np.where(label_mask == target_label)
        rows = coords[0]
        cols = coords[1]
        row_dict = {}
        for r, c in zip(rows, cols):
            if r not in row_dict:
                row_dict[r] = c
            else:
                if stat == 'min':
                    row_dict[r] = min(row_dict[r], c)
                elif stat == 'max':
                    row_dict[r] = max(row_dict[r], c)
        return row_dict

    def show_roads(self, img):
        img_float = img.astype(np.float32)
        if self.roads.shape[:2] != img.shape[:2]:
            self.roads = cv2.resize(
                self.roads, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        colors = {
            1: np.array([255, 0, 0], dtype=np.uint8),   # Red
            2: np.array([0, 255, 0], dtype=np.uint8),   # Green
            3: np.array([0, 0, 255], dtype=np.uint8),   # Blue
            4: np.array([255, 255, 0], dtype=np.uint8), # Cyan
        }
        overlay = np.zeros_like(img, dtype=np.uint8)
        for value, color in colors.items():
            overlay[self.roads == value] = color
        blended = (img_float + overlay.astype(np.float32) * 0.5).astype(np.uint8)
        return blended

    def get_zone_at(self, x, y):
        """Return the zone ID at coordinates (x, y) from the roads mask."""
        try:
            if self.roads.shape[0] <= y or self.roads.shape[1] <= x or y < 0 or x < 0:
                print(f"[⚠️ Lane Error] Coordinates (x={x}, y={y}) out of bounds for roads shape {self.roads.shape}")
                return 0
            zone_id = self.roads[int(y), int(x)]
            print(f"[DEBUG] Zone at (x={x}, y={y}): {zone_id}")
            return zone_id
        except Exception as e:
            print(f"[⚠️ Lane Error] Failed to get zone at (x={x}, y={y}): {e}")
            return 0
