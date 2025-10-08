# src/slpt_validator.py
import cv2
import torch
import numpy as np
# ... (toàn bộ nội dung của tệp slpt_validator.py như trước) ...
import sys
import os
import torchvision.transforms.v2 as transforms

class SLPTValidator:
    def __init__(self, project_root, model_path, initial_points_path, device='cpu'):
        self.device = device
        self.cfg = self._load_slpt_config(project_root)
        self.model = self._load_model(model_path, initial_points_path)
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("[SLPT-VALIDATOR] Khởi tạo thành công.")

    def _load_slpt_config(self, project_root):
        if project_root not in sys.path: sys.path.append(project_root)
        try:
            from Config import cfg
            return cfg
        except ImportError as e: raise ImportError(f"LỖI: Không thể import cấu hình từ dự án SLPT: {e}")

    def _load_model(self, model_path, initial_points_path):
        try:
            from SLPT.SLPT import Sparse_alignment_network
            self.cfg.TRANSFORMER.NUM_DECODER = 12
            model = Sparse_alignment_network(num_point=self.cfg.WFLW.NUM_POINT, d_model=self.cfg.MODEL.OUT_DIM, trainable=False, return_interm_layers=True, dilation=self.cfg.MODEL.DILATION, nhead=self.cfg.TRANSFORMER.NHEAD, feedforward_dim=self.cfg.TRANSFORMER.FEED_DIM, initial_path=initial_points_path, cfg=self.cfg)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
            model.eval()
            return model
        except Exception as e: raise RuntimeError(f"LỖI: Không thể tải model SLPT: {e}")

    def validate_and_visualize(self, face_crop):
        if face_crop.size == 0: return False, np.zeros((self.cfg.MODEL.IMG_SIZE, self.cfg.MODEL.IMG_SIZE, 3), dtype=np.uint8)
        validation_result = False
        debug_image = cv2.resize(face_crop, (self.cfg.MODEL.IMG_SIZE, self.cfg.MODEL.IMG_SIZE))
        try:
            face_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            face_tensor = self.transform(face_rgb).unsqueeze(0)
            with torch.no_grad(): landmarks_list = self.model(face_tensor)
            predicted_landmarks = landmarks_list[2][0, -1].cpu().numpy()
            for point in predicted_landmarks: cv2.circle(debug_image, (int(point[0] * self.cfg.MODEL.IMG_SIZE), int(point[1] * self.cfg.MODEL.IMG_SIZE)), 1, (0, 255, 255), -1)
            left_eye, right_eye = predicted_landmarks[60], predicted_landmarks[72]
            eye_dist_pixels = np.linalg.norm(left_eye - right_eye) * self.cfg.MODEL.IMG_SIZE
            if eye_dist_pixels > 30: validation_result = True
        except Exception as e: print(f"[SLPT-VALIDATOR-ERROR] Lỗi: {e}"); validation_result = False
        return validation_result, debug_image