#YOLODETECTOR_1/detector/yolov5_detector.py

import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Any

from yolodetector.detector.base import BaseDetector
from yolodetector.config.setting import MODEL_PATH, DEVICE
from yolodetector.utils.image_utils import preprocess_image, draw_boxes

class YOLOv5Detector (BaseDetector):
    
    def __init__(self, model_path: str =  MODEL_PATH, device: str = DEVICE):
      self.device= device
      self.model= torch.hub.load('ultralytics/yolo5', 'custom', path=model_path)
      self.model.to(self.device)
      self.model.eval()
      
        

    
