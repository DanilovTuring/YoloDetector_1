import cv2
import numpy as np
import torch
from typing import List, Tuple, Any


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Convierte una imagen OpenCv en un tensor compatible para el modelo

    :param image: Imagen en formato BGR(OpenCv)
    return: Tensor listo para la inferencia (1,2,H,W)
    """

    # 1. Convertimos de BGR (OpenCv) a RGB (FORMATO PARA YOLOv5)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Redimensionamos el tama;o 
    resized = cv2.resize(image_rgb, (640, 640))

    # 3. Normalizamos los píxe;es a rango [0, 1]
    normalized = resized / 255.0

    # 4. Transponemos a forma (C, H, W) y convertimos a tipo float32
    tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).float()

    # 5. Agregamos una dimension extra para el batch (1 imagen)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def draw_boxes(image: Any, detections: List[Tuple[str, float, Tuple[int,int,int,int]]]) -> Any:
    image_with_boxes = image.copy()
    for label, confidence, box in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_with_boxes, (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (0, 255, 0), 2)
        
    return image_with_boxes