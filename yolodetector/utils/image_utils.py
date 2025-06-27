import cv2
import numpy as np
import torch
from typing import List, Tuple, Any


def preprocess_image(image: np.ndarray, target_size: int = 640) -> torch.Tensor:
    """
    Preprocesa una imagen para YOLOv5 manteniendo el aspecto ratio

    Args:
        image: Imagen de entrada en formato BGR (OpenCV)
        target_size: Tamaño del objetivo para el resize (default:640)

    Returns:
        Tensor de PyTorch normalizado en formato (1, 3, H, W)

    """

    # Convertimos de BGR (OpenCv) a RGB (FORMATO PARA YOLOv5)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Obtener las dimensiones originales
    h, w = image_rgb[:2]

    # Calcular escala manteniendo aspecto ratio
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Redimensionamos el tamaño 
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crear canvas cuadrado
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, new_w] = resized

    # Normalizamos y convertimos a tensor
    normalized = canvas.astype(np.float32 / 255.0)
    tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).float()
    
    return tensor_image.unsqueeze(0)

    

def draw_boxes(image: Any, detections: List[Tuple[str, float, Tuple[int,int,int,int]]]) -> Any:
    image_with_boxes = image.copy()
    for label, confidence, box in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_with_boxes, (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (0, 255, 0), 2)
        
    return image_with_boxes