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

    

def draw_boxes(
    image: np.ndarray,
    detections: List[Tuple[str, float, tuple[int, int, int, int]]],
    show_labels: bool = True,
    show_conf: bool = True,
    box_color: Tuple[int, int, int] = (255, 0, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    annotated_img = image.copy()

    for label, confidence, (x1, y1, x2, y2) in detections:
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            box_color,
            box_thickness
        )

        if show_labels or show_conf:
            text_parts = []
            if show_labels:
                text_parts.append(label)
            if show_conf:
                text_parts.append(f"{confidence:.2f}")
            text = " ".join(text_parts)

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness
            )

            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                box_color,
                -1
            )

            cv2.putText(
                annotated_img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
    return annotated_img

def load_image(image_path: str) -> np.ndarray:

    """
    Carga una imagen desde disco con validación

    Args:
        image_path: Ruta de la imagen
    
    Returns:
        Imagen en formato BRG (OpenCV)

    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return img

def save_image(output_path: str, image: np.ndarray) -> None:
    """
    Guarda ina imagen en disco con validación

    Args:
        output_path: Ruta de destino
        image: Imagen a guardar (BGR)

    Raises:
        ValueError: Si falla en guardar
    """
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"No se puedo guardar la imagen en: {output_path}")
    
def load_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen desde disco con validación.

    Args:
        image_path: Ruta a la imagen
    
    Returns:
        Imagen en formato BGR (OpenCV)
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return img