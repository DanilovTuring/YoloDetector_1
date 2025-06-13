import torch
from pathlib import Path

# Ruta del modelo Yolov5s
MODEL_PATH = Path("resources/model/yolov5s.pt")

# Detectar GPU disponible o usar CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Umbral mínimo ded confianza para mostrar detecciones
CONFIDENCE_THRESHOLD = 0.25