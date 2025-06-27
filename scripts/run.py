import sys
import os
import cv2

#Configuracion - Añade el directorio raíz al path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from yolodetector.detector.yolov5_detector import YOLOv5Detector
    from yolodetector.utils.io_utils import load_image, save_image
except ImportError as e:
    print(f"Error de importación:" {str(e)})
    print(f"Verifique si la estructura de directorios es correcta:")
    print("- El directiorio 'yolodetector' debe estar en la raiz del proyecto")
    print("- Debe contener los archivos __init__.py necesarios")
    sys.exit(1)

