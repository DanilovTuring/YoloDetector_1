import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from yolodetector.detector.yolov5_detector import YOLOv5Detector
from yolodetector.utils.io_utils import load_image, save_image


# 1. Cargamos la imagen
image_path = "resources/samples/zinedine.jpg"
image = load_image(image_path)

# 2. Inicializamos el detector
detector = YOLOv5Detector()
print(detector)

# 3. Detectar y dibujar
detections  = detector.detect(image)
annotated_image = detector.draw_detections(image, detections)

# 4. Mostramos y guardamos
cv2.imshow("Detecciones", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_image("resources/outputs/zinedine_output.jpg", annotated_image)
