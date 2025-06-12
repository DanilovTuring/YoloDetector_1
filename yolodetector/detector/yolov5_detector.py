#YOLODETECTOR_1/detector/yolov5_detector.py

import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Any


import numpy as np

from yolodetector.detector.base import BaseDetector
from yolodetector.config.setting import MODEL_PATH, DEVICE
from yolodetector.utils.image_utils import preprocess_image, draw_boxes

class YOLOv5Detector (BaseDetector):
    
    def __init__(self, model_path: str =  MODEL_PATH, device: str = DEVICE):
        self.device= device
        self.model= torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.eval()
 

    def detect(self, image: Any) -> List[Tuple[str, float, Tuple[int, int, int, int ]]]:
              
        """
        Detecta objetos en una imagen y devuelve una lista con:
        (nombre_clase, confianza, caja).
        """

        # 1. Obtener las predicciones crudas
        predictions = self.predict(image)

        # 2. Obtener los nombres legibles de las clases
        #(opc) class_names = self.model.names

        # 3. traducir ids a nombres
        results = []
      
        for class_id, confidence, box in predictions:
            label = self.model.names[class_id]
            results.append((label, confidence, box))

        return results
         
    

    def predict(self, image: Any ) -> List[Tuple[int, float, Tuple[int,int,int,int]]]:
   
        """
        Realiza la detección de objetos en una imagen usando el modelo YOLv5.

        param image; Imagen en formato OpenCV (numpy array)
        :return: Lisra de tuplas con (clase_id, confianza, (x1,y1,x2,y2))
        """

        #1. Preporcesar la imagen (resize y nomralización)
        preprocessed = preprocess_image(image)

        #2. Realizar la inferencia con el modelo
        results = self.model(preprocessed)

        #3. Obtener las predicciones como DataFrame de pandas
        predictions = results.pandas().xyxy[0]  # columnas: xmin, ymin, ymin xmax, ymax, confidence, class, name

        #4. Convertir a lista de tuplas
        output = []

        for _, row in predictions.iterrows():
            class_id = int(row['class'])
            confidence = float(row['confidence'])
            box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
            output.append((class_id, confidence, box))

        return output 

    
    

    
