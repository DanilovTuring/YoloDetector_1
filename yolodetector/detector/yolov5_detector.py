#YOLODETECTOR_1/detector/yolov5_detector.py

import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Any



import numpy as np

import pandas as pd

from yolodetector.detector.base import BaseDetector
from yolodetector.config.setting import MODEL_PATH, DEVICE, CONFIDENCE_THRESHOLD
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
         
    def draw_detections(self, image: Any, detections: List[Tuple[str, float, Tuple[int,int,int,int]]]) -> Any:
      
        """
       Dibuja las detecciones sobre la imagen original.

       :param image: Es la imagen en formato OpenCv
       :param detections: Lista de detecciones (clase, confianza, caja)
       :return Imagen con cajas y etiquetas dibujadas. 
        """

        if image is None:
            raise ValueError("Verifica la carga de la imagen")
        
        annotated_image = draw_boxes(image.copy(), detections)
        return annotated_image

    def predict(self, image: Any ) -> List[Tuple[int, str, float, Tuple[int,int,int,int]]]:
   
        """
        Realiza la detección de objetos en una imagen usando el modelo YOLv5.

        param image; Imagen en formato OpenCV (numpy array)
        :return: Lisra de tuplas con (clase_id, confianza, (x1,y1,x2,y2))
        """

        #1. Preporcesar la imagen (resize y nomralización)
        preprocessed = preprocess_image(image)

        #2. Realizar la inferencia con el modelo
        results = self.model(preprocessed)[0]
        
        #3. Obtener las predicciones como DataFrame de pandas
        detections = results[:, :6].cpu().numpy()
        predictions = pd.DataFrame(results[0].cpu().numpy(), columns =
                                   ["xmin", "ymin", "xmax", "ymax", "confidence", "class"])  # columnas: xmin, ymin, xmax, ymax, confidence, class, name

        #4. Convertir a lista de tuplas
        output = []

        for _, row in predictions.iterrows():
            class_id = int(row['class'])
            confidence = float(row['confidence'])
            box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
            name = self.model.names[class_id]
            output.append((class_id, name, confidence, box))

        return output 
        
    def __str__(self) -> str:
        """
        Devuelve la representacion legible del lector
        """    
               
        model_name = Path(MODEL_PATH).name
        num_classes = len(self.model.names)
        return f"YOLOv5DETECTOR(model='{model_name}', device = '{self.device}', classes={num_classes})"
    

    
