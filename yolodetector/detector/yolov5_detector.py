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
    
    def __init__(self, model_path: str =  MODEL_PATH, device: str = DEVICE, condifedence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Inicializa el detector

        Args: 
            model_path: Ruta dal modelo personalizado (.pt)
            device: Dispositivo para inferencia ('cpu' o 'cuda')
            confidence_thresgold: Umbral minímo de confianza
        """
        
        self.device= device
        self.confidence_threshold = condifedence_threshold
        try:

            self.model= torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, autoshape = True)
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = condifedence_threshold 

        except Exception as e:
            raise RuntimeError(f"Error al cargar el Modelo Yolov5: {str(e)}")
        


 

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
        Realiza la predicción y filtra los resultados

        Args:
            image: La imagen de entrada

            Returns:
                Lista de tu´las (class_id, label, confidence, bounding_box)
        """

        if isinstance(image, str): #Si es path carga la imagen
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen  desde {image}")

        with torch.no_grad(): # desactiva gradientes para la inferencia
            try:
                results = self.model(image)
            except Exception as e:
                raise RuntimeError(f"Error durante la predicción {str(e)}")

        #Extraer y filtrar detecciones
        detections = results.xyxy[0].cpu().numpy()
        output = []

        for det in detections:
            if len(det) >= 6:
                xmin, ymin, xmax, ymax, conf, cls_id = det[:6]

                #Filtrar por confianza y convertir tipos
                if conf < self.confidence_threshold:
                    continue

                try:
                    cls_id = int(cls_id)
                    conf = float(conf)
                    label = self.model.names.get(cls_id, f"Unknown_{cls_id}")
                    box = tuple(map(int,[xmin, ymin, conf, box]))
                    output.append((cls_id, label, conf, box))


                except Exception as e:
                    print(f"Error procesando detección: {e}") 
                    continue
        return output
    
    def __str__(self) -> str:
        """
        Devuelve la representacion legible del lector
        """    
               
        model_name = Path(MODEL_PATH).name
        num_classes = len(self.model.names)
        return f"YOLOv5DETECTOR(model='{model_name}', device = '{self.device}', classes={num_classes})"
    

    