#YOLODETECTOR_1/detector/yolov5_detector.py

import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Any



import numpy as np

import pandas as pd

from typing import Optional

from yolodetector.detector.base import BaseDetector
from yolodetector.config.setting import MODEL_PATH, DEVICE, CONFIDENCE_THRESHOLD
from yolodetector.utils.image_utils import preprocess_image, draw_boxes

class YOLOv5Detector (BaseDetector):
    
    def __init__(self, model_path: str =  MODEL_PATH, device: str = DEVICE, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Inicializa el detector

        Args: 
            model_path: Ruta dal modelo personalizado (.pt)
            device: Dispositivo para inferencia ('cpu' o 'cuda')
            confidence_threshold: Umbral minímo de confianza
        """
        
        self.device= device
        self.confidence_threshold = confidence_threshold
        try:

            self.model= torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, autoshape = True)
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = confidence_threshold 

        except Exception as e:
            raise RuntimeError(f"Error al cargar el Modelo Yolov5: {str(e)}")
        


 

    def detect(self, image: Any, confidence_threshold: Optional[float] = None) -> List[Tuple[str, float, Tuple[int, int, int, int ]]]:
              
        """
        Detecta objetos en una imagen.

        Args:
            image: Imagen de entrada(nump array, path o tensor)
            confidence_threshold: Umbral opcional para filtrar detecciones

            
        Returns:
            Lista de tuplas (nombre_clase, confianza, bounding_box )
        """


        if confidence_threshold is not None:
            self.model.conf = confidence_threshold

        predictions = self.predict(image)
        return [(label, confidence, box) for _, label, confidence, box in predictions]
        
         
    def draw_detections(self, image: Any, detections: List[Tuple[str, float, Tuple[int,int,int,int]]]) -> Any:
      
        """
       Dibuja las detecciones sobre la imagen original.

       Args:
            image: Imagen original (np.array)
            detections: Lista de detecciones (label, confidence, bounding box)
       
        Returns:
            Imagen con cajas y etiquetas dibujadas. 
        """

        if image is None:
            raise ValueError("Verifica la carga de la imagen")
        
        if not detections:
            return image.copy()
        
        return draw_boxes(image.copy(), detections)
        

    def predict(self, image: Any ) -> List[Tuple[int, str, float, Tuple[int,int,int,int]]]:
   
        """
        Realiza la predicción y filtra los resultados

        Args:
            image: La imagen de entrada

            Returns:
                Lista de tuplas (class_id, label, confidence, bounding_box)
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
                    label = self.model.names.get(cls_id, f"unknown_{cls_id}")
                    box = tuple(map(int,[xmin, ymin, xmax, ymax]))
                    output.append((cls_id, label, conf, box))


                except Exception as e:
                    print(f"Error procesando detección: {e}") 
                    continue
        return output
    
    def __str__(self) -> str:
        """
         Representacion legible del detector
        """     
        model_name = Path(self.model.weights.name if hasattr(self.model, 'weights') else MODEL_PATH).name
        num_classes = len(self.model.names)
        return (f"YOLOv5DETECTOR(model='{model_name}', device = '{self.device}', " 
                f"clases={num_classes}, conf_thresh={self.confidence_threshold})")
    

    