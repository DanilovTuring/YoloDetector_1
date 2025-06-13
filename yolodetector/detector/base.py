from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: Any) -> List[Tuple[str, float, Tuple[int,int,int,int]]]:
        """
        Realiza la deteccion de objetos de una imagen
        
        
        :param image: imagen de entrada (formato OpenCv o algo parecido)
        :retunr : lista de tuplas (nombre_clasem confianza, caja)
        """

        pass
    
    @abstractmethod
    def draw_detections(self, image: Any, detections: List[Tuple[str, float, Tuple[int,int,int,int]]]) -> Any:
        """
        Dibuja las cajas y etiquetas sobre la imagen original.

        :para image: Es la imagen de entrada
        :param detections: Es la lista de detecciones
        :return: Imagen con las detecciones ya dibujadas
        """
        pass

