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
    print(f"Error de importación: {str(e)}")
    print(f"Verifique si la estructura de directorios es correcta:")
    print("- El directiorio 'yolodetector' debe estar en la raiz del proyecto")
    print("- Debe contener los archivos __init__.py necesarios")
    sys.exit(1)

def main():
    try:
        # Configura paths absolutos
        resources_dir = os.path.join(project_root, 'resources')
        input_path = os.path.join(resources_dir, 'outputs', 'zinedine.jpg')
        output_path = os.path.join(resources_dir, 'outputs', 'zinedine_output.jpg')

        #Cargar imagen
        image = load_image(input_path)
        print(f"imagen cargada: {input_path}")

        # Inicializar detector
        detector = YOLOv5Detector()
        print(f"Detector inicializado:\n{detector}")

        # Detectar objetos
        detections = detector.detect(image)
        print(f"Detecciones encontradas: {len(detections)}")

        # Visualizar resultados
        annotated_image = detector.draw_detections(image, detections)
        cv2.imshow("Resultados", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Guardar resultados
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image(output_path, annotated_image)
        print(f"Resultados guardados en: {output_path}")
    
    except Exception as e:
        print(f"Error durante la ejecición: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()