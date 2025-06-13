import cv2
import os

def load_image(image_path: str):
    if not os.path.exists(image_path):
            raise FileExistsError(f"No se encontró la imagen en la ruta: {image_path}")
    
    image = cv2.imread(image_path)

    if image is None:
          raise ValueError(f"No se encontró la imagen en la ruta: {image_path}")
    
    return image


def save_image(output_path: str, image):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exis_ok=True)

    cv2.imwrite(output_path, image)
    