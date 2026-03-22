from ultralytics import YOLO
from PIL import Image
import io

# Cargar modelo entrenado
model = YOLO("models/best.pt")

def inferir_imagen(ruta_imagen: str):
    """
    Carga el modelo entrenado y realiza inferencia en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen a procesar
    
    Returns:
        Resultados de la inferencia
    """

    # Realizar inferencia
    results = model(ruta_imagen)
    
    return results

def inferir_imagen_bytes(imagen_bytes: bytes):
    """
    Carga el modelo entrenado y realiza inferencia en una imagen desde bytes.
    
    Args:
        imagen_bytes (bytes): Imagen en formato bytes
    
    Returns:
        Resultados de la inferencia
    """

    # Convertir bytes a imagen PIL
    imagen = Image.open(io.BytesIO(imagen_bytes))
    
    # Realizar inferencia
    results = model(imagen)
    
    return results
