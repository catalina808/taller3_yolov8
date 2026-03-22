from __future__ import annotations

from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
from src.simple_lama_cpu import SimpleLamaCPU

# Cargar modelo entrenado
model = YOLO("models/best.pt")

_simple_lama = None


def _get_simple_lama():
    global _simple_lama
    if _simple_lama is None:
        _simple_lama = SimpleLamaCPU()
    return _simple_lama


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

def detect_poles_obb(model, imagen_bytes: bytes, conf_threshold=0.25, target_classes=[1]):
    """
    Detecta postes usando YOLOv8-OBB a partir de la imagen en bytes.

    Los modelos OBB retornan bounding boxes rotados con 4 vertices.
    result.obb contiene:
      - .xyxyxyxy: 4 vertices del poligono rotado [N, 4, 2]
      - .xywhr: centro (x,y), ancho, alto y angulo de rotacion [N, 5]
      - .cls: clase de cada deteccion
      - .conf: confianza de cada deteccion

    Args:
        model: Modelo YOLO cargado.
        imagen_bytes: Imagen codificada (JPEG, PNG, etc.), igual que en inferir_imagen_bytes.
    """
    pil_imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    results = model(pil_imagen, conf=conf_threshold, verbose=False)
    image = cv2.cvtColor(np.array(pil_imagen), cv2.COLOR_RGB2BGR)

    detections = []
    for result in results:
        obb = result.obb

        if obb is None or len(obb) == 0:
            continue

        for i in range(len(obb)):
            class_id = int(obb.cls[i].item())
            confidence = float(obb.conf[i].item())

            if class_id in target_classes and confidence >= conf_threshold:
                # Obtener los 4 vertices del poligono OBB
                polygon = obb.xyxyxyxy[i].cpu().numpy().astype(int)  # shape: (4, 2)

                # Bounding box alineado con ejes (para referencia)
                x_coords = polygon[:, 0]
                y_coords = polygon[:, 1]
                bbox_aligned = (
                    int(x_coords.min()), int(y_coords.min()),
                    int(x_coords.max()), int(y_coords.max())
                )

                detections.append({
                    'polygon': polygon,
                    'bbox_axis_aligned': bbox_aligned,
                    'confidence': confidence,
                    'class_id': class_id
                })

    return detections, image


CONFIDENCE_THRESHOLD = 0.25
# clases:
#   0: 'Casa'
#   1: 'postes'
# Queremos detectar SOLO postes -> clase 1
TARGET_CLASSES = [1]
MASK_DILATION_PX = 15


def generate_image_with_poles(
    imagen_bytes: bytes,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: list | None = None,
) -> np.ndarray:
    """
    Detecta postes en la imagen (bytes) y devuelve la imagen BGR con polígonos OBB dibujados.

    Returns:
        np.ndarray: Imagen en BGR (mismo formato que cv2), lista para guardar o convertir a RGB.
    """
    classes = TARGET_CLASSES if target_classes is None else target_classes
    detections, image = detect_poles_obb(
        model, imagen_bytes, conf_threshold, classes
    )

    img_vis = image.copy()
    for det in detections:
        pts = det["polygon"].reshape((-1, 1, 2))
        cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        x1, y1 = det["polygon"][0]
        cv2.putText(
            img_vis,
            f"Poste {det['confidence']:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return img_vis


def generate_mask_from_obb(image_shape, detections, dilation_px=15):
    """
    Genera una mascara binaria a partir de poligonos OBB.

    Usa cv2.fillPoly con los 4 vertices del OBB para crear una
    mascara mas precisa que un bounding box rectangular.

    Args:
        image_shape: (H, W, C) de la imagen original
        detections: Lista con campo 'polygon' (4 vertices)
        dilation_px: Pixeles de dilatacion

    Returns:
        mask: Mascara binaria (H x W), uint8, 0 o 255
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        polygon = det['polygon']  # shape: (4, 2)
        cv2.fillPoly(mask, [polygon], 255)

    # Dilatar la mascara para cubrir bordes y sombras
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def generate_mask_image_from_bytes(
    imagen_bytes: bytes,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: list | None = None,
    dilation_px: int = MASK_DILATION_PX,
) -> np.ndarray:
    """
    Detecta postes en la imagen (bytes) y devuelve una máscara binaria (H, W), uint8, valores 0 o 255.
    """
    classes = TARGET_CLASSES if target_classes is None else target_classes
    detections, image = detect_poles_obb(
        model, imagen_bytes, conf_threshold, classes
    )
    return generate_mask_from_obb(image.shape, detections, dilation_px=dilation_px)


def generate_pretty_mask_image_from_bytes(
    imagen_bytes: bytes,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: list | None = None,
    dilation_px: int = MASK_DILATION_PX,
) -> np.ndarray:
    """
    Detecta postes, genera la máscara OBB y devuelve la imagen original con la máscara
    superpuesta en rojo (mezcla 50/50), en BGR.
    """
    classes = TARGET_CLASSES if target_classes is None else target_classes
    detections, image = detect_poles_obb(
        model, imagen_bytes, conf_threshold, classes
    )
    mask = generate_mask_from_obb(image.shape, detections, dilation_px=dilation_px)
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]
    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    return blended


def inpaint_with_lama(image_bgr, mask):
    """
    Aplica inpainting con LaMa.
    Pixeles con valor 255 en la mascara seran reconstruidos.
    """
    if not np.any(mask):
        return image_bgr.copy()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask).convert("L")

    result_pil = _get_simple_lama()(pil_image, pil_mask)

    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    return result_bgr


def eliminar_postes(
    imagen_bytes: bytes,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    target_classes: list | None = None,
    dilation_px: int = MASK_DILATION_PX,
) -> np.ndarray:
    """
    Detecta postes, genera máscara OBB e inpainting con LaMa. Devuelve imagen BGR sin las regiones de poste.
    Si no hay detecciones, devuelve la imagen original.
    """
    classes = TARGET_CLASSES if target_classes is None else target_classes
    detections, image = detect_poles_obb(
        model, imagen_bytes, conf_threshold, classes
    )
    mask = generate_mask_from_obb(image.shape, detections, dilation_px=dilation_px)
    return inpaint_with_lama(image, mask)

