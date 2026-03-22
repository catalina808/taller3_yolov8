from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inferencia import (
    eliminar_postes,
    inferir_imagen_bytes,
    generate_image_with_poles,
    generate_mask_image_from_bytes,
    generate_pretty_mask_image_from_bytes,
)
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import cv2


app = FastAPI()

@app.post("/prediccion/")
async def prediccion(archivo: UploadFile = File(...)):
    """
    Endpoint que recibe una imagen y realiza la inferencia.
    
    Args:
        archivo: Archivo de imagen subido
    
    Returns:
        Resultados de la inferencia en formato JSON
    """
    # Leer la imagen como bytes
    imagen_bytes = await archivo.read()
    
    # Realizar inferencia
    results = inferir_imagen_bytes(imagen_bytes)
    
    # Procesar resultados
    detecciones = []
    for result in results:
        for box in result.obb:
            detecciones.append({
                "clase": int(box.cls[0]),
                "confianza": float(box.conf[0]),
                "coordenadas": box.xyxyxyxy[0].tolist()
            })
    
    return JSONResponse(content={
        "imagen": archivo.filename,
        "detecciones": detecciones,
        "total": len(detecciones)
    })


@app.post("/prediccion-imagen/")
async def prediccion_imagen(archivo: UploadFile = File(...)):
    imagen_bytes = await archivo.read()

    results = inferir_imagen_bytes(imagen_bytes)
    img = results[0].plot()

    image_pil = Image.fromarray(img)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=resultado.png"}
    )


@app.post("/deteccion-postes/")
async def deteccion_postes(archivo: UploadFile = File(...)):
    imagen_bytes = await archivo.read()
    img_bgr = generate_image_with_poles(imagen_bytes)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=postes_detectados.png"},
    )


@app.post("/generar-mascara/")
async def generar_mascara(archivo: UploadFile = File(...)):
    imagen_bytes = await archivo.read()
    mask = generate_mask_image_from_bytes(imagen_bytes)
    image_pil = Image.fromarray(mask, mode="L")
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=mascara_postes.png"},
    )


@app.post("/generar-mascara-overlay/")
async def generar_mascara_overlay(archivo: UploadFile = File(...)):
    imagen_bytes = await archivo.read()
    img_bgr = generate_pretty_mask_image_from_bytes(imagen_bytes)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=mascara_overlay.png"},
    )


@app.post("/eliminar-postes/")
async def eliminar_postes_endpoint(archivo: UploadFile = File(...)):
    imagen_bytes = await archivo.read()
    img_bgr = eliminar_postes(imagen_bytes)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=sin_postes.png"},
    )
