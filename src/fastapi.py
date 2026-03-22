from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inferencia import inferir_imagen_bytes
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import numpy as np



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
