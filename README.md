# taller3_yolov8

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd proyecto_ocr
```

2. Crear entorno virtual (recomendado):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecutar

Ejecutar con "uvicorn src.fastapi:app --reload"
y luego abrir el navegador en http://localhost:8000/docs
