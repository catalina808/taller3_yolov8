from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def main():

    os.makedirs("models", exist_ok=True)

    model = YOLO("yolov8s-obb.pt")
    results = model.train(
        data="dataset/data.yaml",
        epochs=120,
        patience=25,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        # augmentations
        mosaic=0.5,
        mixup=0.05,
        degrees=5.0,
        scale=0.3,
        project="runs/obb",
        name="casa_postes_detector"
    )

    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"

    if best_weights_path.exists():
        shutil.copy(best_weights_path, "models/best.pt")
        print("Pesos del modelo guardados en: models/best.pt")
    else:
        print("No se encontraron los pesos del modelo")

    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()
