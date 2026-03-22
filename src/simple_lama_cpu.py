from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from simple_lama_inpainting.utils.util import download_model, prepare_img_and_mask


class SimpleLamaCPU:
    DEFAULT_MODEL_URL = (
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    )

    def __init__(
        self,
        model_path: str | None = None,
        model_url: str | None = None,
    ) -> None:
        """
        Args:
            model_path: Ruta local al .pt de LaMa. Si es None, se descarga desde model_url.
            model_url: URL si no hay model_path. Por defecto DEFAULT_MODEL_URL.
        """
        url = model_url if model_url is not None else self.DEFAULT_MODEL_URL
        if model_path is not None:
            p = Path(model_path)
            if not p.is_file():
                raise FileNotFoundError(f"lama torchscript model not found: {model_path}")
            resolved = str(p)
        else:
            resolved = download_model(url)

        self.device = torch.device("cpu")
        self.model = torch.jit.load(resolved, map_location="cpu")
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image, mask):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)
