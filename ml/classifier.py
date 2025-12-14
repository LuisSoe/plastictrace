import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from ml.config import CLASSES
from ml.preprocess import preprocess_bgr

class PlastiTraceClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()

    def _extract_state_dict(self, ckpt):
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # strip DataParallel prefix
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        return sd

    def _load_model(self, model_path: str):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

        ckpt = torch.load(model_path, map_location="cpu")
        sd = self._extract_state_dict(ckpt)
        model.load_state_dict(sd, strict=True)

        # force FP32
        model = model.float().to(self.device)
        return model

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        x = tensor.to(self.device, dtype=torch.float32, non_blocking=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs).item())
        label = CLASSES[idx]
        conf = float(probs[idx].item())
        probs_np = probs.detach().cpu().numpy().astype(np.float32)

        return {"label": label, "confidence": conf, "probs": probs_np.tolist()}

    def predict_from_bgr(self, frame_bgr) -> dict:
        t = preprocess_bgr(frame_bgr)
        return self.predict(t)
