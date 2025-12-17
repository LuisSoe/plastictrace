import cv2
import numpy as np
import torch
from ml.config import IMG_SIZE, MEAN, STD

def preprocess_bgr(frame_bgr):
    """
    frame_bgr -> torch.FloatTensor (1,3,224,224) in RGB, normalized.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = rgb.astype("float32") / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, 0)        # NCHW

    return torch.tensor(x, dtype=torch.float32)
