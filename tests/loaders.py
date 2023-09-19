from functools import lru_cache
from pathlib import Path
from typing import Dict

import requests
import torch
from PIL import Image
from torch import nn

from image_hijacks import models
from image_hijacks.models import AbstractLensModel
from image_hijacks.models.blip2 import (
    Blip2FlanT5Xl,
    Blip2LensDecOnly,
    Blip2LensEncDec,
    Blip2Opt2p7b,
    InstructBlipFlanT5Xl,
    InstructBlipVicuna7b,
)
from image_hijacks.models.llava import LlavaLlama2_13b

root = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def load_model(model_name) -> AbstractLensModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2-opt-2.7b":
        return Blip2Opt2p7b.load_model(
            model_dtype=torch.float,
        ).to(device)
    elif model_name == "blip2-flan-t5-xl":
        return Blip2FlanT5Xl.load_model(
            model_dtype=torch.float,
        ).to(device)
    elif model_name == "instructblip-flan-t5-xl":
        return InstructBlipFlanT5Xl.load_model(
            model_dtype=torch.float,
        ).to(device)
    elif model_name == "instructblip-vicuna-7b":
        return InstructBlipVicuna7b.load_model(
            model_dtype=torch.float,
        ).to(device)
    elif model_name == "llava-llama2-13b":
        return LlavaLlama2_13b.load_model(
            model_dtype=torch.float,
        ).to(device)
    else:
        raise NotImplementedError()


IMG_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
IMG = Image.open(requests.get(IMG_URL, stream=True).raw).convert("RGB")
