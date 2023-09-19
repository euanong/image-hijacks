from __future__ import annotations
from enum import Enum
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Optional,
    Callable,
    Union,
)
from einops import rearrange
import numpy as np
import torch
from torch import nn

from transformers.models.blip_2.processing_blip_2 import Blip2Processor

from jaxtyping import Float16, Shaped
from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import re

if TYPE_CHECKING:
    from image_hijacks.config import Config

# Path to root dir (with run.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

Parameters = Dict[str, Tensor]
T = TypeVar("T")
U = TypeVar("U")


def tensor_to_image(img_tensor: Float16[Tensor, "1 c h w"]) -> Image.Image:
    return to_pil_image(rearrange(img_tensor, "() c h w -> c h w"))


def quantise_image(
    img_tensor: Float16[Tensor, "b c h w"]
) -> Float16[Tensor, "b c h w"]:
    # Following PyTorch: https://pytorch.org/vision/main/_modules/torchvision/utils.html#save_image
    return (
        (img_tensor * 255 + 0.5).clamp(0, 255).to(torch.uint8).to(torch.half).div(255)
    )


def get_full_attention_mask(
    embeddings: Shaped[Tensor, "b seq_len ..."]
) -> Shaped[Tensor, "b seq_len"]:
    tgt_shape = embeddings.shape[:2]
    return torch.ones(*tgt_shape, device=embeddings.device, dtype=torch.bool)


def clip_norm(
    x: Shaped[Tensor, "*grad_dims"], maxnorm: float
) -> Shaped[Tensor, "*grad_dims"]:
    x_shape = x.shape
    flat_grads = rearrange(x, "... -> () (...)")
    renorm_grads = torch.renorm(flat_grads, p=2, maxnorm=maxnorm, dim=0)
    return torch.reshape(renorm_grads, x_shape)


class Option:
    @staticmethod
    def map(x: Optional[T], f: Callable[[T], U]) -> Optional[U]:
        return f(x) if x is not None else None

    @staticmethod
    def value(x: Optional[T], default: T) -> T:
        return x if x is not None else default

    @staticmethod
    def get_first_if_exists(xs: Sequence[Optional[T]]) -> Optional[T]:
        return next((i for i in xs if i is not None), None)


def all_equal(
    xs: Sequence,
    compare: Callable[[T, T], bool] = lambda x, y: x == y,
) -> bool:
    if xs is []:
        return True
    return all(compare(x, xs[0]) for x in xs)


def load_config_list(
    config_path: Path,
) -> List[Tuple[str, Callable[[], Config]]]:
    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    module_name = f"experiment_config_{str(int(time.time()))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)
    return config_module.gen_configs()


def load_model_with_cache(
    model_fn: Callable[[], nn.Module],
    model_id_components: Sequence,
    cache_dir: Path = PROJECT_ROOT / "downloads/cache",
) -> nn.Module:
    cache_dir.mkdir(exist_ok=True)
    hash = hashlib.sha256()
    for component in model_id_components:
        hash.update(str(component).encode("utf-8"))
    model_name = hash.hexdigest()
    cache_path = cache_dir / f"{model_name}.pt"
    if cache_path.exists():
        model = torch.load(cache_path).eval()
    else:
        model = model_fn()
        torch.save(model, cache_path)
    return model


def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    return tensor.numpy()


def remove_no_grad(f):
    if (
        f.__closure__ is not None
        and len(f.__closure__) == 2
        and (
            f.__closure__[0].cell_contents.__self__.__class__
            == torch.autograd.grad_mode.no_grad
        )
    ):
        return f.__closure__[1].cell_contents
    else:
        return f


def get_best_last_ckpts(experiment, name):
    ckpt_base = (
        PROJECT_ROOT / "experiments" / experiment / "logs" / name / "checkpoints"
    )
    ckpts = list(ckpt_base.glob("*step=*"))
    ckpt_stats = [
        (
            re.match(
                r"epoch=[0-9]*-step=([0-9]*)-val_[a-z_]*acc=([0-9.]*)(?:-v([0-9]*))?.ckpt",
                i.name,
            ).groups(),
            i,
        )
        for i in ckpts
    ]
    ckpt_stats = [
        (float(acc), int(epoch), int(version) if version is not None else 0, i)
        for (epoch, acc, version), i in ckpt_stats
    ]

    max_version = max([v for _, _, v, _ in ckpt_stats])
    ckpt_stats_max = [(a, e, v, i) for a, e, v, i in ckpt_stats if v == max_version]
    _, _, _, best_path = sorted(ckpt_stats_max, reverse=True)[0]
    last_path = ckpt_base / "last.ckpt"
    return {"best": best_path, "last": last_path}


def split_train_val_test(
    ds: List[T], seed: int, train_split: int, val_split: int, test_split: int
) -> Dict[str, List[T]]:
    Random(seed).shuffle(ds)
    ptr = 0
    train_data = ds[ptr : ptr + train_split]
    ptr += train_split
    val_data = ds[ptr : ptr + val_split]
    ptr += val_split
    test_data = ds[ptr : ptr + test_split]
    return {"train": train_data, "val": val_data, "test": test_data}
