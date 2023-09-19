from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, TypeVar
from einops import rearrange, reduce, repeat

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
from torch.func import grad_and_value, vmap

from typing import TYPE_CHECKING

from tqdm import tqdm

from image_hijacks.utils import Parameters

if TYPE_CHECKING:
    from image_hijacks.config import Config

T = TypeVar("T")


class GradientEstimator(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def grad_and_value(
        self,
        loss_fn: Callable[[Parameters], Tuple[Float[Tensor, ""], T]],
        params: Parameters,
    ) -> Tuple[Parameters, Tuple[Float[Tensor, ""], T]]:
        """Estimate gradient of loss_fn at pixel_values"""
        ...


class ExactGradientEstimator(GradientEstimator):
    def grad_and_value(
        self,
        loss_fn: Callable[[Parameters], Tuple[Float[Tensor, ""], T]],
        parameters: Parameters,
    ) -> Tuple[Parameters, Tuple[Float[Tensor, ""], T]]:
        return grad_and_value(loss_fn, has_aux=True)(parameters)
