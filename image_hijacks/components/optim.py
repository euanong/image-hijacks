from __future__ import annotations

import typing
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torchopt
from jaxtyping import Float
from torch import Tensor
import torch.nn as nn


if typing.TYPE_CHECKING:
    from image_hijacks.config import Config
from image_hijacks.utils import Parameters, clip_norm
from image_hijacks.components.processor import Processor


class AttackOptimizer(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def step(self, processor: Processor, grads: Dict[str, Tensor]) -> None:
        """Given an image and (true or estimated) gradients, take one optimisation step
        with those gradients, updating parameters in-place."""
        ...


class TorchOptOptimizer(AttackOptimizer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.optimizer = config.torchopt_optimizer(config)
        self.optimizer_state = None
        self.orig_parameters: Optional[Dict[str, Tensor]] = None

    def step(self, processor: Processor, grads: Dict[str, Tensor]) -> None:
        # For now, we optimise all parameters.
        parameters = processor.get_parameter_dict()

        if self.optimizer_state is None:
            self.optimizer_state = self.optimizer.init(parameters)

        if self.config.clip_grad_mag is not None:
            grads = {
                k: clip_norm(v, maxnorm=self.config.clip_grad_mag)
                for k, v in grads.items()
            }

        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state
        )
        torchopt.apply_updates(parameters, updates, inplace=True)
        processor.clamp_params()


class IterFGSMOptimizer(AttackOptimizer):
    def step(self, processor: Processor, grads: Dict[str, Tensor]) -> None:
        parameters = processor.get_parameter_dict()
        for k in parameters.keys():
            parameters[k].sub_(self.config.iterfgsm_alpha * grads[k].sign())
        processor.clamp_params()
