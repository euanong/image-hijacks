from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL.Image import Image
from jaxtyping import Float
from lightning import LightningModule
from torch import Tensor, nn
import functorch

from typing import TYPE_CHECKING, Any, Dict, Optional

import wandb

from image_hijacks.data import WrapperContextDataModule
from image_hijacks.models import AbstractLensModel
from image_hijacks.utils import all_equal, tensor_to_image

if TYPE_CHECKING:
    from image_hijacks.config import Config


class AttackDriver(LightningModule, ABC):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = config
        self.step = 0

        self.processor = self.config.processor_factory(self.config).requires_grad_(
            False
        )

        self.train_models: Dict[str, AbstractLensModel] = nn.ModuleDict(  # type: ignore
            self.config.target_models_train
        )
        self.train_models.requires_grad_(False)
        self.eval_models: Dict[str, AbstractLensModel] = nn.ModuleDict(  # type: ignore
            self.config.target_models_eval
        )
        self.eval_models.requires_grad_(False)

        self.loss_fn = self.config.loss_fn_factory(self.config)
        self.gradient_estimator = self.config.gradient_estimator_factory(self.config)
        self.attack_optimizer = self.config.attack_optimizer_factory(self.config)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        keys = list(checkpoint["state_dict"].keys())
        for k in keys:
            if k.startswith("train_models") or k.startswith("eval_models"):
                del checkpoint["state_dict"][k]
        checkpoint["step"] = self.step

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        for key in self.state_dict().keys():
            if key.startswith("train_models") or key.startswith("eval_models"):
                checkpoint["state_dict"][key] = self.state_dict()[key]
        self.step = checkpoint["step"]

    def configure_optimizers(self) -> Any:
        return None

    @abstractmethod
    def get_datamodule(self) -> WrapperContextDataModule:
        ...

    def save_images(self, name):
        self.processor.save_images(name, self)
