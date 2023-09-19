from __future__ import annotations
from abc import ABC, abstractmethod
from lightning import LightningModule
import torch.nn as nn

from jaxtyping import Float, Int64
import torch
from torch import Tensor

from typing import TYPE_CHECKING, Tuple, Union
from pathlib import Path

import wandb
from image_hijacks.utils import Parameters, all_equal, tensor_to_image
from image_hijacks.utils.factory import Factory
from image_hijacks.utils.patching import get_patches, set_patches

from einops import repeat

if TYPE_CHECKING:
    from image_hijacks.config import Config


class Processor(nn.Module, ABC):
    def __init__(self, config: Config) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, image: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        """Given input image, apply some transformation (e.g. add a learnable patch,
        add learnable noise)
        """
        ...

    def __call__(self, image: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        return super(Processor, self).__call__(image)

    @abstractmethod
    def save_images(self, name: str, trainer: LightningModule) -> None:
        ...

    def save_individual_image(
        self,
        image: Float[Tensor, "b c h w"],
        name: str,
        trainer: LightningModule,
        tag: str = "adversarial_image",
        caption: str = "Adversarial image",
    ):
        # Save image to TB
        trainer.logger.experiment.add_image(  # type: ignore
            caption,
            image[0].cpu(),
            dataformats="CHW",
            global_step=trainer.step,
        )
        # Save image to filesystem
        img_dir = (
            Path(trainer.trainer.logger.save_dir) / trainer.trainer.logger.version / "imgs"  # type: ignore
        )
        img_dir.mkdir(parents=True, exist_ok=True)
        img = tensor_to_image(image)
        img.save(img_dir / f"{name}.png")
        # Save image to WandB
        if len(trainer.loggers) > 1:
            trainer.loggers[1].experiment.log(  # type: ignore
                {tag: wandb.Image(img, caption=caption)}
            )

    def get_parameter_dict(self) -> Parameters:
        return dict(self.named_parameters())

    @abstractmethod
    def clamp_params(self):
        """Clamp parameter values if needed after update"""
        ...


class LearnedImageProcessor(Processor):
    init_image: Tensor

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.register_buffer(
            "init_image", config.lift_to_model_device_dtype(config.init_image(config))
        )
        self.learned_image = nn.Parameter(data=self.init_image.clone())

    def forward(self, image: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        # print("training" if self.training else "eval")
        return self.learned_image

    def save_images(self, base_name: str, trainer: LightningModule) -> None:
        # Save / log image with filename `base_name.png`
        self.save_individual_image(
            self.learned_image,
            base_name,
            trainer,
            tag="adversarial_image",
            caption="Adversarial image",
        )

    def clamp_params(self):
        self.learned_image.clamp_(
            min=self.init_image - self.config.image_update_eps,
            max=self.init_image + self.config.image_update_eps,
        ).clamp_(
            min=0,
            max=1,
        )


class PatchImageProcessor(Processor, ABC):
    init_image: Tensor
    init_patch: Tensor

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.register_buffer(
            "init_image", config.lift_to_model_device_dtype(config.init_image(config))
        )
        self.register_buffer(
            "init_patch", config.lift_to_model_device_dtype(config.init_patch(config))
        )
        # TODO: allow for more flexible patch initialisations
        self.learned_patch = nn.Parameter(data=self.init_patch.clone())

    @abstractmethod
    def get_patch_locations(
        self, batch_size: int
    ) -> Tuple[Int64[Tensor, "b"], Int64[Tensor, "b"]]:
        ...

    def forward(self, image: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        # print("training" if self.training else "eval")
        b, c, h, w = image.shape
        top_left_rows, top_left_cols = self.get_patch_locations(b)

        patched_image = set_patches(
            image,
            repeat(self.learned_patch, "() c h w -> b c h w", b=b),
            top_left_rows,
            top_left_cols,
        )
        return patched_image

    def save_images(self, base_name: str, trainer: LightningModule) -> None:
        # Save / log image with filename `base_name.png`
        self.save_individual_image(
            self.learned_patch,
            f"{base_name}_patch",
            trainer,
            tag="adversarial_image_patch",
            caption="Adversarial image patch",
        )
        self.save_individual_image(
            self(self.init_image),
            f"{base_name}_full",
            trainer,
            tag="adversarial_image_full",
            caption="Adversarial image",
        )

    def clamp_params(self):
        self.learned_patch.clamp_(
            min=self.init_patch - self.config.image_update_eps,
            max=self.init_patch + self.config.image_update_eps,
        ).clamp_(
            min=0,
            max=1,
        )


class StaticPatchImageProcessor(PatchImageProcessor):
    def get_patch_locations(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        row, col = self.config.static_patch_loc(self.config)
        return torch.full((batch_size,), row), torch.full((batch_size,), col)

    @staticmethod
    def center_patch_at(r: Union[float, int], c: Union[float, int], relative: bool):
        def get_center(config: Config):
            init_image = config.init_image(config)
            init_patch = config.init_patch(config)
            _, _, patch_h, patch_w = init_patch.shape
            _, _, h, w = init_image.shape
            if relative:
                centre_r_offset = int(r * h)
                centre_c_offset = int(c * w)
            else:
                centre_r_offset = int(r)
                centre_c_offset = int(c)
            topleft_r_offset = centre_r_offset - (patch_h // 2)
            topleft_c_offset = centre_c_offset - (patch_w // 2)
            return (topleft_r_offset, topleft_c_offset)

        return Factory(get_center)


class RandomPatchImageProcessor(PatchImageProcessor):
    def get_patch_locations(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        _, _, patch_h, patch_w = self.learned_patch.shape
        _, c, h, w = self.init_image.shape
        start_rows = torch.randint(0, h - patch_h + 1, (batch_size,))
        start_cols = torch.randint(0, w - patch_w + 1, (batch_size,))
        return (start_rows, start_cols)


"""
    # === CONFIG ===
    @staticmethod
    def random_patch
"""
