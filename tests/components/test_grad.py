from pathlib import Path
from lightning_fabric import seed_everything

import torch

from image_hijacks.attacks.context import SpecificOutputStringAttack
from image_hijacks.components.grad import (
    ExactGradientEstimator,
    RGFGradientEstimator,
)
from image_hijacks.components.loss import VLMCrossEntropyLoss
from image_hijacks.components.processor import LearnedImageProcessor
from image_hijacks.config import Config
from image_hijacks.models.blip2 import Blip2LensEncDec
from image_hijacks.utils.factory import Factory
from image_hijacks.utils.testing import TestCase
from tests.loaders import load_model, IMG


class TestGrad(TestCase):
    def test_exact_gradient_estimator(self):
        models = {"blip2": load_model("blip2-flan-t5-xl")}
        config = Config(
            data_root=Path("."),
            target_models_train=models,
            target_models_eval=models,
            attack_driver_factory=SpecificOutputStringAttack,
            loss_fn_factory=VLMCrossEntropyLoss,
            seed=1337,
            init_image=Factory(lambda _: torch.zeros((1, 3, 224, 224))),
        )
        model: Blip2LensEncDec = models["blip2"]
        loss_fn = config.loss_fn_factory(config)
        gradient_estimator = config.gradient_estimator_factory(config)
        processor = LearnedImageProcessor(config)

        pixel_values, _ = model.preprocess_image(IMG)
        input_ids, input_attn_masks = model.tokenize(
            "What is in this picture? Give a detailed answer.", "encoder"
        )
        output_ids, output_attn_masks = model.tokenize(
            "the marina bay sands, singapore", "encoder"
        )

        def f(x):
            return (
                loss_fn.get_loss(
                    model, x, input_ids, input_attn_masks, output_ids, output_attn_masks
                ),
                None,
            )

        # Autograd
        processor.requires_grad_(True)
        loss, _ = f(processor(pixel_values))
        loss.backward()
        autograd_grad = processor.learned_image.grad.clone()

        # Our grad
        processor.requires_grad_(False)
        params = processor.get_parameter_dict()

        our_grad, _ = gradient_estimator.grad_and_value(
            lambda params: f(
                torch.func.functional_call(processor, params, (pixel_values,))
            ),
            params,
        )

        self.assertTrue(torch.allclose(autograd_grad, our_grad["learned_image"]))
