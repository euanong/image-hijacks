from pathlib import Path

import torch
from torch.nn import Parameter
from torch.optim import SGD
import torchopt

from image_hijacks.attacks.context import SpecificOutputStringAttack
from image_hijacks.components.grad import ExactGradientEstimator
from image_hijacks.components.loss import VLMCrossEntropyLoss
from image_hijacks.components.optim import TorchOptOptimizer
from image_hijacks.components.processor import LearnedImageProcessor
from image_hijacks.config import Config
from image_hijacks.models.blip2 import Blip2LensEncDec
from image_hijacks.utils.factory import Factory
from image_hijacks.utils.testing import TestCase
from tests.loaders import load_model, IMG


class TestOptim(TestCase):
    def test_sgd_optimiser(self):
        models = {"blip2": load_model("blip2-flan-t5-xl")}
        config = Config(
            data_root=Path("."),
            target_models_train=models,
            target_models_eval=models,
            attack_driver_factory=SpecificOutputStringAttack,
            loss_fn_factory=VLMCrossEntropyLoss,
            attack_optimizer_factory=TorchOptOptimizer,
            torchopt_optimizer=Factory(lambda config: torchopt.sgd(lr=config.lr)),
            seed=1337,
            init_image=Factory(lambda _: torch.zeros((1, 3, 224, 224))),
        )
        config.clip_grad_mag = 0.001
        model: Blip2LensEncDec = config.target_models_train["blip2"]
        loss_fn = config.loss_fn_factory(config)
        gradient_estimator = config.gradient_estimator_factory(config)
        attack_optimizer = config.attack_optimizer_factory(config)

        pixel_values, _ = model.preprocess_image(IMG)
        input_ids, input_attn_masks = model.tokenize(
            "What is in this picture? Give a detailed answer.", "encoder"
        )
        output_ids, output_attn_masks = model.tokenize(
            "the marina bay sands, singapore", "decoder"
        )

        def f(x):
            return loss_fn.get_loss(
                model, x, input_ids, input_attn_masks, output_ids, output_attn_masks
            )

        # Autograd
        auto_processor = LearnedImageProcessor(config).requires_grad_(True)
        optimizer = SGD(auto_processor.parameters(), lr=config.lr)

        for i in range(2):
            optimizer.zero_grad()
            loss = f(auto_processor(pixel_values))
            loss.backward()
            if config.clip_grad_mag is not None:
                torch.nn.utils.clip_grad_norm_(
                    auto_processor.parameters(), max_norm=config.clip_grad_mag
                )
            optimizer.step()
            for parameter in auto_processor.parameters():
                parameter.requires_grad = False
                parameter.clamp_(0, 1)
                parameter.requires_grad = True

        # Our grad
        our_processor = LearnedImageProcessor(config).requires_grad_(False)
        for i in range(2):
            our_grad, _ = gradient_estimator.grad_and_value(
                lambda x: (
                    f(
                        torch.func.functional_call(
                            our_processor,
                            x,
                            pixel_values,
                        )
                    ),
                    None,
                ),
                our_processor.get_parameter_dict(),
            )
            attack_optimizer.step(our_processor, our_grad)

        self.assertTrue(
            torch.allclose(auto_processor(pixel_values), our_processor(pixel_values))
        )
