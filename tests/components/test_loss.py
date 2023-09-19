from pathlib import Path
from lightning_fabric import seed_everything

import torch

from image_hijacks.attacks.context import SpecificOutputStringAttack
from image_hijacks.components.loss import (
    EmbeddingMatchingLoss,
    Image,
    Text,
    Tokens,
    VLMCrossEntropyLoss,
    avg_cosine_similarity_per_vector_loss,
)
from image_hijacks.config import Config
from image_hijacks.models.blip2 import Blip2LensEncDec
from image_hijacks.utils.factory import Factory
from image_hijacks.utils.testing import TestCase
from tests.loaders import load_model, IMG


class TestLoss(TestCase):
    def test_vlm_cross_entropy_loss(self):
        models = {"blip2": load_model("blip2-flan-t5-xl")}
        config = Config(
            data_root=Path("."),
            target_models_train=models,
            target_models_eval=models,
            attack_driver_factory=SpecificOutputStringAttack,
            loss_fn_factory=VLMCrossEntropyLoss,
            seed=1337,
        )
        model: Blip2LensEncDec = models["blip2"]  # type: ignore
        loss_fn = config.loss_fn_factory(config)

        pixel_values, _ = model.preprocess_image(IMG)
        input_ids, input_attn_masks = model.tokenize(
            "What is in this picture? Give a detailed answer.", "encoder"
        )
        output_ids, output_attn_masks = model.tokenize(
            "the marina bay sands, singapore", "decoder"
        )

        # Our loss
        loss = loss_fn.get_loss(
            model,
            pixel_values,
            input_ids,
            input_attn_masks,
            output_ids,
            output_attn_masks,
        )
        self.assertExpectedPretty(loss.to("cpu"), """tensor(0.467)""")

        # Model loss
        inputs = model.processor(
            images=IMG,
            text="What is in this picture? Give a detailed answer.",
            return_tensors="pt",
        )
        outputs = model.model(
            inputs["pixel_values"].to(loss.device),
            inputs["input_ids"].to(loss.device),
            attention_mask=inputs["attention_mask"].to(loss.device),
            labels=output_ids[:, 1:],  # omit the pad token
        )
        self.assertExpectedPretty(outputs.loss.to("cpu"), """tensor(0.467)""")

        assert torch.allclose(loss.to("cpu"), outputs.loss.to("cpu"))

    def test_embedding_matching_loss_identical_imgs(self):
        models = {"blip2": load_model("blip2-flan-t5-xl")}
        config = Config(
            data_root=Path("."),
            target_models_train=models,
            target_models_eval=models,
            loss_fn_factory=EmbeddingMatchingLoss,
            embedding_matching_target=Factory(
                lambda c: Image(c.target_models_train["blip2"].preprocess_image(IMG)[0])
            ),
            embedding_matching_loss_fn=avg_cosine_similarity_per_vector_loss,
            embedding_matching_pad_target_seq=False,
            seed=1337,
        )
        model: Blip2LensEncDec = config.target_models_train["blip2"]
        loss_fn = config.loss_fn_factory(config)

        pixel_values, _ = model.preprocess_image(IMG)

        # These shouldn't matter.
        input_ids, input_attn_masks = model.tokenize(
            "What is in this picture? Give a detailed answer.", "encoder"
        )
        output_ids, output_attn_masks = model.tokenize(
            "the marina bay sands, singapore", "decoder"
        )

        # Loss with identical image should be 0.
        loss = (
            loss_fn.get_loss(
                model,
                pixel_values,
                input_ids,
                input_attn_masks,
                output_ids,
                output_attn_masks,
            )
            .detach()
            .cpu()
            .item()
        )
        self.assertAlmostEqual(loss, 0.0)
