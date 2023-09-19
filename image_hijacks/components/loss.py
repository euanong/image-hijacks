from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from attr import dataclass

from einops import einsum, repeat
from jaxtyping import Float, Int64, Bool
from torch import Tensor

from typing import TYPE_CHECKING, Any, Literal, NewType, Protocol, Union
from image_hijacks.models import AbstractLensModel

from image_hijacks.models.blip2 import Blip2Lens

if TYPE_CHECKING:
    from image_hijacks.config import Config, Factory

from einops import reduce, rearrange, repeat
import torch.nn.functional as F


class Loss(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_loss(
        self,
        model: AbstractLensModel,
        pixel_values: Float[Tensor, "() c h w"],
        input_ids: Int64[Tensor, "b src_seq_len"],
        input_attn_masks: Float[Tensor, "b src_seq_len"],
        target_ids: Int64[Tensor, "b tgt_seq_len"],
        target_attn_masks: Float[Tensor, "b tgt_seq_len"],
    ) -> Float[Tensor, ""]:
        ...


# VLM cross entropy


class VLMCrossEntropyLoss(Loss):
    def get_loss(
        self,
        model: AbstractLensModel,
        pixel_values: Float[Tensor, "() c h w"],
        input_ids: Int64[Tensor, "b src_seq_len"],
        input_attn_masks: Bool[Tensor, "b src_seq_len"],
        target_ids: Int64[Tensor, "b tgt_seq_len"],
        target_attn_masks: Bool[Tensor, "b tgt_seq_len"],
    ) -> Float[Tensor, ""]:
        logits = model.get_logits_end_to_end(
            repeat(pixel_values, "() c h w -> b c h w", b=input_ids.shape[0]),
            tokens=input_ids,
            token_attention_mask=input_attn_masks,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attn_masks,
        )
        return model.loss(logits, target_ids, padding_tok=model.pad_token_id())


# Black-box textual similarity


class BBTextSimLoss(Loss):
    def get_loss(
        self,
        model,
        pixel_values: Float[Tensor, "() c h w"],
        input_ids: Int64[Tensor, "b src_seq_len"],
        input_attn_masks: Bool[Tensor, "b src_seq_len"],
        target_ids: Int64[Tensor, "b tgt_seq_len"],
        target_attn_masks: Bool[Tensor, "b tgt_seq_len"],
    ) -> Float[Tensor, ""]:
        whitebox_model = self.config.whitebox_model
        assert whitebox_model is not None
        whitebox_model = whitebox_model.to(pixel_values.device)

        # WARNING: some assumptions here (specifically about enc / dec lengths)
        # that may only hold for blip
        assert isinstance(model, Blip2Lens)
        assert isinstance(whitebox_model, Blip2Lens)
        n_target_toks = target_ids.shape[1]

        # Get target embeddings
        target_embeds = whitebox_model.get_token_embeddings(target_ids)

        # Run black-box model
        generated_ids = model.generate_end_to_end(
            repeat(pixel_values, "() c h w -> b c h w", b=input_ids.shape[0]),
            tokens=input_ids,
            token_attention_mask=input_attn_masks,
            max_new_tokens=n_target_toks,
        )
        generated_texts = model.to_string(generated_ids)

        # Use white-box encoder
        wb_toks, wb_attn_mask = whitebox_model.tokenize(
            generated_texts,
            mode="encoder",
            max_length=n_target_toks,
            pad_to_max_length=True,
        )
        wb_embeds = model.get_token_embeddings(wb_toks)

        # mse_loss_per_element: Float[Tensor, "b n h"] = (
        #     rearrange(target_attn_masks, "b n -> b n ()")
        #     * (target_embeds - wb_embeds) ** 2
        # )
        # mse_loss_per_vector = reduce(
        #     mse_loss_per_element, "b n h -> b n", reduction="mean"
        # )
        # return mse_loss_per_vector.sum() / target_attn_masks.sum()

        # Return negated dot product of embeds
        return -1 * reduce(
            einsum(target_embeds, wb_embeds, "b seq_len h, b seq_len h -> b"),
            "b ->",
            reduction="mean",
        )
        # Or since unconstrained, return L2 loss


# Embedding matching


class EmbeddingMatchingLossFn(Protocol):
    def __call__(
        self, pred_h: Float[Tensor, "n h"], target_h: Float[Tensor, "n h"]
    ) -> Float[Tensor, ""]:
        ...


def flattened_l1_loss(
    pred_h: Float[Tensor, "n h"], target_h: Float[Tensor, "n h"]
) -> Float[Tensor, ""]:
    return F.l1_loss(
        rearrange(pred_h, "n h -> (n h)"), rearrange(target_h, "n h -> (n h)")
    )


def flattened_mse_loss(
    pred_h: Float[Tensor, "n h"], target_h: Float[Tensor, "n h"]
) -> Float[Tensor, ""]:
    return F.mse_loss(
        rearrange(pred_h, "n h -> (n h)"), rearrange(target_h, "n h -> (n h)")
    )


def flattened_cosine_similarity_loss(
    pred_h: Float[Tensor, "n h"], target_h: Float[Tensor, "n h"]
) -> Float[Tensor, ""]:
    return 1 - F.cosine_similarity(
        rearrange(pred_h, "n h -> (n h)"),
        rearrange(target_h, "n h -> (n h)"),
        dim=0,
    )


def avg_cosine_similarity_per_vector_loss(
    pred_h: Float[Tensor, "n h"], target_h: Float[Tensor, "n h"]
) -> Float[Tensor, ""]:
    return 1 - reduce(
        F.cosine_similarity(
            pred_h,
            target_h,
            dim=1,
        ),
        "b ->",
        reduction="mean",
    )


@dataclass
class Image:
    data: Float[Tensor, "() c h w"]


@dataclass
class Text:
    data: str


@dataclass
class Tokens:
    data: Float[Tensor, "seq_len"]


EmbeddingMatchingTarget = Union[Image, Text, Tokens]


class EmbeddingMatchingLoss(Loss):
    """Match image embeddings with
    - either the input embedding
    - or a different target image"""

    def __init__(self, config: Config):
        self.config = config
        self.target = self.config.embedding_matching_target(self.config)

    def get_target_embeds(
        self, model: AbstractLensModel, max_length: int, device
    ) -> Float[Tensor, "() img_toks h"]:
        target = self.target
        if isinstance(target, Image):
            return model.get_image_embeddings(target.data)
        elif isinstance(target, Text):
            tokens, attn_mask = model.tokenize(
                target.data,
                "no_special_tokens",
                max_length=max_length,
                pad_to_max_length=self.config.embedding_matching_pad_target_seq,
            )
            return model.get_token_embeddings(tokens.to(device))
        elif isinstance(target, Tokens):
            return model.get_token_embeddings(rearrange(target.data, "n -> () n"))
        else:
            assert False

    def get_loss(
        self,
        model: AbstractLensModel,
        pixel_values: Float[Tensor, "() c h w"],
        input_ids: Int64[Tensor, "b src_seq_len"],
        input_attn_masks: Float[Tensor, "b src_seq_len"],
        target_ids: Int64[Tensor, "b tgt_seq_len"],
        target_attn_masks: Float[Tensor, "b tgt_seq_len"],
    ) -> Float[Tensor, ""]:
        img_embeds: Float[Tensor, "() img_toks h"] = model.get_image_embeddings(
            pixel_values
        )

        n_img_toks = img_embeds.shape[1]
        target_embeds = self.get_target_embeds(
            model, max_length=n_img_toks, device=pixel_values.device
        )

        b, n_target_toks, h = target_embeds.shape
        assert b == 1

        if n_target_toks > n_img_toks:
            raise ValueError("Target token length must be less than # image tokens.")

        return self.config.embedding_matching_loss_fn(
            img_embeds[0, :n_target_toks], target_embeds[0]
        )
