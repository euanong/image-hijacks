from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, List, TypeVar, Type
import torch
from PIL import Image

from typing import Union

from jaxtyping import Float, Int64, Bool
from torch import Tensor, nn, dtype

AbstractLensModelT = TypeVar("AbstractLensModelT", bound="AbstractLensModel")


class AbstractLensModel(ABC, nn.Module):
    """Abstract base class for all VLM used in this project"""

    model_dtype: dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode: bool):
        """avoid pytorch lightning auto set train mode"""
        # TODO: get rid of this hack
        return super().train(False)

    # === Pre/post-processing ===
    @abstractmethod
    def input_image_dims(self) -> Tuple[int, int]:
        """Returns (h, w) of input image"""
        ...

    @abstractmethod
    def preprocess_image(
        self, img: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Float[Tensor, "b c h w"], Bool[Tensor, "b img_seq_len"]]:
        """Converts PIL image to unnormalised tensor with pixel values in [0, 1]"""
        ...

    @abstractmethod
    def normalize_image(
        self, pixel_values: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        """Normalise batch of images"""
        ...

    @abstractmethod
    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: Literal["encoder", "decoder", "no_special_tokens"],
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        randomly_sample_system_prompt: bool = False,
    ) -> Tuple[Int64[Tensor, "b max_seq_len"], Bool[Tensor, "b max_seq_len"]]:
        """Given text or a list of text, returns batched tokenised text along with a padding mask
        (mask is 1 if the token is non-padding).
        - The returned batch of text has token length (ignoring special chars)
            min(max_length, max(len(i) for i in text))
            with strings padded / truncated as necessary to achieve this.
        - If pad_to_max_length is set, we will pad / truncate our batch of text such that it has
            token length max_length.
        - Behaviour if pad_to_max_length is True when max_length is None is undefined.
        """
        ...

    @abstractmethod
    def to_string(
        self, tokens: Int64[Tensor, "b seq_len"], skip_special_tokens=True
    ) -> List[str]:
        """Given a batch of sequences of tokens, detokenise each sequence.
        - If skip_special_tokens set, we omit pad / BOS / EOS tokens."""
        ...

    # === Embeddings ===

    @abstractmethod
    def get_image_embeddings(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Optional[Float[Tensor, "b tok_seq_len h_lm"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Float[Tensor, "b img_seq_len h_lm"]:
        """Given a batch of unnormalised input images (along input embeddings), return a batch of sequences of image embeddings."""
        ...

    @abstractmethod
    def get_token_embeddings(
        self, tokens: Int64[Tensor, "b max_seq_len"]
    ) -> Float[Tensor, "b max_seq_len h_lm"]:
        """Given a batch of padded tokens, returns language model embeddings."""

    @abstractmethod
    def get_embeddings_from_image_and_tokens(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b seq_len h_lm"], Int64[Tensor, "b seq_len"]]:
        """Given pixel values and input tokens, returns input embeddings and attention
        mask. If attention masks not provided, we assume all 1s."""
        ...

    @abstractmethod
    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        """Given input embeddings (and optionally decoder input IDs), return per-position logits.
        - If decoder input IDs not provided, [BOS] passed to decoder.
        - Attention mask 0 if token should be ignored (i.e. padding).
        - If attention mask not provided, we use all 1s (i.e. no padding)

        decoder_input_ids: BS T0 T1 T2
        return logits:     T0 T1 T2 T3
        """
        ...

    def get_logits_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Int64[Tensor, "b src_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b tgt_seq_len n_tokens"]:
        """Given input tokens and pixel values (and optionally decoder input IDs), return per-position logits.
        - If decoder input IDs not provided, [BOS] passed to decoder.
        - Attention mask 0 if token should be ignored (i.e. padding).
        - If attention mask not provided, we use all 1s (i.e. no padding)

        decoder_input_ids: BS T0 T1 T2
        return logits:     T0 T1 T2 T3
        """
        embs, attn_mask = self.get_embeddings_from_image_and_tokens(
            pixel_values, tokens, image_attention_mask, token_attention_mask
        )
        return self.get_logits_from_embeddings(
            embs,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    # === Generation ===
    @abstractmethod
    def generate_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b tok_seq_len n_tokens"]:
        """Given input tokens and pixel values, return generated output tokens.
        - Attention mask 0 if token should be ignored (i.e. padding).
        - If attention mask not provided, we use all 1s (i.e. no padding)
        n.b. max_new_tokens includes BOS / EOS tokens

        WARNING: for dec-only models, prepends an extra BOS"""
        ...

    @abstractmethod
    def generate_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        """Given input embeddings, return generated output tokens.
        - Attention mask 0 if token should be ignored (i.e. padding).
        - If attention mask not provided, we use all 1s (i.e. no padding)

        WARNING: for dec-only models, prepends an extra BOS"""
        ...

    # === Computing loss ===

    @abstractmethod
    def pad_token_id(self) -> int:
        ...

    @abstractmethod
    def loss(
        self,
        logits: Float[Tensor, "b seq_len n_toks"],
        label_toks: Int64[Tensor, "b seq_len"],
        padding_tok: Optional[int] = None,
    ) -> Float[Tensor, ""]:
        """Returns masked language modelling loss computed between logits[:-1] and label_toks[1:]

        Expected input:
        - logits:     L0 L1 L2 L3
        - label_toks: BS L0 L1 L2

        Note: Indices should either be in [0, ..., config.vocab_size] or [padding_tok].
        Tokens with indices set to [padding_tok] are ignored (masked); the loss is only
        computed for the tokens with labels in [0, ..., config.vocab_size].
        """
        # TODO: pass in attention masks instead of padding tok...
        ...

    # === Loading model ===

    @classmethod
    @abstractmethod
    def load_model(
        cls: Type[AbstractLensModelT],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> AbstractLensModelT:
        """Load model and processor.
        Args:
            model_dtype -- Datatype used for loaded model.
            requires_grad -- Whether to compute gradients for model params
        """
        ...
