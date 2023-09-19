from pathlib import Path
import traceback
from typing import Callable, List, Literal, Optional, Tuple, Type, TypeVar, Union
from einops import rearrange, repeat
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor
import torch
import torch.nn.functional as F
from tqdm import tqdm

from image_hijacks.models import AbstractLensModel
from PIL import Image

from jaxtyping import Bool, Float, Int64
from llava.model import LlavaLlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from torch.nn import CrossEntropyLoss

from llava.model import LlavaLlamaForCausalLM
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from transformers import AutoTokenizer, AutoConfig
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from image_hijacks.utils import remove_no_grad

from image_hijacks.utils import PROJECT_ROOT, detach_numpy, load_model_with_cache

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

LlavaLensT = TypeVar("LlavaLensT", bound="LlavaLens")

from abc import abstractmethod


# TODO this should wrap Llava conditional generation
class LlavaLens(AbstractLensModel):
    IMAGE_TOKEN = "<image_tkn>"
    # index with which we have to replace tokenised version of <image_tkn> before feeding
    # to LLaVA

    def __init__(
        self,
        model: LlavaLlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        image_processor: CLIPImageProcessor,
        model_dtype: torch.dtype,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_dtype = model_dtype

    def input_image_dims(self) -> Tuple[int, int]:
        size_dict = self.image_processor.crop_size
        return (size_dict["height"], size_dict["width"])

    def preprocess_image(
        self, img: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Float[Tensor, "b c h w"], Bool[Tensor, "b img_seq_len"]]:
        """Converts PIL image to unnormalised tensor with pixel values in [0, 1]"""
        preprocessed_img = self.image_processor.preprocess(
            img,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=False,
        )["pixel_values"].to(self.model_dtype)
        # TODO: return attention mask
        return (preprocessed_img, None)

    def normalize_image(
        self, pixel_values: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        """Normalise batch of images"""
        # TODO: write allclose test
        mean = rearrange(
            torch.tensor(
                self.image_processor.image_mean,
                device=pixel_values.device,
                dtype=self.model_dtype,
            ),
            "c -> () c () ()",
        )
        std = rearrange(
            torch.tensor(
                self.image_processor.image_std,
                device=pixel_values.device,
                dtype=self.model_dtype,
            ),
            "c -> () c () ()",
        )
        return (pixel_values - mean) / std

    def tokeniser_ids_to_llava_input(
        self, input_ids: Int64[Union[Tensor, ndarray], "b max_seq_len"]
    ) -> None:
        # WARNING: in-place update
        image_token_id = self.tokenizer.convert_tokens_to_ids(LlavaLens.IMAGE_TOKEN)
        input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX

    def llava_input_to_tokeniser_ids(
        self, input_ids: Int64[Union[Tensor, ndarray], "b max_seq_len"]
    ) -> None:
        # WARNING: in-place update
        image_token_id = self.tokenizer.convert_tokens_to_ids(LlavaLens.IMAGE_TOKEN)
        input_ids[input_ids == IMAGE_TOKEN_INDEX] = image_token_id

    def wrap_with_system_prompt(self, random: bool = False) -> Callable[[str], str]:
        prompts = [
            lambda x: (
                f"A chat between a curious user and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: {LlavaLens.IMAGE_TOKEN}\n{x} ASSISTANT:"
            ),
            lambda x: (
                f"A chat between a curious user and an artificial intelligence assistant. "
                f"The assistant is able to understand the visual content that the user provides, "
                f"and assist the user with a variety of tasks using natural language.The visual "
                f"content will be provided with the following format: <Image>visual content</Image>. "
                f"USER: <Image>{LlavaLens.IMAGE_TOKEN}</Image> ASSISTANT: Received.</s>"
                f"USER: {x} ASSISTANT:"
            ),
            lambda x: (
                f"A chat between a curious human and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the human's questions."
                f"###Human: Hi!###Assistant: Hi there! How can I help you today?"
                f"###Human: {x}\n{LlavaLens.IMAGE_TOKEN}###Assistant:"
            ),
            lambda x: (  # LLaVA-2 prompt
                f"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand "
                f"the visual content that the user provides, and assist the user with a variety of tasks using "
                f"natural language.\n<</SYS>>\n\n{LlavaLens.IMAGE_TOKEN}\n{x}[/INST]"
            ),
        ]
        if random:
            return prompts[int(torch.randint(len(prompts), tuple()))]
        else:
            assert False

    def _tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX):
        # from llava.mm_utils
        prompt_chunks = [
            self.tokenizer(chunk).input_ids
            for chunk in prompt.split(LlavaLens.IMAGE_TOKEN)
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == self.tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)

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

        - Encoder:
            [BOS] {### Human: <img_tkn> } 0 1 2 { ###Assistant:}
              [BOS] {### Human: <img_tkn> } 0 1 { ###Assistant:}
        - Decoder:
            [BOS] 0 1 2
            [BOS] 0 1
        [As before, we strip the BOS token when actually decoding]
        BOS_ID = model.config.bos_token_id
        """
        # TODO WARNING: We aren't truncating.
        if mode == "encoder":
            assert pad_to_max_length == False

            if isinstance(text, str):
                text = [text]

            inputs: List[Int64[Tensor, "seq_len"]] = []
            for t in tqdm(text) if len(text) > 10000 else text:
                inputs.append(
                    self._tokenizer_image_token(
                        self.wrap_with_system_prompt(
                            random=randomly_sample_system_prompt
                        )(t[:max_length])
                    )
                )
            max_len = max(i.shape[-1] for i in inputs)
            padded_input_ids = [
                F.pad(i, (max_len - i.shape[-1], 0), value=self.pad_token_id())
                for i in inputs
            ]
            padded_attn_masks = [
                F.pad(
                    torch.ones_like(i),
                    (max_len - i.shape[-1], 0),
                    value=self.pad_token_id(),
                )
                for i in inputs
            ]
            input_ids: Int64[Tensor, "b seq_len"] = torch.stack(padded_input_ids)
            attn_mask: Int64[Tensor, "b seq_len"] = torch.stack(padded_attn_masks)
            self.tokeniser_ids_to_llava_input(input_ids)
        elif mode == "decoder":
            BOS_ID = self.model.config.bos_token_id
            assert BOS_ID is not None

            if isinstance(text, str):
                text = [text]
            text = [i + "</s>" for i in text]

            results = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
            )
            input_ids: Int64[Tensor, "b seq_len"] = results["input_ids"]  # type: ignore
            attn_mask: Int64[Tensor, "b seq_len"] = results["attention_mask"]  # type: ignore

            b, _ = input_ids.shape
            device = input_ids.device
            input_ids = torch.cat(
                [torch.full((b, 1), BOS_ID, device=device), input_ids], dim=1
            )
            attn_mask = torch.cat(
                [torch.full((b, 1), 1, device=device, dtype=torch.bool), attn_mask],
                dim=1,
            )
        elif mode == "no_special_tokens":
            results = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
            )
            input_ids: Int64[Tensor, "b seq_len"] = results["input_ids"]  # type: ignore
            attn_mask: Int64[Tensor, "b seq_len"] = results["attention_mask"]  # type: ignore
        else:
            assert False
        return input_ids.to(self.device), attn_mask.to(self.device)

    def to_string(
        self, tokens: Int64[Tensor, "b seq_len"], skip_special_tokens=True
    ) -> List[str]:
        """Given a batch of sequences of tokens, detokenise each sequence.
        - If skip_special_tokens set, we omit pad / BOS / EOS tokens."""
        np_tokens = detach_numpy(tokens)
        self.llava_input_to_tokeniser_ids(np_tokens)
        return self.tokenizer.batch_decode(
            sequences=np_tokens,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def get_logits_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Int64[Tensor, "b src_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b tgt_seq_len n_tokens"]:
        assert image_attention_mask is None
        # TODO: once we've implemented embeddings, remove this impl and rely on
        # superclass impl
        # NOTE: We strip BOS token for decoder
        input_ids = torch.cat([tokens, decoder_input_ids[:, 1:]], dim=1)
        attention_mask = torch.cat(
            [token_attention_mask, decoder_attention_mask[:, 1:]], dim=1
        )

        padding_idxs_left = 1 - torch.cumsum(
            input_ids == self.tokenizer.convert_tokens_to_ids("<s>"), dim=1
        )
        shift_left_to_right = padding_idxs_left.argsort(dim=1, stable=True)

        right_shift_toks = torch.gather(input_ids, 1, shift_left_to_right)
        right_shift_attn_masks = torch.gather(attention_mask, 1, shift_left_to_right)
        padding_idxs_right = torch.gather(padding_idxs_left, 1, shift_left_to_right)

        if pixel_values.shape[0] == 1:
            pixel_values = repeat(
                pixel_values, "() c h w -> b c h w", b=right_shift_toks.shape[0]
            )
        logits = self.model.forward(
            images=self.normalize_image(pixel_values),
            input_ids=right_shift_toks,
            attention_mask=right_shift_attn_masks,
        ).logits

        result_padding_idxs_right = F.pad(
            padding_idxs_right,
            (logits.shape[1] - padding_idxs_right.shape[1], 0, 0, 0),
            value=0,
        )
        shift_right_to_left = result_padding_idxs_right.argsort(
            dim=1, stable=True, descending=True
        )
        left_shift_logits = torch.gather(
            logits, 1, repeat(shift_right_to_left, "b n -> b n h", h=logits.shape[-1])
        )
        return left_shift_logits[:, -(decoder_input_ids.shape[1]) :]

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
        assert image_attention_mask is None
        assert self.model.config.bos_token_id is not None
        if pixel_values.shape[0] == 1:
            pixel_values = repeat(
                pixel_values, "() c h w -> b c h w", b=tokens.shape[0]
            )
        model_ids = []
        normalized_pixels = self.normalize_image(pixel_values)

        n_padding_idxs_left = (1 - torch.cumsum(tokens == 1, dim=1)).sum(dim=1)
        # TODO: fix hack... batch inference is broken
        for i in range(tokens.shape[0]):
            model_ids.append(
                self.model.generate(  # type: ignore
                    images=normalized_pixels[i : i + 1],
                    input_ids=tokens[i : i + 1, n_padding_idxs_left[i] :],
                    attention_mask=token_attention_mask[
                        i : i + 1, n_padding_idxs_left[i] :
                    ],
                    max_new_tokens=max_new_tokens - 1,
                )[:, (tokens.shape[1] - n_padding_idxs_left[i]) :]
            )
        gen_ids = torch.nn.utils.rnn.pad_sequence(
            [rearrange(x, "() x -> x") for x in model_ids],
            batch_first=True,
            padding_value=self.pad_token_id(),
        )
        return torch.cat(
            [
                torch.full(
                    (gen_ids.shape[0], 1),
                    self.model.config.bos_token_id,
                    device=gen_ids.device,
                ),
                gen_ids,
            ],
            dim=1,
        )

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
        # default: loss computed with padding -100
        if padding_tok is None:
            padding_tok = self.pad_token_id()

        labels = label_toks.to(logits.device)
        logits = logits[:, -labels.size(1) :, :]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=padding_tok)
        loss = loss_fct(
            shift_logits.view(-1, self.model.config.vocab_size),
            shift_labels.view(-1),
        )
        return loss

    # === Computing loss ===
    def pad_token_id(self) -> int:
        # return self.tokenizer.convert_tokens_to_ids("<pad>")
        return self.tokenizer.convert_tokens_to_ids("<unk>")

    # === Loading model ===

    @classmethod
    def load_model_from_path(
        cls: "Type[LlavaLensT]",
        model_path: Path,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "LlavaLensT":
        """Load model and processor.
        Args:
            model_dtype -- Datatype used for loaded model.
            requires_grad -- Whether to compute gradients for model params
        """
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model: LlavaLlamaForCausalLM = load_model_with_cache(  # type: ignore
            model_fn=lambda: LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                # low_cpu_mem_usage=True,
                config=cfg_pretrained,
                torch_dtype=model_dtype,
            ).eval(),
            model_id_components=(Path(model_path), model_dtype),
        )

        tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)  # type: ignore
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [LlavaLens.IMAGE_TOKEN]
            }  # , "pad_token": "<pad>"}
        )
        model.resize_token_embeddings(len(tokenizer))
        # model.model.padding_idx = tokenizer.convert_tokens_to_ids("<pad>")

        # Load vision tower
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower = vision_tower.to(dtype=model_dtype)

        # Extract image processor
        image_processor = vision_tower.image_processor
        image_processor.do_normalize = False

        model.requires_grad_(requires_grad)

        CLIPVisionTower.forward = remove_no_grad(CLIPVisionTower.forward)
        return cls(model, tokenizer, image_processor, model_dtype)

    # ====== TODO (needed for more advanced attacks) ======

    # === Embeddings ===

    def get_image_embeddings(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Optional[Float[Tensor, "b tok_seq_len h_lm"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Float[Tensor, "b img_seq_len h_lm"]:
        """Given a batch of unnormalised input images (along input embeddings), return a batch of sequences of image embeddings."""
        raise NotImplementedError

    def get_token_embeddings(
        self, tokens: Int64[Tensor, "b max_seq_len"]
    ) -> Float[Tensor, "b max_seq_len h_lm"]:
        """Given a batch of padded tokens, returns language model embeddings."""
        raise NotImplementedError

    def get_embeddings_from_image_and_tokens(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b seq_len h_lm"], Int64[Tensor, "b seq_len"]]:
        """Given pixel values and input tokens, returns input embeddings and attention
        mask. If attention masks not provided, we assume all 1s."""
        raise NotImplementedError

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
        raise NotImplementedError

    # === Generation ===

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
        raise NotImplementedError


class LlavaLlama2_13b(LlavaLens):
    def wrap_with_system_prompt(self, random: bool = False) -> Callable[[str], str]:
        if random:
            return super().wrap_with_system_prompt(random)
        else:
            return lambda text: (  # LLaVA-2 prompt
                f"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand "
                f"the visual content that the user provides, and assist the user with a variety of tasks using "
                f"natural language.\n<</SYS>>\n\n{LlavaLens.IMAGE_TOKEN}\n{text} [/INST]"
            )

    @classmethod
    def load_model(
        cls,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "LlavaLlama2_13b":
        return cls.load_model_from_path(
            PROJECT_ROOT / "downloads/model_checkpoints/llava-llama-2-13b-chat",
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class LlavaLlama2_7b(LlavaLens):
    @classmethod
    def load_model(
        cls,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "LlavaLlama2_7b":
        return cls.load_model_from_path(
            PROJECT_ROOT / "downloads/model_checkpoints/llava-llama-2-7b-chat",
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class LlavaLlama1_13b(LlavaLens):
    def wrap_with_system_prompt(self, random: bool = False) -> Callable[[str], str]:
        if random:
            return super().wrap_with_system_prompt(random)
        else:
            print("WARNING: Original LLaVA-v1 prompt unknown")
            return lambda x: (
                f"A chat between a curious user and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: {LlavaLens.IMAGE_TOKEN}\n{x} ASSISTANT:"
            )  # TODO: uncertain if this is right prompt

    @classmethod
    def load_model(
        cls,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> "LlavaLlama1_13b":
        return cls.load_model_from_path(
            PROJECT_ROOT / "downloads/model_checkpoints/llava-v1.3-13b-336px",
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )
