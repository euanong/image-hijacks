import hashlib
from abc import ABC
from pathlib import Path
from typing import Generic, Literal, Union, List, Tuple, Type, Optional, TypeVar, cast

import torch
from PIL import Image
from einops import pack, rearrange, repeat
from jaxtyping import Float16, Int64, Float, Bool
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import (
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    BlipImageProcessor,
    PreTrainedTokenizer,
    Blip2Processor,
    InstructBlipProcessor,
    Blip2Config,
    InstructBlipConfig,
    modeling_outputs as mo,
)

from image_hijacks.models import AbstractLensModel
from image_hijacks.utils import (
    PROJECT_ROOT,
    detach_numpy,
    get_full_attention_mask,
    load_model_with_cache,
)

Blip2T = TypeVar("Blip2T", bound="Blip2Lens")

BlipForConditionalGenerationT = TypeVar(
    "BlipForConditionalGenerationT",
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
)
BlipProcessorT = TypeVar(
    "BlipProcessorT",
    Blip2Processor,
    InstructBlipProcessor,
)


class Blip2Lens(
    AbstractLensModel, Generic[BlipForConditionalGenerationT, BlipProcessorT], ABC
):
    """Class that adds additional functionality for experimenting with BLIP-2"""

    def __init__(
        self,
        model: BlipForConditionalGenerationT,
        processor: BlipProcessorT,
        dtype: torch.dtype,
    ):
        super().__init__()
        if type(model) == Blip2ForConditionalGeneration:
            assert type(processor) == Blip2Processor
            self.is_instructblip = False
        elif type(model) == InstructBlipForConditionalGeneration:
            assert type(processor) == InstructBlipProcessor
            self.is_instructblip = True
        else:
            assert False
        self.config = cast(Blip2Config, model.config)
        self.model: BlipForConditionalGenerationT = model
        self.processor: BlipProcessorT = processor
        self.image_processor: BlipImageProcessor = processor.image_processor  # type: ignore
        self.tokenizer: PreTrainedTokenizer = processor.tokenizer  # type: ignore
        self.model_dtype: torch.dtype = dtype
        # self.model_dtype: torch.dtype = model.config.torch_dtype  # type: ignore

    def input_image_dims(self) -> Tuple[int, int]:
        return (
            self.config.vision_config.image_size,
            self.config.vision_config.image_size,
        )

    def preprocess_image(
        self, img: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Float16[Tensor, "b c h w"], Float16[Tensor, "b img_seq_len"]]:
        preprocessed_img = (
            self.image_processor.preprocess(
                img,
                return_tensors="pt",
                do_resize=True,
                do_rescale=True,
                do_normalize=False,
            )["pixel_values"]
            .type(self.model_dtype)
            .to(self.device)
        )
        return_attn_mask = torch.ones(
            (preprocessed_img.shape[0], self.config.num_query_tokens),
            dtype=torch.bool,
            device=self.device,
        )
        return preprocessed_img, return_attn_mask

    def normalize_image(
        self, pixel_values: Float16[Tensor, "b c h w"]
    ) -> Float16[Tensor, "b c h w"]:
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
        pixel_values = (pixel_values - mean) / std
        return pixel_values

    def get_image_embeddings(
        self,
        pixel_values: Float16[Tensor, "b c h w"],
        tokens: Optional[Float[Tensor, "b tok_seq_len h_lm"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Float16[Tensor, "b img_seq_len h_lm"]:
        # step 0: normalise images
        pixel_values = self.normalize_image(pixel_values)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        image_embeds: Float16[Tensor, "b n_patches h_img"] = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask: Bool[Tensor, "b n_patches"] = torch.ones(
            image_embeds.size()[:-1], dtype=torch.bool, device=image_embeds.device
        )

        query_tokens: Float16[Tensor, "b n_qry_toks h_qry"] = repeat(
            self.model.query_tokens,
            "() n_qry_toks h_qry -> b n_qry_toks h_qry",
            b=image_embeds.shape[0],
        )
        if self.is_instructblip:
            assert tokens is not None
            qformer_inputs = self.processor.qformer_tokenizer(
                self.to_string(tokens, skip_special_tokens=True),
                return_tensors="pt",
                padding=True,
            )
            qformer_input_ids = qformer_inputs["input_ids"].to(image_embeds.device)
            qformer_attention_mask = qformer_inputs["attention_mask"].to(
                image_embeds.device
            )
            query_attention_mask = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            attention_mask = torch.cat(
                [query_attention_mask, qformer_attention_mask], dim=1
            )
            query_outputs: mo.BaseModelOutputWithPoolingAndCrossAttentions = (
                self.model.qformer(
                    input_ids=qformer_input_ids,
                    attention_mask=attention_mask,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            )
            query_output: Float16[
                Tensor, "b n_qry_toks h_qry"
            ] = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        else:
            query_outputs: mo.BaseModelOutputWithPoolingAndCrossAttentions = (
                self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            )
            query_output: Float16[
                Tensor, "b n_qry_toks h_qry"
            ] = query_outputs.last_hidden_state

        # step 3: project into language model embedding space
        return_embeds: Float16[
            Tensor, "b n_qry_toks h_lm"
        ] = self.model.language_projection(query_output)
        return return_embeds

    def get_token_embeddings(
        self, tokens: Int64[Tensor, "b max_seq_len"]
    ) -> Float16[Tensor, "b max_seq_len h_lm"]:
        return self.model.language_model.get_input_embeddings()(tokens).to(
            self.model_dtype
        )

    def get_embeddings_from_image_and_tokens(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "b seq_len h_lm"], Int64[Tensor, "b seq_len"]]:
        """Given pixel values and input tokens, returns embeddings and attention
        mask. If attention masks not provided, we assume all 1s."""
        image_embeds = self.get_image_embeddings(
            pixel_values, tokens, token_attention_mask
        )
        token_embeds = self.get_token_embeddings(tokens)
        image_attention_mask = (
            image_attention_mask
            if image_attention_mask is not None
            else get_full_attention_mask(image_embeds)
        )
        token_attention_mask = (
            token_attention_mask
            if token_attention_mask is not None
            else get_full_attention_mask(token_embeds)
        )
        input_embeds, _ = pack([image_embeds, token_embeds], "b * h_lm")
        attention_mask, _ = pack([image_attention_mask, token_attention_mask], "b *")
        return input_embeds, attention_mask

    def generate_end_to_end(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Int64[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        assert image_attention_mask is None
        if pixel_values.shape[0] == 1:
            pixel_values = repeat(
                pixel_values, "() c h w -> b c h w", b=tokens.shape[0]
            )
        if self.is_instructblip:
            qformer_inputs = self.processor.qformer_tokenizer(
                self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True),
                return_tensors="pt",
                padding=True,
            )
            qformer_input_ids = qformer_inputs["input_ids"].to(tokens.device)
            qformer_attention_mask = qformer_inputs["attention_mask"].to(tokens.device)

            # TODO: fix hack... instructblip generate is broken.
            model_ids = []
            normalized_pixels = self.normalize_image(pixel_values)
            for i in range(tokens.shape[0]):
                model_ids.append(
                    self.model.generate(
                        pixel_values=normalized_pixels[i : i + 1],
                        input_ids=tokens[i : i + 1],
                        attention_mask=token_attention_mask[i : i + 1],
                        max_length=max_new_tokens,
                        qformer_input_ids=qformer_input_ids[i : i + 1],
                        qformer_attention_mask=qformer_attention_mask[i : i + 1],
                    )
                )
            return torch.nn.utils.rnn.pad_sequence(
                [rearrange(x, "() x -> x") for x in model_ids],
                batch_first=True,
                padding_value=self.pad_token_id(),
            )
        else:
            return self.model.generate(
                pixel_values=self.normalize_image(pixel_values),
                input_ids=tokens,
                attention_mask=token_attention_mask,
                max_length=max_new_tokens,
            )

    @torch.no_grad()
    def generate_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        return self.model.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_length=max_new_tokens,
        )

    def pad_token_id(self) -> int:
        return self.processor.tokenizer.pad_token_id  # type: ignore

    def loss(
        self,
        logits: Float[Tensor, "b seq_len n_toks"],
        label_toks: Int64[Tensor, "b seq_len"],
        padding_tok: Optional[int] = None,
    ) -> Float[Tensor, ""]:
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
            shift_logits.view(-1, self.config.text_config.vocab_size),
            shift_labels.view(-1),
        )
        return loss

    def to_string(
        self, tokens: Int64[Tensor, "b seq_len"], skip_special_tokens=True
    ) -> List[str]:
        np_tokens = detach_numpy(tokens)
        # https://github.com/huggingface/transformers/blob/0fd8d2aa2cc9e172a8af9af8508b2530f55ca14c/src/transformers/models/instructblip/modeling_instructblip.py#L1561C9-L1562C38
        if (
            self.is_instructblip
            and self.model.config.text_config.architectures[0] == "LLaMAForCausalLM"
        ):
            np_tokens[np_tokens == 0] = 2
        return self.tokenizer.batch_decode(
            sequences=np_tokens,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    @classmethod
    def load_model_and_processor(
        cls: Type[Blip2T],
        model_path: Union[Path, str],
        is_instructblip: bool,
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ) -> Blip2T:
        model_cls = (
            InstructBlipForConditionalGeneration
            if is_instructblip
            else Blip2ForConditionalGeneration
        )
        processor_cls = InstructBlipProcessor if is_instructblip else Blip2Processor
        model = load_model_with_cache(
            model_fn=lambda: model_cls.from_pretrained(
                model_path, torch_dtype=model_dtype
            ).eval(),  # type: ignore
            model_id_components=(Path(model_path), model_dtype),
        )
        processor = processor_cls.from_pretrained(model_path)
        if not requires_grad:
            model.requires_grad_(False)
        return cls(model, processor, model_dtype)


class Blip2LensEncDec(
    Blip2Lens[BlipForConditionalGenerationT, BlipProcessorT],
    Generic[BlipForConditionalGenerationT, BlipProcessorT],
):
    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: Literal["encoder", "decoder", "no_special_tokens"],
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
    ) -> Tuple[Int64[Tensor, "b max_seq_len"], Float16[Tensor, "b max_seq_len"]]:
        """
        - Encoder:
            0 1 2 [EOS]
            0 1 [EOS]
        EOS_ID = self.config.text_config.bos_token_id
        - Decoder:
            [BOS] 0 1 2
            [BOS] 0 1
        BOS_ID = self.config.text_config.decoder_start_token_id
        """
        if mode == "encoder":
            results = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
            )
            input_ids: Int64[Tensor, "b seq_len"] = results["input_ids"]  # type: ignore
            attn_mask: Int64[Tensor, "b seq_len"] = results["attention_mask"]  # type: ignore
        elif mode == "decoder":
            BOS_ID = self.config.text_config.decoder_start_token_id
            assert BOS_ID is not None
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

    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        if decoder_input_ids is None:
            batch_size = input_embeddings.shape[0]
            decoder_input_ids = torch.zeros(
                batch_size, 1, dtype=torch.int, device=self.device
            )
        return self.model.language_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        ).logits


class Blip2LensDecOnly(
    Blip2Lens[BlipForConditionalGenerationT, BlipProcessorT],
    Generic[BlipForConditionalGenerationT, BlipProcessorT],
):
    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: Literal["encoder", "decoder", "no_special_tokens"],
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
    ) -> Tuple[Int64[Tensor, "b max_seq_len"], Bool[Tensor, "b max_seq_len"]]:
        """
        - Encoder:
            [BOS] 0 1 2
              [BOS] 0 1
        - Decoder:
            [BOS] 0 1 2
            [BOS] 0 1
        - No special tokens:
            0 1 2
            0 1

        What's fed to the model?
        [BOS] 0 1 2 0 1 2
          [BOS] 0 1 0 1
        (n.b. the decoder BOS is stripped and included only for ease of logit handling)
        """
        BOS_ID = self.config.text_config.bos_token_id
        assert self.tokenizer.padding_side == "left"
        if mode == "encoder":
            results = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids, attn_mask = results["input_ids"], results["attention_mask"]
        elif mode == "decoder":
            self.tokenizer.padding_side = "right"
            # NOTE: Tokenising in this way apparently naturally encodes a space???
            # if isinstance(text, str):
            #     text = " " + text
            # else:
            #     text = [" " + i for i in text]
            results = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
                add_special_tokens=False,
            )
            self.tokenizer.padding_side = "left"
            input_ids, attn_mask = results["input_ids"], results["attention_mask"]
            input_ids = F.pad(
                input=input_ids, pad=(1, 0, 0, 0), mode="constant", value=BOS_ID
            )
            attn_mask = F.pad(
                input=attn_mask, pad=(1, 0, 0, 0), mode="constant", value=0
            )
        elif mode == "no_special_tokens":
            self.tokenizer.padding_side = "right"
            results = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length" if pad_to_max_length else "longest",
                truncation="longest_first",
                max_length=max_length,
                add_special_tokens=False,
            )
            self.tokenizer.padding_side = "left"
            input_ids, attn_mask = results["input_ids"], results["attention_mask"]
        else:
            assert False
        return input_ids.to(self.device), attn_mask.to(self.device)

    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        if decoder_input_ids is None:
            batch_size = input_embeddings.shape[0]
            decoder_input_ids = torch.zeros(
                batch_size, 1, dtype=torch.int, device=self.device
            )
        decoder_embeddings = self.get_token_embeddings(decoder_input_ids)

        if attention_mask is None:
            attention_mask = get_full_attention_mask(input_embeddings)
        if decoder_attention_mask is None:
            decoder_attention_mask = get_full_attention_mask(decoder_embeddings)

        # Remove BOS token from start of decoder embeddings
        embeddings = torch.cat([input_embeddings, decoder_embeddings[:, 1:]], dim=1)
        attention_masks = torch.cat(
            [attention_mask, decoder_attention_mask[:, 1:]], dim=1
        )

        logits = self.model.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention_masks,
            return_dict=True,
        ).logits
        return logits[:, input_embeddings.shape[1] - 1 :, :]


# === Models ===


class Blip2Opt2p7b(Blip2LensEncDec[Blip2ForConditionalGeneration, Blip2Processor]):
    @classmethod
    def load_model(
        cls: Type["Blip2Opt2p7b"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ):
        return cls.load_model_and_processor(
            PROJECT_ROOT / "downloads/model_checkpoints/blip2-opt-2.7b",
            is_instructblip=False,
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class Blip2FlanT5Xl(Blip2LensEncDec[Blip2ForConditionalGeneration, Blip2Processor]):
    @classmethod
    def load_model(
        cls: Type["Blip2FlanT5Xl"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ):
        return cls.load_model_and_processor(
            PROJECT_ROOT / "downloads/model_checkpoints/blip2-flan-t5-xl",
            is_instructblip=False,
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class Blip2FlanT5XlCoco(Blip2LensEncDec[Blip2ForConditionalGeneration, Blip2Processor]):
    @classmethod
    def load_model(
        cls: Type["Blip2FlanT5XlCoco"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ):
        return cls.load_model_and_processor(
            PROJECT_ROOT / "downloads/model_checkpoints/blip2-flan-t5-xl-coco",
            is_instructblip=False,
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class InstructBlipFlanT5Xl(
    Blip2LensEncDec[InstructBlipForConditionalGeneration, InstructBlipProcessor]
):
    @classmethod
    def load_model(
        cls: Type["InstructBlipFlanT5Xl"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ):
        return cls.load_model_and_processor(
            PROJECT_ROOT / "downloads/model_checkpoints/instructblip-flan-t5-xl",
            is_instructblip=True,
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )


class InstructBlipVicuna7b(
    Blip2LensDecOnly[InstructBlipForConditionalGeneration, InstructBlipProcessor]
):
    @classmethod
    def load_model(
        cls: Type["InstructBlipVicuna7b"],
        model_dtype: torch.dtype = torch.half,
        requires_grad: bool = False,
    ):
        return cls.load_model_and_processor(
            PROJECT_ROOT / "downloads/model_checkpoints/instructblip-vicuna-7b",
            is_instructblip=True,
            model_dtype=model_dtype,
            requires_grad=requires_grad,
        )
