from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import List, Optional, Dict, Callable, Tuple, Type, TypeVar, cast
from urllib.request import urlretrieve

from datasets import load_dataset, Dataset
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd

from image_hijacks.models.blip2 import Blip2LensEncDec
from jaxtyping import Int64
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from typing import TYPE_CHECKING

from image_hijacks.utils import split_train_val_test

if TYPE_CHECKING:
    from image_hijacks.config import Config
from image_hijacks.models import AbstractLensModel
from lightning.pytorch.utilities import CombinedLoader


class WrapperContextDataModule(LightningDataModule):
    def __init__(
        self,
        config: Config,
        get_label: Callable[[str], str] = lambda x: "",
    ) -> None:
        super().__init__()
        self.config = config
        self.train_datamodule = {
            k: self.config.context_data_module_train(config, m, get_label)
            for k, m in self.config.target_models_train.items()
        }
        self.eval_datamodules = [
            {
                k: cdm(config, m, get_label)
                for k, m in self.config.target_models_eval.items()
            }
            for _, cdm in self.config.context_data_modules_eval.items()
        ]
        self.has_setup = False

    def setup(self, stage: str) -> None:
        if self.has_setup:
            return
        self.has_setup = True
        for datamodule_dict in [self.train_datamodule] + self.eval_datamodules:
            for m in datamodule_dict.values():
                m.setup(stage)

    def train_dataloader(self):
        return CombinedLoader(
            {k: m.train_dataloader() for k, m in self.train_datamodule.items()},
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return [
            CombinedLoader(
                {k: m.val_dataloader() for k, m in edm.items()},
                mode="max_size_cycle",
            )
            for edm in self.eval_datamodules
        ]

    def test_dataloader(self):
        return [
            CombinedLoader(
                {k: m.test_dataloader() for k, m in edm.items()},
                mode="max_size_cycle",
            )
            for edm in self.eval_datamodules
        ]


class ContextLabelDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        config: Config,
        target_model: AbstractLensModel,
        get_label: Callable[[str], str] = lambda x: "",
    ):
        """
        Build context data module.
        Args:
            get_label: function to apply to each context element to get the label
        """
        super().__init__()
        self.config = config
        self.target_model = target_model

    @abstractmethod
    def get_dataset(self) -> Dict[str, Tuple[List[str], List[str]]]:
        ...

    def setup(self, stage: str) -> None:
        dataset = self.get_dataset()
        # (seqs, attn_masks)
        self.datasets = {}
        for key in ["train", "val", "test"]:
            contexts, labels = dataset[key]
            self.datasets[key] = TensorDataset(
                *self.target_model.tokenize(
                    contexts,
                    mode="encoder",
                    max_length=self.config.train_max_length,
                    randomly_sample_system_prompt=self.config.randomly_sample_system_prompt,
                ),
                *self.target_model.tokenize(
                    labels,
                    mode="decoder",
                    max_length=self.config.train_max_length,
                ),
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.config.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.config.eval_batch_size)


class ContextOnlyDataModule(ContextLabelDataModule, ABC):
    def __init__(
        self,
        config: Config,
        target_model: AbstractLensModel,
        get_label: Callable[[str], str] = lambda x: "",
    ):
        """
        Build context data module.
        Args:
            get_label: function to apply to each context element to get the label
        """
        super().__init__(config, target_model, get_label)
        self.get_label = get_label

    @abstractmethod
    def get_contexts(self) -> Dict[str, List[str]]:
        ...

    def get_dataset(self) -> Dict[str, Tuple[List[str], List[str]]]:
        datasets = {}
        for key, contexts in self.get_contexts().items():
            labels = [self.get_label(i) for i in contexts]
            datasets[key] = (contexts, labels)
        return datasets


def Uniform(
    cls: Type[ContextOnlyDataModule], min_length: int = 1, max_length: int = 100
) -> Type[ContextOnlyDataModule]:
    def convert_dataset_to_uniform_words(
        sentences: List[str], min_length: int, max_length: int
    ) -> List[str]:
        # Generate uniform dataset whose entries have length between min_len and max_len inclusive
        words = [word for sentence in sentences for word in sentence.split()]

        new_sentences = []

        full_cycle_len = sum(range(min_length, max_length + 1))
        num_full_cycles = len(words) // full_cycle_len
        assert num_full_cycles > 0

        ptr = 0
        for _ in range(num_full_cycles):
            for length in range(min_length, max_length + 1):
                new_sentence = " ".join(words[ptr : ptr + length])
                new_sentences.append(new_sentence)
                ptr += length
        return new_sentences

    class UniformDataModule(cls):
        def get_contexts(self) -> Dict[str, List[str]]:
            return {
                k: convert_dataset_to_uniform_words(v, min_length, max_length)
                for k, v in super().get_contexts().items()
            }

    return UniformDataModule


class WikitextDataModule(ContextOnlyDataModule):
    def get_contexts(self) -> Dict[str, List[str]]:
        """Load Wikitext dataset.

        Dataset should be in json format as can be found at Alpaca github:
        https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

        Return:
            Dict with keys "test", "train", values being the respective dataset splits in list
        of string format.
        """

        # Right now this is loading to home directory which is bad, TODO fix this
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_val_data = [x["text"] for x in dataset["train"]]
        test_data = [x["text"] for x in dataset["test"]]
        Random(self.config.dataset_gen_seed).shuffle(train_val_data)
        return {
            "train": train_val_data[self.config.wikitext_val_split_size :],
            "val": train_val_data[: self.config.wikitext_val_split_size],
            "test": test_data,
        }


class AlpacaDataModule(ContextOnlyDataModule):
    @staticmethod
    def load_dataset(config: Config) -> List[str]:
        data_dir = config.data_root / "alpaca"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "alpaca_data.json"

        if not data_path.exists():
            urlretrieve(
                "https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json",
                data_path,
            )

        with open(data_path, "r") as file:
            data = json.load(file)

        return [item["instruction"] + " " + item["input"] for item in data]

    def get_contexts(self) -> Dict[str, List[str]]:
        """Load Alpaca dataset.

        Dataset should be in json format as can be found at Alpaca github:
        https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

        Return:
            Dict with keys "test", "train", values being the respective dataset splits in list
            of string format. In our case, instruction and input fields from original dataset are
            combined and label is removed, so we only have the question input
        """
        all_data = self.load_dataset(self.config)
        Random(self.config.dataset_gen_seed).shuffle(all_data)
        test_data = all_data[: self.config.alpaca_test_split_size]
        train_val_data = all_data[self.config.alpaca_test_split_size :]
        return {
            "train": train_val_data[self.config.alpaca_val_split_size :],
            "val": train_val_data[: self.config.alpaca_val_split_size],
            "test": test_data,
        }


class LlavaDataModule(ContextOnlyDataModule):
    @staticmethod
    def load_dataset(config: Config) -> List[str]:
        data_dir = config.data_root / "llava"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / "llava_instruct_150k.json"

        if not data_path.exists():
            urlretrieve(
                "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
                data_path,
            )

        with open(data_path, "r") as file:
            data = json.load(file)

        return [
            item["conversations"][0]["value"].replace("<image>", "").strip()
            for item in data
        ]

    def get_contexts(self) -> Dict[str, List[str]]:
        all_data = self.load_dataset(self.config)
        return split_train_val_test(
            all_data,
            seed=self.config.dataset_gen_seed,
            train_split=self.config.llava_train_split_size,
            val_split=self.config.llava_val_split_size,
            test_split=self.config.llava_test_split_size,
        )


class AlpacaLlavaDataModule(AlpacaDataModule):
    def get_contexts(self) -> Dict[str, List[str]]:
        llava_ds = LlavaDataModule.load_dataset(self.config)
        alpaca_ds = AlpacaDataModule.load_dataset(self.config)

        result = {}

        llava_ptr = 0
        alpaca_ptr = 0
        for i, (k, (llava_split, alpaca_split)) in enumerate(
            [
                ("train", self.config.alpaca_llava_train_split_size),
                ("val", self.config.alpaca_llava_val_split_size),
                ("test", self.config.alpaca_llava_test_split_size),
            ]
        ):
            ds = (
                llava_ds[llava_ptr : llava_ptr + llava_split]
                + alpaca_ds[alpaca_ptr : alpaca_ptr + alpaca_split]
            )
            llava_ptr += llava_split
            alpaca_ptr += alpaca_split
            Random(self.config.dataset_gen_seed + 1 + i).shuffle(ds)
            result[k] = ds
        return result


class FixedContextsDataModule(ContextOnlyDataModule):
    def get_contexts(self) -> Dict[str, List[str]]:
        return {
            "train": self.config.fixed_context_list,
            "val": self.config.fixed_context_list,
            "test": self.config.fixed_context_list,
        }


class CSVDataModule(FixedContextsDataModule):
    def get_dataset(self) -> Dict[str, Tuple[List[str], List[str]]]:
        df = pd.read_csv(self.config.csv_path)
        df = df.sample(frac=1, random_state=self.config.dataset_gen_seed).reset_index(
            drop=True
        )
        test_df = df[: self.config.csv_test_split_size]
        train_val_df = df[self.config.csv_test_split_size :]
        val_df = train_val_df[: self.config.csv_val_split_size]
        train_df = train_val_df[self.config.csv_val_split_size :]
        return {
            k: (list(v[self.config.csv_ctx_col]), list(v[self.config.csv_label_col]))
            for k, v in [("train", train_df), ("val", val_df), ("test", test_df)]
        }
