from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    List,
    TypeVar,
    Union,
)

import torch

from image_hijacks.attacks import (
    AttackDriver,
)
from image_hijacks.attacks.context import (
    ContextLabelAttack,
    LeakContextAttack,
    RepeatContextAttack,
    SpecificOutputStringAttack,
    JailbreakAttack,
)
from image_hijacks.components.loss import (
    EmbeddingMatchingLossFn,
    EmbeddingMatchingTarget,
    Loss,
    Text,
    VLMCrossEntropyLoss,
    flattened_cosine_similarity_loss,
)
from image_hijacks.components.grad import (
    GradientEstimator,
    ExactGradientEstimator,
)
from image_hijacks.components.optim import AttackOptimizer, TorchOptOptimizer
from image_hijacks.components.processor import (
    LearnedImageProcessor,
    Processor,
    StaticPatchImageProcessor,
    RandomPatchImageProcessor,
)
from image_hijacks.data import (
    AlpacaDataModule,
    AlpacaLlavaDataModule,
    CSVDataModule,
    ContextLabelDataModule,
    WikitextDataModule,
    FixedContextsDataModule,
)
from image_hijacks.models import AbstractLensModel
from image_hijacks.utils import PROJECT_ROOT
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

import torchopt
from torchopt.typing import GradientTransformation

from image_hijacks.utils.factory import Factory

from PIL.Image import Image

T = TypeVar("T")

# fmt: off
@dataclass
class Config:
    target_models_train: Dict[str, AbstractLensModel]
    target_models_eval: Dict[str, AbstractLensModel]
    seed: int

# === Model ===
    randomly_sample_system_prompt: bool = False

# === Init image ===
    init_image: Factory[Float[Tensor, "b c h w"]] = Factory(lambda _: torch.zeros((1, 3, 500, 500)))
    downsample: Callable[
        [Float[Tensor, "b c h w"], AbstractLensModel], 
        Float[Tensor, "b c h_new w_new"]
    ] = lambda img, model: F.interpolate(img, size=model.input_image_dims(), antialias=True, mode="bilinear")

# === Processor ===
    processor_factory: Type[Processor] = LearnedImageProcessor
    
    # - PatchImageProcessor
    init_patch: Factory[Float[Tensor, "b c patch_h patch_w"]] = Factory(lambda _: torch.rand(1, 3, 50, 50))
    static_patch_loc: Factory[Tuple[int, int]] = StaticPatchImageProcessor.center_patch_at(r=0.2, c=0.5, relative=True)
    #   ordered tuple of (row, column)

# === Attack type ===
    attack_driver_factory: Type[AttackDriver] = SpecificOutputStringAttack

    # - SpecificOutputStringAttack
    target_string: str = ""
    
    # - JailbreakAttack
    jailbreak_output_len_threshold: int = 10
    #   Number of CHARACTERS (not tokens) by which we must exceed context in order to mark successful jailbreak.

# === Losses ===
    loss_fn_factory: Type[Loss] = VLMCrossEntropyLoss
    whitebox_model: Optional[AbstractLensModel] = None
    
    # - EmbeddingMatchingLoss
    embedding_matching_target: Factory[EmbeddingMatchingTarget] = Factory(lambda _: Text("Jane's bank account number is 123456789"))
    embedding_matching_loss_fn: EmbeddingMatchingLossFn = flattened_cosine_similarity_loss
    embedding_matching_pad_target_seq: bool = False
    #   In Text case, whether or not to pad target token sequence if it's shorter than the number of
    #   image embedding tokens. If True, we pad our token sequence to match number of
    #   image embedding tokens; if False, we truncate our image embedding token sequence
    #   to match number of target tokens.
    #   Note: we do not support target token sequences longer than the number of image embedding tokens.
    embedding_matching_n_gen_tokens: int = 30
    #   Number of tokens to generate during evaluation

# === Gradient estimators ===
    gradient_estimator_factory: Type[GradientEstimator] = ExactGradientEstimator
    
    # - RGFGradientEstimator
    rgf_n_queries: int = 100
    rgf_batch_size: int = 2
    rgf_sigma: float = 16 / 255

# === Optimisers ===
    attack_optimizer_factory: Type[AttackOptimizer] = TorchOptOptimizer
    
    image_update_eps: float = 1.0 # by default, no clamping
    #   Max Lp-norm away from original image
    
    # - TorchOptOptimizer
    torchopt_optimizer: Factory[GradientTransformation] = Factory(lambda config: torchopt.sgd(lr=config.lr))
    adam_eps: float = 1e-7
    #   Warning: any lower and float16 doesn't like it
    
    # - IterFGSMOptimizer
    iterfgsm_alpha: float = 1.0 / 255
    
# === Dataset ===
    context_data_module_train: Type[ContextLabelDataModule] = WikitextDataModule
    context_data_modules_eval: Dict[str, Type[ContextLabelDataModule]] = field(default_factory=lambda: {"wikitext": WikitextDataModule})
    data_root: Path = PROJECT_ROOT / "downloads/data"
    dataset_gen_seed: int = 1337
    batch_size: int = 1
    eval_batch_size: int = 1
    train_max_length: Optional[int] = 200
    # Max STRING length of sequences for training.
    # - If provided, sequences padded / truncated to this length
    # - If not provided, sequences padded to max sequence length
    test_max_gen_length: Optional[int] = None
    #   (Max length of strings generated when testing:
    #   if None, we generate strings of length len(label) + test_max_extra_gen_length)
    test_max_extra_gen_length: int = 0

    # - AlpacaDataModule (61002 elements)
    alpaca_train_split_size: int = 59002
    alpaca_val_split_size: int = 1000
    alpaca_test_split_size: int = 1000
    
    # - LlavaDataModule (157712 elements)
    llava_train_split_size: int = 156712
    llava_val_split_size: int = 1000
    llava_test_split_size: int = 1000
    
    # - AlpacaLlavaDataModule ()_train_split_size: int = 59000
    alpaca_llava_train_split_size: Tuple[int, int] = (59000, 59000)
    alpaca_llava_val_split_size: Tuple[int, int] = (100, 100)
    alpaca_llava_test_split_size: Tuple[int, int] = (1000, 1000)


    # - WikitextDataModule (36718 elements train / 4358 elements test)
    wikitext_val_split_size: int = 1000

    # - FixedContextsDataModule
    fixed_context_list: List[str] = field(default_factory=lambda: [""])

    # - CSVDataModule
    csv_path: Path = PROJECT_ROOT
    csv_ctx_col: str = "goal"
    csv_label_col: str = "target"
    csv_val_split_size: int = 50
    csv_test_split_size: int = 50

# === Training ===
    clip_grad_mag: Optional[float] = None
    #   (L2 norm to use for grad clipping, if None then no clipping is used.)
    lr: float = 0.005
    epochs: int = 100
    #   (Number of times to iterate over train_contexts. If train_contexts is
    #   no context, then this number should be very high, e.g. 4000.)

# === Experiment ===
    trainer_args: Dict[str, Any] = field(default_factory=lambda: {
        # "val_check_interval": 20
        # float = frac of epoch; int = every n batches
        # limit_val_batches=0.0 for no val
        # check_val_every_n_epoch for epoch-wise val
    })
    monitor_name: str = "val_acc"
    validate_every: Optional[
        Tuple[int, Union[Literal["steps"], Literal["epochs"]]]
    ] = (20, "steps")

# === Evaluation ===
    load_checkpoint_from_path: Optional[Path] = None

# fmt: on

    def lift_to_model_device_dtype(self, x: T) -> T:
        dummy_model = next(iter(self.target_models_train.values()))
        return x.to( # type: ignore
            dtype=dummy_model.model_dtype,
            device=dummy_model.device,
        )

    def get_datamodule_names(self) -> List[str]:
        return list(self.context_data_modules_eval.keys())

    def load_attack_driver_from_checkpoint(self, path) -> AttackDriver:
        return self.attack_driver_factory.load_from_checkpoint(path, config=self)

# === Transforms ===



@dataclass
class Transform:
    fn: Union[Callable[[Config], None], List[Callable[[Config], None]]]
    key: Optional[str] = None

    def apply(self, config: Config) -> None:
        if isinstance(self.fn, list):
            for f in self.fn:
                f(config)
        else:
            self.fn(config)


def concat_transforms(fs: Sequence[Transform]) -> Transform:
    ids = [f.key for f in fs if f.key is not None]

    def apply(config: Config):
        for f in fs:
            f.apply(config)

    return Transform(key=".".join(ids), fn=apply)


def compose_sweeps(
    fs: Sequence[Callable[[List[str]], List[Transform]]]
) -> List[Transform]:
    key_lists: List[List[str]] = [[]]
    transform_lists: List[List[Transform]] = [[Transform(lambda config: None)]]
    for f in fs:
        new_keys_transforms = [
            (key_list + [new_transform.key], transform_list + [new_transform])
            for key_list, transform_list in zip(key_lists, transform_lists)
            for new_transform in f(key_list)
        ]
        key_lists, transform_lists = zip(*new_keys_transforms)  # type: ignore
    return [concat_transforms(ts) for ts in transform_lists]

# === Defaults ===    

# Processors

def proc_learnable_image(config: Config):
    config.processor_factory = LearnedImageProcessor

def proc_patch_static(config: Config, patch_h: int = 50, patch_w: int = 50, rel_r: float = 0.2, rel_c: float = 0.5):
    config.processor_factory = StaticPatchImageProcessor
    config.init_patch = Factory(lambda _: torch.rand(1, 3, patch_h, patch_w))
    config.static_patch_loc = StaticPatchImageProcessor.center_patch_at(r=rel_r, c=rel_c, relative=True)

def proc_patch_random_loc(config: Config, patch_h: int = 50, patch_w: int = 50):
    config.processor_factory = RandomPatchImageProcessor
    config.init_patch = Factory(lambda _: torch.rand(1, 3, patch_h, patch_w))

# Optimisers

def opt_sgd(config: Config, clip_grad_mag: float = 20):
    config.attack_optimizer_factory = TorchOptOptimizer
    config.torchopt_optimizer = Factory(
        lambda config: torchopt.sgd(lr=config.lr)
    )
    config.clip_grad_mag = clip_grad_mag

def opt_adam(config: Config, clip_grad_mag: float = 20):
    config.attack_optimizer_factory = TorchOptOptimizer
    config.torchopt_optimizer = Factory(
        lambda config: torchopt.adam(lr=config.lr, eps=config.adam_eps)
    )
    config.clip_grad_mag = clip_grad_mag

# Parameters

def set_input_image(config: Config, img: Image):
    config.init_image = Factory(
        lambda config: next(iter(config.target_models_train.values())).preprocess_image(
            img
        )[0]
    )

def set_eps(config: Config, eps: float):
    config.image_update_eps = eps

def set_lr(config: Config, lr: float):
    config.lr = lr
