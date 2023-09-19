import dataclasses
import importlib.util
import shutil
import sys
import time
from image_hijacks.config import Config
import pathspec

import click

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
from lightning.fabric import seed_everything
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import DataLoader
import wandb
import uuid
import pickle

from image_hijacks.utils import load_config_list


@click.group()
def cli():
    pass


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    https://gist.github.com/Microsheep/11edda9dee7c1ba0c099709eb7f8bea7
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, "__name__") else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def backup_parent_codebase(src: Path, dst: Path) -> Optional[Path]:
    """Makes a ZIP copy of the codebase containing the file / directory src,
    respecting the .gitignore file, saved to dst/codebase-%Y%m%d-%H%M%S.zip

    We walk up the directory tree until we encounter the first folder containing a
    .gitignore file.

    Args:
        src (Path): Path within codebase
        dst (Path): Path to save ZIP backup

    Returns:
        Path: The full path to the backed-up codebase
    """
    # taken from https://waylonwalker.com/til/gitignore-python/
    while not (src / ".gitignore").exists():
        if src == src.parent:
            return None
        src = src.parent

    files = src.glob("**/*")
    lines = (src / ".gitignore").read_text().splitlines() + [
        ".git",
        "experiments",
        "wandb",
    ]
    spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)

    matched_files = [file for file in files if not spec.match_file(str(file))]

    codebase_name = f'codebase-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4()}'
    dst_folder = dst / codebase_name

    for file in matched_files:
        if os.path.isdir(file):
            continue
        dst_path = dst_folder / file.relative_to(src)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(file, dst_path)

    shutil.make_archive(str(dst_folder).rstrip("/"), "zip", dst_folder)
    shutil.rmtree(dst_folder)
    return dst / f"{codebase_name}.zip"


# fmt: off
@cli.command()
@click.option("--config_path", type=click.Path(path_type=Path), required=True)
@click.option("--log_dir", type=click.Path(path_type=Path), required=True)
@click.option("--playground/--no-playground", type=bool, default=False)
@click.option("--job_id", type=int, default=0)
@click.option("--wandb_project", type=str, default=None)
@click.option("--wandb_entity", type=str, default=None)
# fmt: on
def train(
    config_path: Path,
    log_dir: Path,
    playground: bool,
    job_id: int,
    wandb_project: str,
    wandb_entity: str,
):
    torch.set_float32_matmul_precision("high")
    exp_path = config_path.parent
    exp_name = exp_path.name
    print(f"Experiment {exp_name}")
    print(f"Loading config from {config_path}")
    configs = load_config_list(config_path)
    run_name, cfg_gen = configs[job_id]
    config = cfg_gen()

    print(f"Run {run_name}")
    seed_everything(config.seed)
    print("Dumping config:")
    print(config)

    if playground:
        print("In playground mode: not backing up codebase")
    else:
        save_path = log_dir / run_name  # type: ignore
        backup_path = backup_parent_codebase(
            Path(os.path.realpath(__file__)).parent, save_path
        )
        if backup_path is not None:
            print(f"Codebase backed up at {backup_path}")
        else:
            print("Failed to back up codebase")

    # Validation arguments
    callbacks_args = {}
    trainer_args = {}
    if config.validate_every is None:
        # TODO: implement
        assert 1 == 0
    else:
        n, val_on = config.validate_every
        if val_on == "steps":
            callbacks_args = {"every_n_train_steps": n}
            trainer_args = {"val_check_interval": n}
        elif val_on == "epochs":
            callbacks_args = {"every_n_epochs": n}
            trainer_args = {"check_val_every_n_epoch": n}
        else:
            raise ValueError

    callbacks: List[pl.Callback] = [  # type: ignore
        ModelCheckpoint(
            monitor=config.monitor_name,
            mode="max",
            save_last=True,
            save_top_k=5,
            filename=f"epoch={{epoch}}-step={{step}}-val_opt_acc={{{config.monitor_name}:.6f}}",
            auto_insert_metric_name=False,
            # save_on_train_epoch_end=False,
            **callbacks_args,
        ),
        # LearningRateMonitor(logging_interval="step"),
    ]
    loggers: List[Logger] = [
        TensorBoardLogger(
            save_dir=Path(log_dir),
            name=None,
            version=run_name,
        ),
    ]
    if playground:
        pass
        # callbacks.append(RichProgressBar())
    else:
        wandb_logger = WandbLogger(
            save_dir=Path(log_dir),
            name=run_name,
            project=wandb_project,
            entity=wandb_entity,
            tags=[exp_name],
        )
        # wandb_logger.experiment.config["exp_group"] = f"{exp_tag}_{full_spec}"
        loggers.append(wandb_logger)
        wandb.init(
            dir=Path(log_dir),
            name=run_name,
            project=wandb_project,
            entity=wandb_entity,
            tags=[exp_name],
        )
        wandb.save(str(config_path))
        wandb.Table.MAX_ARTIFACTS_ROWS = 10000000
        if backup_path is not None:
            wandb.save(str(backup_path))
        wandb_logger.experiment.config.update(
            transform_dict(dataclasses.asdict(config))
        )

    trainer = pl.Trainer(
        accelerator="auto",
        # accelerator="cpu",
        devices=1,
        max_epochs=config.epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=loggers,
        **trainer_args,
        **config.trainer_args,
    )
    attack_driver = config.attack_driver_factory(config)
    datamodule = attack_driver.get_datamodule()
    trainer.validate(attack_driver, datamodule)
    trainer.fit(attack_driver, datamodule)
    print("Val: Best model")
    trainer.validate(attack_driver, datamodule, ckpt_path="best")
    print("Val: Last model")
    trainer.validate(attack_driver, datamodule, ckpt_path="last")
    print("Test: Best model")
    trainer.test(attack_driver, datamodule, ckpt_path="best")
    print("Test: Last model")
    trainer.test(attack_driver, datamodule, ckpt_path="last")


# fmt: off
@cli.command()
@click.option("--config_path", type=click.Path(path_type=Path), required=True)
@click.option("--log_dir", type=click.Path(path_type=Path), required=True)
@click.option("--job_id_min", type=int, default=0)
@click.option("--job_id_max", type=int, default=-1)
# fmt: on
def test(
    config_path: Path,
    log_dir: Path,
    job_id_min: int,
    job_id_max: int,
):
    # job id min / max is inclusive...
    torch.set_float32_matmul_precision("high")
    exp_path = config_path.parent
    exp_name = exp_path.name
    print(f"Experiment {exp_name}")
    print(f"Loading config from {config_path}")

    configs = load_config_list(config_path)

    if job_id_max == -1:
        job_id_max = len(configs) - 1

    for run_name, cfg_gen in configs[job_id_min : job_id_max + 1]:
        config = cfg_gen()

        print(f"Run {run_name}")
        seed_everything(config.seed)
        print("Dumping config:")
        print(config)

        callbacks: List[pl.Callback] = [RichProgressBar()]
        loggers: List[Logger] = [
            TensorBoardLogger(
                save_dir=Path(log_dir),
                name=None,
                version=f"test_{run_name}_{time.time()}",
            ),
        ]

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            logger=loggers,
            **config.trainer_args,
        )

        if config.load_checkpoint_from_path is not None:
            attack_driver = config.load_attack_driver_from_checkpoint(
                config.load_checkpoint_from_path
            )
        else:
            attack_driver = config.attack_driver_factory(config)

        datamodule = attack_driver.get_datamodule()
        output = trainer.test(attack_driver, datamodule)
        with open(Path(log_dir) / f"test_{run_name}.pkl", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    cli()  # type: ignore
