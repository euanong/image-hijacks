from typing import Any, Dict, List, Protocol
from huggingface_hub import snapshot_download
from pathlib import Path
import click
import llava.model.apply_delta

from image_hijacks.utils import PROJECT_ROOT


@click.group()
def cli():
    pass


# === Models ===


class ModelDownloader(Protocol):
    def __call__(self, model_id: str, model_dir: Path, **kwargs) -> None:
        ...


def hf(repo_id) -> ModelDownloader:
    def downloader(model_id: str, model_dir: Path, **kwargs) -> None:
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir / model_id,
            local_dir_use_symlinks=False,
            force_download=False,
        )

    return downloader


MODEL_IDS: Dict[str, ModelDownloader] = {
    "blip2-flan-t5-xl": hf("Salesforce/blip2-flan-t5-xl"),
    "blip2-flan-t5-xxl": hf("Salesforce/blip2-flan-t5-xxl"),
    "blip2-flan-t5-xl-coco": hf("Salesforce/blip2-flan-t5-xl-coco"),
    "blip2-flan-t5-xl-llava": hf("rulins/blip2-t5-llava"),
    "blip2-opt-2.7b": hf("Salesforce/blip2-opt-2.7b"),
    "instructblip-vicuna-7b": hf("Salesforce/instructblip-vicuna-7b"),
    "instructblip-vicuna-13b": hf("Salesforce/instructblip-vicuna-13b"),
    "instructblip-flan-t5-xl": hf("Salesforce/instructblip-flan-t5-xl"),
    "instructblip-flan-t5-xxl": hf("Salesforce/instructblip-flan-t5-xxl"),
    "llava-llama-2-13b-chat": hf("liuhaotian/llava-llama-2-13b-chat-lightning-preview"),
    "llava-llama-2-7b-chat": hf(
        "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"
    ),
    "llava-v1.3-13b-336px": hf(
        "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
    ),
}


@cli.command()  # @cli, not @click!
@click.argument("model_ids", type=str, nargs=-1)
@click.option(
    "--model_dir",
    type=click.Path(path_type=Path),
    default=PROJECT_ROOT / "downloads/model_checkpoints",
)
@click.option(
    "--llama_dir",
    type=click.Path(path_type=Path),
    default=None,
)
def models(model_ids: List[str], model_dir: Path, **kwargs):
    for model_id in model_ids:
        click.echo(f"Downloading {model_id} to {model_dir}")
        MODEL_IDS[model_id](model_id, model_dir, **kwargs)
    click.echo(f"Downloaded {len(model_ids)} models.")


# === Data ===
# ...

# === Init ===
if __name__ == "__main__":
    cli()  # type: ignore
