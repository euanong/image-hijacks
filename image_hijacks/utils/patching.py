from jaxtyping import Shaped, Int64
from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, repeat


def get_patches(
    images: Shaped[Tensor, "b c h w"],
    top_left_rows: Int64[Tensor, "b"],
    top_left_cols: Int64[Tensor, "b"],
    patch_h: int,
    patch_w: int,
) -> Shaped[Tensor, "b c patch_h patch_w"]:
    b, c, h, w = images.shape
    start_rows = rearrange(top_left_rows, "b -> b () () ()")
    start_cols = rearrange(top_left_cols, "b -> b () () ()")

    # Generate the full indices for the patches
    row_indices = start_rows + repeat(
        torch.arange(patch_h, device=images.device),
        "patch_h -> b c patch_h w",
        b=b,
        c=c,
        w=w,
    )
    col_indices = start_cols + repeat(
        torch.arange(patch_w, device=images.device),
        "patch_w -> b c patch_h patch_w",
        b=b,
        c=c,
        patch_h=patch_h,
    )

    # Use gather to extract the patches
    patches_row = torch.gather(images, 2, row_indices)
    patches = torch.gather(patches_row, 3, col_indices)

    return patches


def set_patches(
    images: Shaped[Tensor, "b c h w"],
    patches: Shaped[Tensor, "b c m n"],
    top_left_rows: Int64[Tensor, "b"],
    top_left_cols: Int64[Tensor, "b"],
) -> Shaped[Tensor, "b c h w"]:
    b, c, h, w = images.shape
    _, _, m, n = patches.shape

    updated_images = images.clone()

    # Randomly select the starting row and column indices
    start_rows = rearrange(top_left_rows, "b -> b () () ()")
    start_cols = rearrange(top_left_cols, "b -> b () () ()")

    # Construct the full mesh of indices for the batch dimension, row dimension, and column dimension
    row_indices = start_rows + rearrange(
        torch.arange(m, dtype=torch.int64), "h -> () () h ()"
    )
    col_indices = start_cols + rearrange(
        torch.arange(n, dtype=torch.int64), "w -> () () () w"
    )

    # Use the indices to place the patch into the target image
    updated_images[
        rearrange(torch.arange(b), "b -> b () () ()"),
        rearrange(torch.arange(c), "c -> () c () ()"),
        row_indices,
        col_indices,
    ] = patches

    return updated_images
