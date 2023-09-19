import torch
from image_hijacks.utils.testing import TestCase
from image_hijacks.utils.patching import get_patches, set_patches


class TestUtils(TestCase):
    def test_get_patches(self):
        images = torch.arange(4 * 2 * 10 * 10).reshape(4, 2, 10, 10)
        # First digit is batch / channel; second digit is row; last digit is column

        top_left_rows = torch.tensor([0, 1, 2, 3])
        top_left_cols = torch.tensor([0, 2, 4, 6])

        patches = get_patches(
            images, top_left_rows, top_left_cols, patch_h=3, patch_w=2
        )
        self.assertExpectedPretty(
            patches,
            """\
tensor([[[[  0,   1],
          [ 10,  11],
          [ 20,  21]],

         [[100, 101],
          [110, 111],
          [120, 121]]],


        [[[212, 213],
          [222, 223],
          [232, 233]],

         [[312, 313],
          [322, 323],
          [332, 333]]],


        [[[424, 425],
          [434, 435],
          [444, 445]],

         [[524, 525],
          [534, 535],
          [544, 545]]],


        [[[636, 637],
          [646, 647],
          [656, 657]],

         [[736, 737],
          [746, 747],
          [756, 757]]]])""",
        )

    def test_set_patches(self):
        patch_h, patch_w = 1, 2

        # First digit is batch / channel; second digit is row; last digit is column
        images = torch.arange(3 * 2 * 5 * 5).reshape(3, 2, 5, 5).float()
        patches = (
            torch.arange(3 * 2 * patch_h * patch_w).reshape(3, 2, patch_h, patch_w) + 1
        ) * -1.0

        top_left_rows = torch.tensor([0, 1, 2])
        top_left_cols = torch.tensor([2, 1, 0])

        # Insert patches into the images and allow for gradient propagation
        updated_images_grad = set_patches(images, patches, top_left_rows, top_left_cols)
        self.assertExpectedPretty(
            updated_images_grad,
            """\
tensor([[[[  0.,   1.,  -1.,  -2.,   4.],
          [  5.,   6.,   7.,   8.,   9.],
          [ 10.,  11.,  12.,  13.,  14.],
          [ 15.,  16.,  17.,  18.,  19.],
          [ 20.,  21.,  22.,  23.,  24.]],

         [[ 25.,  26.,  -3.,  -4.,  29.],
          [ 30.,  31.,  32.,  33.,  34.],
          [ 35.,  36.,  37.,  38.,  39.],
          [ 40.,  41.,  42.,  43.,  44.],
          [ 45.,  46.,  47.,  48.,  49.]]],


        [[[ 50.,  51.,  52.,  53.,  54.],
          [ 55.,  -5.,  -6.,  58.,  59.],
          [ 60.,  61.,  62.,  63.,  64.],
          [ 65.,  66.,  67.,  68.,  69.],
          [ 70.,  71.,  72.,  73.,  74.]],

         [[ 75.,  76.,  77.,  78.,  79.],
          [ 80.,  -7.,  -8.,  83.,  84.],
          [ 85.,  86.,  87.,  88.,  89.],
          [ 90.,  91.,  92.,  93.,  94.],
          [ 95.,  96.,  97.,  98.,  99.]]],


        [[[100., 101., 102., 103., 104.],
          [105., 106., 107., 108., 109.],
          [ -9., -10., 112., 113., 114.],
          [115., 116., 117., 118., 119.],
          [120., 121., 122., 123., 124.]],

         [[125., 126., 127., 128., 129.],
          [130., 131., 132., 133., 134.],
          [-11., -12., 137., 138., 139.],
          [140., 141., 142., 143., 144.],
          [145., 146., 147., 148., 149.]]]])""",
        )
