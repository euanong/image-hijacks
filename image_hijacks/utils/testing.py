from pprint import pformat

import expecttest as expecttest
import torch


class TestCase(expecttest.TestCase):
    def assertExpectedIgnoreWhitespace(self, actual, expected):
        return self.assertExpectedInline(
            "".join(actual.split()), "".join(expected.split()), skip=1
        )

    def assertExpectedPretty(
        self, actual, expected, width=120, postprocess=None, **kwargs
    ):
        torch.set_printoptions(precision=3)
        actual = pformat(actual, width=width, **kwargs)
        if postprocess:
            actual = postprocess(actual)
        torch.set_printoptions(profile="default")
        return self.assertExpectedInline(actual, expected, skip=1)
