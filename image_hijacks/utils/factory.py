from __future__ import annotations

from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from image_hijacks.config import Config

T = TypeVar("T")


class Factory(Generic[T]):
    def __init__(self: "Factory[T]", f: Callable[[Config], T]) -> None:
        self.f = f
        self.result: Optional[Tuple[Config, T]] = None

    def __call__(self, config: Config) -> T:
        if self.result is None:
            self.result = (config, self.f(config))
        else:
            assert config == self.result[0]
        return self.result[1]
