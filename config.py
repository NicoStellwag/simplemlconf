from typing import Any, Union
from os import environ
from functools import lru_cache
from pathlib import Path


class Config:

    # ==========================================================================

    LEARNING_RATE: float = 5e-4
    DS_MODE: str = "val"
    EPOCHS: int = 10
    BATCH_SIZE = 64  # you don't have to type them

    def _validate(self):
        assert 0.0 < self.LEARNING_RATE < 1.0
        assert self.DS_MODE in ["train", "val", "test"]
        assert self.EPOCHS > 0

    # ==========================================================================

    def __init__(self):
        self._validate()

    @lru_cache(maxsize=None)
    def __getattribute__(self, __name: str) -> Any:
        val = object.__getattribute__(self, __name)
        overwrite = environ.get(__name, None)
        return type(val)(overwrite) if overwrite else val

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise NotImplementedError(
            "Don't change config values from the training script."
        )

    def dump(self, filename: Union[str, Path] = None):
        entries = {
            prop: getattr(self, prop)
            for prop in dir(self)
            if not callable(getattr(self, prop)) and not prop.startswith("__")
        }
        lines = [f"{k:40}{v}" for k, v in entries.items()]
        if filename:
            if isinstance(filename, str):
                filename = Path(filename)
            filename.parent.mkdir(exist_ok=True, parents=True)
            filename.touch(exist_ok=True)
            with open(filename, "w") as fl:
                fl.writelines([l + "\n" for l in lines])
        else:
            for l in lines:
                print(l)
