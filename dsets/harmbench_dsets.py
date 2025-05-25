import json
import typing
from pathlib import Path

from torch.utils.data import Dataset

from util.globals import *


class HarmbenchDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        data_name: str = '',
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        if data_name != '':
            cf_loc = data_dir / data_name
        else:
            cf_loc = data_dir / (
                "Harmbench.json" if not multi else "multi_counterfact.json"
            )

        with open(cf_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiHarmbenchDataset(HarmbenchDataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)
