import torch
import pandas as pd

import json
from typing import NamedTuple, Tuple, Union


# Custom dataset with our model parameters
class ModelParamsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, angle_change: int = 45):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.angle_change = angle_change
        self.weights = self.dataset.drop(columns=["model_name", "angle"]).astype('float32')
        self.angles = self.dataset["angle"]
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weights = torch.tensor(self.weights.iloc[idx].values)
        angle = torch.tensor(self.angles.iloc[idx], dtype=torch.int64)

        return weights, angle
    
    def normalize(self):
        self.max_weight = self.weights.max().max()
        self.min_weight = self.weights.min().min()
        self.weights = (self.weights - self.min_weight) / (self.max_weight - self.min_weight)
        return self.min_weight, self.max_weight

    def denormalize(self, weights):
        return weights * (self.max_weight - self.min_weight) + self.min_weight
    


# Source code from Equivariant Architectures for Learning in Deep Weight Spaces
# https://github.com/AvivNavon/DWSNets

class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])

class ModelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        permutation=False,
        statistics_path="data/statistics.pth",
    ):
        # assert split in ["test", "train"]
        self.split = split
        self.dataset = json.load(open(path, "r"))[self.split]

        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            self.stats = torch.load(statistics_path, map_location="cpu")

    def __len__(self):
        return len(self.dataset["angle"])

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        label = int(int(self.dataset["angle"][item]) / 45)
        label = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=8
        ).float()

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)


if __name__ == "__main__":
    dataset = ModelDataset("data/dataset_splits.json")
    print(dataset[0])