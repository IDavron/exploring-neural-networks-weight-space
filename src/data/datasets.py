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
        angle = torch.tensor(self.angles.iloc[idx]/self.angle_change, dtype=torch.int64)
        angle = torch.nn.functional.one_hot(angle, num_classes=int(360/self.angle_change)).float()

        return weights, angle
    
    def normalize(self):
        self.max_weight = self.weights.max().max()
        self.min_weight = self.weights.min().min()
        self.weights = (self.weights - self.min_weight) / (self.max_weight - self.min_weight)
        return self.min_weight, self.max_weight

    def denormalize(self, weights):
        return weights * (self.max_weight - self.min_weight) + self.min_weight
    

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

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return new_weights, new_biases

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        label = int(self.dataset["angle"][item])

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)


if __name__ == "__main__":
    dataset = ModelDataset("data/dataset_splits.json")
    print(dataset[0])