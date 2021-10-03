from collections import Counter

import numpy as np
import pandas as pd
import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, balancing_coeff, indices=None, num_samples=None):
        """
        Dataset sampler - "balanced" sampling of samples with different labels.
        Creates samples with replacement.
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = (
            len(self.indices) if (num_samples is None) or (num_samples > len(self.indices)) else num_samples
        )

        weights = self.calculate_target_weights(balancing_coeff)
        self.weights = torch.DoubleTensor(weights)

    def calculate_target_weights(self, balancing_coeff):
        track_nums = []
        for dataset in self.dataset.datasets:
            track_nums.extend([dataset.track_num] * len(dataset))

        label_to_count = Counter(track_nums)
        balancing_coeff = eval(balancing_coeff)
        weights = np.asarray(
            [1.0 / label_to_count[track_nums[idx]] * float(balancing_coeff[track_nums[idx]]) for idx in self.indices]
        )
        return weights

    def generate_dataset_sample(self):
        indices = np.asarray([self.indices[i] for i in torch.multinomial(self.weights, self.num_samples)])
        return indices

    def __iter__(self):
        indices = self.generate_dataset_sample()
        return iter(indices)

    def __len__(self):
        return self.num_samples


class UndersampledDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, num_samples=None):
        """
        Dataset sampler - takes subset of each dataset of size `sample_size` or less.
        Creates samples without replacement.
        """
        self.dataset = dataset
        self.sample_size = (
            num_samples // len(dataset.datasets)
            if num_samples is not None
            else max([len(ds) for ds in dataset.datasets])
        )
        self._init_markup()
        self.random_state = 0
        self.generate_dataset_sample()

    def _init_markup(self):
        track_nums = []
        for dataset in self.dataset.datasets:
            track_nums.extend([dataset.track_num] * len(dataset))
        self.markup = pd.DataFrame(np.vstack([track_nums]).T, columns=["track_num"])

    def generate_dataset_sample(self):
        self.random_state += 1
        indices = (
            self.markup.sample(frac=1, random_state=self.random_state)
            .groupby(["track_num"])
            .head(self.sample_size)
            .index.values
        )
        self.num_samples = indices.shape[0]
        return indices

    def __iter__(self):
        sample_indices = self.generate_dataset_sample()
        return iter(sample_indices)

    def __len__(self):
        return self.num_samples
