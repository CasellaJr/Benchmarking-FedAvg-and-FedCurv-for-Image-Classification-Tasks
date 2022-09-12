# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cifar10 Shard Descriptor."""

import logging
import os
from typing import List
from tensorflow import keras 
from tensorflow.keras.datasets import cifar10

import numpy as np
import requests

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class Cifar10ShardDataset(ShardDataset):
    """Cifar10 Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Initialize Cifar10Dataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class Cifar10ShardDescriptor(ShardDescriptor):
    """Cifar10 Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize Cifar10ShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return Cifar10ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['32', '32', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['32', '32', '3']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Cifar10 dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self):
        """Download prepared dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
        print('CIFAR10 data was loaded!')
        return (x_train, y_train), (x_test, y_test)
