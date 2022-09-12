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
from openfl.utilities.data_splitters import RandomNumPyDataSplitter
from openfl.utilities.data_splitters import QuantitySkewSplitter
from openfl.utilities.data_splitters import QuantitySkewLabelsSplitter
from openfl.utilities.data_splitters import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters import PathologicalSkewLabelsSplitter
from openfl.utilities.data_splitters import CovariateShiftSplitter2D
from openfl.utilities.data_splitters import CovariateShiftSplitter3D

logger = logging.getLogger(__name__)


class Cifar10ShardDataset(ShardDataset):
    """Cifar10 Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1):
        """Initialize Cifar10Dataset."""
        #np.random.seed(42)
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x
        self.y = y

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
        '''
        #QUANTITY SKEW FUNZIONANTE
        train_splitter = QuantitySkewSplitter(alpha=2)
        test_splitter = QuantitySkewSplitter(alpha=2)
        train_idx = train_splitter.split(y_train, self.worldsize)[self.rank-1]
        test_idx = test_splitter.split(y_test, self.worldsize)[self.rank-1]
        x_train_shard = x_train[train_idx]
        x_test_shard = x_test[test_idx]
        y_train_shard = y_train[train_idx]
        y_test_shard = y_test[test_idx]
        '''
        #QUANTITY SKEW LABELS
        '''
        train_splitter = QuantitySkewLabelsSplitter(class_per_client=2)
        train_idx = train_splitter.split(x_train, y_train, self.worldsize)[self.rank-1]
        x_train_shard = x_train[train_idx]
        y_train_shard = y_train[train_idx]
        myclasses = np.unique(y_train_shard)
        set(myclasses)
        idx = (y_test==myclasses[0]) | (y_test==myclasses[1])
        y_test_shard = y_test[idx]
        x_test_shard = x_test[idx]
        '''
        #DIRICHLET SKEW LABELS
        '''
        train_splitter = DirichletNumPyDataSplitter()
        test_splitter = DirichletNumPyDataSplitter()
        train_idx = train_splitter.split(y_train, self.worldsize)[self.rank-1]
        test_idx = test_splitter.split(y_test, self.worldsize)[self.rank-1]
        x_train_shard = x_train[train_idx]
        x_test_shard = x_test[test_idx]
        y_train_shard = y_train[train_idx]
        y_test_shard = y_test[test_idx]
        '''
        #PATHOLOGICAL SKEW LABELS
        '''
        train_splitter = PathologicalSkewLabelsSplitter(shards_per_client=3)
        test_splitter = PathologicalSkewLabelsSplitter(shards_per_client=3)
        train_idx = train_splitter.split(y_train, self.worldsize)[self.rank-1]
        x_train_shard = x_train[train_idx]
        y_train_shard = y_train[train_idx]
        myclasses = np.unique(y_train_shard)
        set(myclasses)
        if len(myclasses)==2:
          idx = (y_test==myclasses[0]) | (y_test==myclasses[1])
        elif len(myclasses)==3:
          idx = (y_test==myclasses[0]) | (y_test==myclasses[1]) | (y_test==myclasses[2])
        elif len(myclasses)==4:
          idx = (y_test==myclasses[0]) | (y_test==myclasses[1]) | (y_test==myclasses[2]) | (y_test==myclasses[3])
        elif len(myclasses)==5:
          idx = (y_test==myclasses[0]) | (y_test==myclasses[1]) | (y_test==myclasses[2]) | (y_test==myclasses[3]) | (y_test==myclasses[4])
        elif len(myclasses)==6:
          idx = (y_test==myclasses[0]) | (y_test==myclasses[1]) | (y_test==myclasses[2]) | (y_test==myclasses[3]) | (y_test==myclasses[4]) | (y_test==myclasses[5])
        y_test_shard = y_test[idx]
        x_test_shard = x_test[idx]
        '''
        #COVARIATE SKEW 
        
        train_splitter = CovariateShiftSplitter3D(x_train)
        test_splitter = CovariateShiftSplitter3D(x_test)
        train_idx = train_splitter.split(x_train, y_train, self.worldsize)[self.rank-1]
        test_idx = test_splitter.split(x_test, y_test, self.worldsize)[self.rank-1]
        x_train_shard = x_train[train_idx]
        x_test_shard = x_test[test_idx]
        y_train_shard = y_train[train_idx]
        y_test_shard = y_test[test_idx]

        self.data_by_type = {
            'train': (x_train_shard, y_train_shard),
            'val': (x_test_shard, y_test_shard)
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
