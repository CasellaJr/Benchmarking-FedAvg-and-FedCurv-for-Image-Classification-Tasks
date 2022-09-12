"""UnbalancedFederatedDataset module."""

from abc import abstractmethod
from typing import List

import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA
from scipy.stats.mstats import mquantiles

from openfl.utilities.data_splitters.data_splitter import DataSplitter
from numpy.random import randint, shuffle, power, choice, dirichlet, normal, permutation

def get_label_count(labels, label):
    """Count samples with label `label` in `labels` array."""
    return len(np.nonzero(labels == label)[0])


def one_hot(labels, classes):
    """Apply One-Hot encoding to labels."""
    return np.eye(classes)[labels]


class NumPyDataSplitter(DataSplitter):
    """Base class for splitting numpy arrays of data."""

    @abstractmethod
    def split(self, data: np.ndarray, num_collaborators: int) -> List[List[int]]:
        """Split the data."""
        raise NotImplementedError


class EqualNumPyDataSplitter(NumPyDataSplitter):
    """Splits the data evenly."""

    def __init__(self, shuffle=True, seed=0):
        """Initialize.

        Args:
            shuffle(bool): Flag determining whether to shuffle the dataset before splitting.
            seed(int): Random numbers generator seed.
                For different splits on envoys, try setting different values for this parameter
                on each shard descriptor.
        """
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        idx = range(len(data))
        if self.shuffle:
            idx = np.random.permutation(idx)
        slices = np.array_split(idx, num_collaborators)
        return slices

class QuantitySkewSplitter(NumPyDataSplitter):
    def __init__(self, min_quantity: int=2, alpha: float=2., seed=0):
        self.seed = seed
        self.min_quantity = min_quantity
        self.alpha = alpha
        
    def split(self, data, num_collaborators):       
        assert self.min_quantity*num_collaborators <= data.shape[0], "# of instances must be > than min_quantity*n"
        assert self.min_quantity > 0, "min_quantity must be >= 1"
        s = np.array(power(self.alpha, data.shape[0] - self.min_quantity*num_collaborators) * num_collaborators, dtype=int)
        m = np.array([[i] * self.min_quantity for i in range(num_collaborators)]).flatten()
        assignment = np.concatenate([s, m])
        shuffle(assignment)
        return [np.where(assignment == i)[0] for i in range(num_collaborators)]

class QuantitySkewLabelsSplitter(NumPyDataSplitter):
    def __init__(self, class_per_client, seed=0):
        self.seed = seed
        self.class_per_client = class_per_client
        
    def split(self, data, y, num_collaborators):       
        labels = set(y)
        assert 0 < self.class_per_client <= len(labels), "class_per_client must be > 0 and <= #classes"
        assert self.class_per_client * num_collaborators >= len(labels), "class_per_client * n must be >= #classes"
        nlbl = [choice(len(labels), self.class_per_client, replace=False)  for u in range(num_collaborators)]
        check = set().union(*[set(a) for a in nlbl])
        while len(check) < len(labels):
            missing = labels - check
            for m in missing:
                nlbl[randint(0, num_collaborators)][randint(0, self.class_per_client)] = m
            check = set().union(*[set(a) for a in nlbl])
        class_map = {c:[u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
        assignment = np.zeros(y.shape[0])
        for lbl, users in class_map.items():
            ids = np.where(y == lbl)[0]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(num_collaborators)]

class PathologicalSkewLabelsSplitter(NumPyDataSplitter):
    def __init__(self, shards_per_client: int=2, seed=0):
        self.seed = seed
        self.shards_per_client = shards_per_client
        
    def split(self, data, num_collaborators): 
        sorted_ids = np.argsort(data)
        n_shards = int(self.shards_per_client * num_collaborators)
        shard_size = int(np.ceil(len(data) / n_shards))
        assignments = np.zeros(data.shape[0])
        perm = permutation(n_shards)
        j = 0
        for i in range(num_collaborators):
            for _ in range(self.shards_per_client):
                left = perm[j] * shard_size
                right = min((perm[j]+1) * shard_size, len(data))
                assignments[sorted_ids[left:right]] = i
                j += 1
        return [np.where(assignments == i)[0] for i in range(num_collaborators)]  

class CovariateShiftSplitter2D(NumPyDataSplitter):
    def __init__(self, X: np.ndarray, modes: int=2, seed=0):
        self.seed = seed
        self.modes = modes
        self.X = X
        
    def split(self, X, y, num_collaborators): 
        assert 2 <= self.modes <= num_collaborators, "modes must be >= 2 and <= n"

        ids_mode = [[] for _ in range(self.modes)]
        for lbl in set(y):
            ids = np.where(y == lbl)[0]
            nsamples, nw, nh = self.X.shape
            d2_X = self.X.reshape((nsamples,nw*nh)) 
            X_pca = PCA(n_components=2).fit_transform(d2_X[ids])
            quantiles = mquantiles(X_pca[:, 0], prob=np.linspace(0, 1, num=self.modes+1)[1:-1])

            y_ = np.zeros(y[ids].shape)
            for i, q in enumerate(quantiles):
                if i == 0: continue
                id_pos = np.where((quantiles[i-1] < X_pca[:, 0]) & (X_pca[:, 0] <= quantiles[i]))[0]
                y_[id_pos] = i
            y_[np.where(X_pca[:, 0] > quantiles[-1])[0]] = self.modes-1

            for m in range(self.modes):
                ids_mode[m].extend(ids[np.where(y_ == m)[0]])

        ass_mode = (list(range(self.modes)) * int(np.ceil(num_collaborators/self.modes)))[:num_collaborators]
        shuffle(ass_mode)
        mode_map = {m:[u for u, mu in enumerate(ass_mode) if mu == m] for m in range(self.modes)}
        assignment = np.zeros(y.shape[0])
        for mode, users in mode_map.items():
            ids = ids_mode[mode]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(num_collaborators)] 


class CovariateShiftSplitter3D(NumPyDataSplitter):
    def __init__(self, X: np.ndarray, modes: int=2, seed=0):
        self.seed = seed
        self.modes = modes
        self.X = X
        
    def split(self, X, y, num_collaborators): 
        assert 2 <= self.modes <= num_collaborators, "modes must be >= 2 and <= n"

        ids_mode = [[] for _ in range(self.modes)]
        for lbl in set(y):
            ids = np.where(y == lbl)[0]
            nsamples, nw, nh, nc = self.X.shape
            d2_X = self.X.reshape((nsamples,nw*nh*nc)) 
            X_pca = PCA(n_components=2).fit_transform(d2_X[ids])
            quantiles = mquantiles(X_pca[:, 0], prob=np.linspace(0, 1, num=self.modes+1)[1:-1])

            y_ = np.zeros(y[ids].shape)
            for i, q in enumerate(quantiles):
                if i == 0: continue
                id_pos = np.where((quantiles[i-1] < X_pca[:, 0]) & (X_pca[:, 0] <= quantiles[i]))[0]
                y_[id_pos] = i
            y_[np.where(X_pca[:, 0] > quantiles[-1])[0]] = self.modes-1

            for m in range(self.modes):
                ids_mode[m].extend(ids[np.where(y_ == m)[0]])

        ass_mode = (list(range(self.modes)) * int(np.ceil(num_collaborators/self.modes)))[:num_collaborators]
        shuffle(ass_mode)
        mode_map = {m:[u for u, mu in enumerate(ass_mode) if mu == m] for m in range(self.modes)}
        assignment = np.zeros(y.shape[0])
        for mode, users in mode_map.items():
            ids = ids_mode[mode]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(num_collaborators)] 


class RandomNumPyDataSplitter(NumPyDataSplitter):
    """Splits the data randomly."""

    def __init__(self, shuffle=True, seed=0):
        """Initialize.

        Args:
            shuffle(bool): Flag determining whether to shuffle the dataset before splitting.
            seed(int): Random numbers generator seed.
                For different splits on envoys, try setting different values for this parameter
                on each shard descriptor.
        """
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        idx = range(len(data))
        if self.shuffle:
            idx = np.random.permutation(idx)
        random_idx = np.sort(np.random.choice(len(data), num_collaborators - 1, replace=False))

        return np.split(idx, random_idx)


class LogNormalNumPyDataSplitter(NumPyDataSplitter):
    """Unbalanced (LogNormal) dataset split.

    This split assumes only several classes are assigned to each collaborator.
    Firstly, it assigns classes_per_col * min_samples_per_class items of dataset
    to each collaborator so all of collaborators will have some data after the split.
    Then, it generates positive integer numbers by log-normal (power) law.
    These numbers correspond to numbers of dataset items picked each time from dataset
    and assigned to a collaborator.
    Generation is repeated for each class assigned to a collaborator.
    This is a parametrized version of non-i.i.d. data split in FedProx algorithm.
    Origin source: https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py#L30

    NOTE: This split always drops out some part of the dataset!
    Non-deterministic behavior selects only random subpart of class items.
    """

    def __init__(self, mu,
                 sigma,
                 num_classes,
                 classes_per_col,
                 min_samples_per_class,
                 seed=0):
        """Initialize the generator.

        Args:
            mu(float): Distribution hyperparameter.
            sigma(float): Distribution hyperparameter.
            classes_per_col(int): Number of classes assigned to each collaborator.
            min_samples_per_class(int): Minimum number of collaborator samples of each class.
            seed(int): Random numbers generator seed.
                For different splits on envoys, try setting different values for this parameter
                on each shard descriptor.
        """
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes
        self.classes_per_col = classes_per_col
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data.

        Args:
            data(np.ndarray): numpy-like label array.
            num_collaborators(int): number of collaborators to split data across.
                Should be divisible by number of classes in ``data``.
        """
        np.random.seed(self.seed)
        idx = [[] for _ in range(num_collaborators)]
        samples_per_col = self.classes_per_col * self.min_samples_per_class
        for col in range(num_collaborators):
            for c in range(self.classes_per_col):
                label = (col + c) % self.num_classes
                label_idx = np.nonzero(data == label)[0]
                slice_start = col // self.num_classes * samples_per_col
                slice_start += self.min_samples_per_class * c
                slice_end = slice_start + self.min_samples_per_class
                print(f'Assigning {slice_start}:{slice_end} of class {label} to {col} col...')
                idx[col] += list(label_idx[slice_start:slice_end])
        if any([len(i) != samples_per_col for i in idx]):
            raise SystemError(f'''All collaborators should have {samples_per_col} elements
but distribution is {[len(i) for i in idx]}''')

        props_shape = (
            self.num_classes,
            num_collaborators // self.num_classes,
            self.classes_per_col
        )
        props = np.random.lognormal(self.mu, self.sigma, props_shape)
        num_samples_per_class = [[[get_label_count(data, label) - self.min_samples_per_class]]
                                 for label in range(self.num_classes)]
        num_samples_per_class = np.array(num_samples_per_class)
        props = num_samples_per_class * props / np.sum(props, (1, 2), keepdims=True)
        for col in trange(num_collaborators):
            for j in range(self.classes_per_col):
                label = (col + j) % self.num_classes
                num_samples = int(props[label, col // self.num_classes, j])

                print(f'Trying to append {num_samples} samples of {label} class to {col} col...')
                slice_start = np.count_nonzero(data[np.hstack(idx)] == label)
                slice_end = slice_start + num_samples
                label_count = get_label_count(data, label)
                if slice_end < label_count:
                    label_subset = np.nonzero(data == (col + j) % self.num_classes)[0]
                    idx_to_append = label_subset[slice_start:slice_end]
                    idx[col] = np.append(idx[col], idx_to_append)
                else:
                    print(f'Index {slice_end} is out of bounds '
                          f'of array of length {label_count}. Skipping...')
        print(f'Split result: {[len(i) for i in idx]}.')
        return idx


class DirichletNumPyDataSplitter(NumPyDataSplitter):
    """Numpy splitter according to dirichlet distribution.

    Generates the random sample of integer numbers from dirichlet distribution
    until minimum subset length exceeds the specified threshold.
    This behavior is a parametrized version of non-i.i.d. split in FedMA algorithm.
    Origin source: https://github.com/IBM/FedMA/blob/master/utils.py#L96
    """

    def __init__(self, alpha=0.5, min_samples_per_col=10, seed=0):
        """Initialize.

        Args:
            alpha(float): Dirichlet distribution parameter.
            min_samples_per_col(int): Minimal amount of samples per collaborator.
            seed(int): Random numbers generator seed.
                For different splits on envoys, try setting different values for this parameter
                on each shard descriptor.
        """
        self.alpha = alpha
        self.min_samples_per_col = min_samples_per_col
        self.seed = seed

    def split(self, data, num_collaborators):
        """Split the data."""
        np.random.seed(self.seed)
        classes = len(np.unique(data))
        min_size = 0

        n = len(data)
        while min_size < self.min_samples_per_col:
            idx_batch = [[] for _ in range(num_collaborators)]
            for k in range(classes):
                idx_k = np.where(data == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, num_collaborators))
                proportions = [p * (len(idx_j) < n / num_collaborators)
                               for p, idx_j in zip(proportions, idx_batch)]
                proportions = np.array(proportions)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_splitted = np.split(idx_k, proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_splitted)]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        return idx_batch
