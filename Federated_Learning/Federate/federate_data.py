import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, RandomSampler, Subset


train_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=ToTensor())

def split_data(data, n_splits, batch_size, equal_sizes):
    """
    Splits training data either uniformly or randomly across client models

    Parameters
    ----------
    data: dataset
    n_splits: int, number of client models in which data must be split
    batch_size: int, batch size
    equal_sizes: bool, whether data should be uniformly distributed or not

    Returns
    -------
    data_splits: list[dataloader objects], data that each model will train on
    split_sizes: list[int], amount of data that each model will train on
    """
    if equal_sizes:
        split_sizes = [len(data) // n_splits for _ in range(n_splits)]
    else:
        total_size = len(data)
        split_sizes = []
        for i in range(n_splits - 1):
            max_split_size = total_size - (n_splits - i - 1)
            split = random.randrange(1, max_split_size)
            print(f"split: {split}")
            split_sizes.append(split)
            total_size -= split
            print(f"data remaining: {total_size}")
        split_sizes.append(total_size)

    indices = list(range(len(data)))
    data_splits = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        subset_indices = indices[start_idx:end_idx]
        subset = Subset(data, subset_indices)
        data_loader = DataLoader(subset, batch_size=batch_size)
        data_splits.append(data_loader)
        start_idx = end_idx
    return data_splits, split_sizes