import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchinfo import summary
import torchmetrics

import copy
from collections import OrderedDict
from tqdm.auto import tqdm
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import json
import os
import argparse
import Federated_Model as FM
import Hierarchy_Aggregation as HA

def experiment_config():
    device = "gpu"
    learning_rate = 0.0001
    EPOCHS = 3
    ROUNDS = 25
    NUM_MODELS = [i for i in range(1, 11)]
    BRANCHING_FACTOR = [2, 3, 4, 5]
    equal_data_dist = [True, False]

    input_shape = 1
    hidden_units = 10
    output_shape = 10

    train_data =
    test_data =

    model =