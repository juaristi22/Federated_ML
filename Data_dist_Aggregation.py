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
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
BATCH_SIZE = 256
NUM_MODELS = 5
MAX_EPOCHS = 10


local_models_list, naming_dict = HA.initialize_models(NUM_MODELS)

local_trainloader, split_proportions = FM.split_data(
    data=FM.train_data,
    n_splits=NUM_MODELS,
    batch_size=BATCH_SIZE,
    equal_sizes=False)

data_prop_dict = {}
for i in range(len(split_proportions)):
    data_prop_dict[split_proportions[i]] = local_trainloader[i]
sorted_split_proportions = sort(split_proportions)
highest_dataload = sorted_split_proportions[0]

for i in range(len(local_trainloader)):
    model = local_models_list[i]
    model.data = local_trainloader[i]
    current_dataload = split_proportions[i]
    if current_dataload == highest_dataload:
        model.epochs = MAX_EPOCHS
    else:
        pass