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
NUM_MODELS = 5
MAX_EPOCHS = 10

def map_epochs_to_data(NUM_MODELS, MAX_EPOCHS):
    """
    Randomly distributes data across all client models and
        calculates the number of epochs that each client model
        should run given its data load

    Parameters
    ----------
    NUM_MODELS: int, number of client models
    MAX_EPOCHS: int, number of epochs that the model with largest
        data load should run for
    """
    BATCH_SIZE = 256
    local_models_list, naming_dict = HA.initialize_models(NUM_MODELS)

    local_trainloader, split_proportions = FM.split_data(
        data=FM.train_data,
        n_splits=NUM_MODELS,
        batch_size=BATCH_SIZE,
        equal_sizes=False)

    sorted_split_proportions = copy.deepcopy(split_proportions)
    sorted_split_proportions.sort()
    highest_dataload = sorted_split_proportions[-1]

    data_prop_dict = {}
    for i in range(len(local_trainloader)):
        model = local_models_list[i]
        data_prop_dict[split_proportions[i]] = model
        model.data = local_trainloader[i]
        current_dataload = split_proportions[i]
        if current_dataload == highest_dataload:
            model.epochs = MAX_EPOCHS
        else:
            proportion = current_dataload / highest_dataload
            if proportion >= 0.75:
                model.epochs = MAX_EPOCHS
            elif proportion >= 0.5 and proportion < 0.75:
                EPOCHS = MAX_EPOCHS // (4/3)
                model.epochs = EPOCHS
            elif proportion >= 0.25 and proportion < 0.5:
                EPOCHS = MAX_EPOCHS // (4/2)
                model.epochs = EPOCHS
            elif proportion < 0.25:
                EPOCHS = MAX_EPOCHS // (4/1)
                model.epochs = EPOCHS

    return local_models_list, naming_dict, data_prop_dict, sorted_split_proportions

local_models_list, naming_dict, data_prop_dict, sorted_split_proportions = map_epochs_to_data(10, 10)

BATCH_SIZE = 256

general_testloader = DataLoader(
    dataset=FM.test_data, batch_size=BATCH_SIZE, shuffle=True)

def create_hierarchy(local_models_list, naming_dict, data_prop_dict, sorted_split_proportions, general_testloader, device=device):
    NUM_ROUNDS = 10

    loss_fn = nn.CrossEntropyLoss()
    client_results = {}
    for i in local_models_list:
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        client_results[i.name] = results
    aggregator_results = {}
    results = {"test_loss": [], "test_acc": []}

    ### TO DO
