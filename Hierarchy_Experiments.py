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


def experiment_configs(max_n_models, max_bf):
    NUM_MODELS = [i for i in range(2, max_n_models+1)]
    BRANCHING_FACTOR = [i for i in range(2, max_bf+1)]
    equal_data_dist = [True, False]

    configs_dict = {}
    configurations = []
    config_descriptions = []
    for n_models in NUM_MODELS:
        for bf in BRANCHING_FACTOR:
            for data_dist in equal_data_dist:
                configs_dict["n_models"] = n_models
                configs_dict["bf"] = bf
                configs_dict["data_dist"] = data_dist
                configurations.append(configs_dict)
                config_descriptions.append(f"n_models_{n_models}_bf_{bf}_equal_data_dist{data_dist}")

    return configurations, config_descriptions
def experiment_running(max_n_models, max_bf):
    device = "mps"
    learning_rate = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 3
    ROUNDS = 2

    general_testloader = HA.general_testloader

    configurations, config_descriptions = experiment_configs(
                                            max_n_models=max_n_models,
                                            max_bf=max_bf)

    for configuration in configurations:
        i = 0
        local_models_list, naming_dict = HA.initialize_models(
            NUM_MODELS=configuration["n_models"],
            epochs=EPOCHS,
            lr=learning_rate)
        local_trainloader, split_proportions = FM.split_data(
            data=FM.train_data,
            n_splits=configuration["n_models"],
            batch_size=BATCH_SIZE,
            equal_sizes=configuration["data_dist"])

        HA.create_hierarchy(local_models_list=local_models_list,
                            naming_dict=naming_dict,
                            local_trainloader=local_trainloader,
                            general_testloader=general_testloader,
                            NUM_ROUNDS=ROUNDS,
                            split_proportions=split_proportions,
                            device=device,
                            branch_f=configuration["bf"])

        print(config_descriptions[i])
        i += 1

experiment_running(max_n_models=4, max_bf=3)