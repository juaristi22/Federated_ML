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

def logger_setup(filename):
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_filename = os.path.join(filename, "FL_test_out.txt")
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler(sys.stdout)

    # log format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def experiment_configs(max_n_models, max_bf):
    """
    Configures all possible combinations of
        the desired hyperparameters to be tested

    Parameters
    ----------
    max_n_models: int, maximum number of client models to test for
    max_bf: int, maximum branching factor to test for

    Returns
    -------
    configurations: list[dicts], all configurations with
        the respective values of each hyperparameter
    config_descriptions: list[str], descriptions of all produced configurations
    """
    NUM_MODELS = [i for i in range(2, max_n_models+1)]
    BRANCHING_FACTOR = [i for i in range(2, max_bf+1)]
    equal_data_dist = [True, False]

    configurations = []
    config_descriptions = []
    for bf in BRANCHING_FACTOR:
        for n_models in NUM_MODELS:
            #for data_dist in equal_data_dist:
            if n_models >= bf:
                configs_dict = {}
                configs_dict["n_models"] = n_models
                configs_dict["bf"] = bf
                configs_dict["data_dist"] = True
                configurations.append(configs_dict)
                config_descriptions.append(f"n_models_{n_models}_bf_{bf}_equal_data_dist{True}")

    return configurations, config_descriptions
def experiment_running(max_n_models, max_bf):
    """
    Runs an experiment for each hyperparameter
        configuration on Hierarchy_Aggregation.py

    Parameters
    ----------
    n_models: int, maximum number of client models to test for
    bf: int, maximum branching factor to test for
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    learning_rate = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 5
    ROUNDS = 10

    general_testloader = HA.general_testloader

    configurations, config_descriptions = experiment_configs(
                                            max_n_models=max_n_models,
                                            max_bf=max_bf)
    i = 0
    for configuration in tqdm(configurations):
        print(f"Running experiment {i} on configuration: {configurations[i]}")
        local_models_list, naming_dict = HA.initialize_models(
            NUM_MODELS=configuration["n_models"],
            epochs=EPOCHS,
            lr=learning_rate)
        local_trainloader, split_proportions = FM.split_data(
            data=FM.train_data,
            n_splits=configuration["n_models"],
            batch_size=BATCH_SIZE,
            equal_sizes=True)

        client_results, aggregator_results, naming_dict, genealogy, filename = HA.create_hierarchy(
                            local_models_list=local_models_list,
                            naming_dict=naming_dict,
                            local_trainloader=local_trainloader,
                            general_testloader=general_testloader,
                            NUM_ROUNDS=ROUNDS,
                            height=None,
                            split_proportions=split_proportions,
                            device=device,
                            branch_f=configuration["bf"],
                            experiment_config=config_descriptions[i])
        i += 1

    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Federated Model Hierarchy testing",
        description="Runs and experiments on a federated model to explore different hierarchical aggregations"
    )
    parser.add_argument(
        "--max_n_models",
        type=int,
        help="maximum number of models for the last experiment config"
    )
    parser.add_argument(
        "--max_bf",
        type=int,
        help="maximum branching factor for the last experiment config",
    )

    args = vars(parser.parse_args())
    max_n_models = args["max_n_models"]
    max_bf = args["max_bf"]
    experiment_running(max_n_models=32, max_bf=10)
    #filename = experiment_running(n_models=max_n_models, bf=max_bf)
    #logger_setup(filename)

