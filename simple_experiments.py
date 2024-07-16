import Federated_model as Fd
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchinfo import summary

import copy
from tqdm.auto import tqdm
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import json
import os
import argparse
def run_experiments(learning_rate, n_models, num_experiments):
    EPOCHS = 2
    ROUNDS = 2
    device = "mps"

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor())

    results_dict = dict()

    for local_models in range(1, n_models+1):
        experiment_results = 0
        for i in range(num_experiments):
            global_model = Fd.FashionMNISTModel(input_shape=1, hidden_units=10, output_shape=10).to(device)

            global_results = Fd.federate_model(global_model_instance=global_model,
                                               train_data=train_data,
                                               test_data=test_data,
                                               learning_rate=learning_rate,
                                               equal_sizes=True,
                                               NUM_MODELS=local_models,
                                               NUM_ROUNDS=ROUNDS,
                                               EPOCHS=EPOCHS)

            if experiment_results != 0:
                current_test_loss = global_results["test_loss"]
                current_test_acc = global_results["test_acc"]

                avg_test_loss = experiment_results["test_loss"]
                avg_test_acc = experiment_results["test_acc"]

                for i in range(len(avg_test_acc)):
                    avg_test_loss[i] += current_test_loss[i]
                    avg_test_loss[i] /= 2

                    avg_test_acc[i] += current_test_acc[i]
                    avg_test_acc[i] /= 2

                    experiment_results["test_loss"] = avg_test_loss
                    experiment_results["test_acc"] = avg_test_acc

            else:
                experiment_results = global_results

        results_dict[f"equal data dist, {local_models} clients"] = experiment_results

    for local_models in range(1, n_models+1):
        experiment_results = 0
        for i in range(num_experiments):
            global_model = Fd.FashionMNISTGlobal(input_shape=1, hidden_units=10, output_shape=10).to(device)

            global_results = Fd.federate_model(global_model_instance=global_model,
                                               train_data=train_data,
                                               test_data=test_data,
                                               learning_rate=learning_rate,
                                               equal_sizes=True,
                                               NUM_MODELS=local_models,
                                               NUM_ROUNDS=ROUNDS,
                                               EPOCHS=EPOCHS)

            if experiment_results != 0:
                current_test_loss = global_results["test_loss"]
                current_test_acc = global_results["test_acc"]

                avg_test_loss = experiment_results["test_loss"]
                avg_test_acc = experiment_results["test_acc"]

                for i in range(len(avg_test_acc)):
                    avg_test_loss[i] += current_test_loss[i]
                    avg_test_loss[i] /= 2

                    avg_test_acc[i] += current_test_acc[i]
                    avg_test_acc[i] /= 2

                    experiment_results["test_loss"] = avg_test_loss
                    experiment_results["test_acc"] = avg_test_acc

            else:
                experiment_results = global_results

        results_dict[f"random data dist, {local_models} clients"] = experiment_results

        #Fd.plot_loss_curves(experiment_results)
    print(results_dict)

run_experiments(learning_rate=0.0001, n_models=2, num_experiments=2)
