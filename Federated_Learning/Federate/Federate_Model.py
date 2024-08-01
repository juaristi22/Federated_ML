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

from Federated_Learning.Learn.Model import CNNModel, NewModel
from Federated_Learning.Learn.training import train_step
from Federated_Learning.Learn.testing import test_step
from federate_data import split_data
from Federated_Learning.helper_functions import average, plot_loss_curves

train_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=ToTensor())

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class_names = train_data.classes
class_to_idx = train_data.class_to_idx
class_to_idx

global_model = CNNModel(input_shape=3, hidden_units=10, output_shape=10).to(
    device)

accuracy_fn = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

def run_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs):
    """
    Runs the training and testing steps for the machine learning model

    Parameters
    ----------
    model: CNNModel instance
    train_dataloader: dataloader object, train data
    test_dataloader: dataloader object, test data
    loss_fn: nn.CrossEntropyLoss instance, loss function
    optimizer: torch,optim,SGD instance, learning optimizer
    device: str, device in which to train
    epochs: int, number of epochs for which to run both steps

    Returns
    -------
    train_loss: float, average training loss for the dataloader at hand
    train_acc: float, average training accuracy for the dataloader at hand
    test_loss: float, average testing loss for the dataloader at hand
    test_acc: float, average testing accuracy for the dataloader at hand
    """
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

    return train_loss, train_acc, test_loss, test_acc

def record_experiments(
    global_model,
    learning_rate,
    num_local_models,
    split_proportions,
    n_rounds,
    n_epochs,
    train_time,
    global_results,
    local_results):
    """
    Saves configuration and model performance results to json file

    Parameters
    ----------
    global_model: CNNModel instance
    learning_rate: float, model's learning rate
    num_local_models: int, number of client models
    split_proportions: list[int], list containing the
        data distribution across client models
    n_rounds: int, number of rounds for which the federated model runs
    n_epochs: int, number of epochs for which each client model trains
    train_time: float, total time it took to run the federated model
    global_results: dict, results of the global model
    local_results, dict, results of each of the client models
    """

    results_dict = {
        "model_name": global_model.__class__.__name__,
        "learning_rate": learning_rate,
        "n_local_models": num_local_models,
        "data_split_proportions": split_proportions,
        "n_rounds": n_rounds,
        "n_epochs": n_epochs,
        "train_time": train_time,
        "global_results": global_results,
        "local_results": local_results,
    }

    # check there is no other file with that name to avoid overwriting
    experiment = 0
    while os.path.exists(
        os.path.join(
            os.getcwd(),
            "FM_results/experiment_"
            + str(experiment)
            + "_"
            + global_model.__class__.__name__
            + "_lr"
            + str(learning_rate)
            + "_localmodels_"
            + str(num_local_models)
            + "_nrounds_"
            + str(n_rounds)
            + ".json",
        )
    ):
        experiment += 1

    # save results
    with open(
        os.path.join(
            os.getcwd(),
            "FM_results/experiment_"
            + str(experiment)
            + "_"
            + global_model.__class__.__name__
            + "_lr"
            + str(learning_rate)
            + "_localmodels_"
            + str(num_local_models)
            + "_nrounds_"
            + str(n_rounds)
            + ".json",
        ),
        "w+",
    ) as f:
        json.dump(results_dict, f)

def federate_model(
    global_model_instance,
    train_data,
    test_data,
    learning_rate,
    equal_sizes,
    NUM_MODELS,
    NUM_ROUNDS,
    EPOCHS,
    device=device
):
    """
    Builds a global model through federated learning by averaging
        the parameters of trained client models

    Parameters
    ----------
    global_model_instance: CNNModel instance
    train_data: dataloader object, train data
    test_data: dataloader object, test data
    learning_rate: float, model's learning rate
    equal_sizes: bool, whether data should be
        uniformly distributed or not
    NUM_MODELS: int, number of client models
    NUM_ROUNDS:  int, number of rounds for
        which the global model is going to run
    EPOCHS: int, number of epochs for which
        each client model is going to train
    device: str, device on which to perform the computation

    Returns
    -------
    global_results: dict, results of the global model
    """

    BATCH_SIZE = 256
    loss_fn = nn.CrossEntropyLoss()

    input_shape = 3
    hidden_units = 10
    output_shape = 10

    # split training data for the local models
    local_trainloader, split_proportions = split_data(
        data=train_data,
        n_splits=NUM_MODELS,
        batch_size=BATCH_SIZE,
        equal_sizes=equal_sizes,
    )

    # to use in case we want all models to train on the same data
    general_trainloader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
    )

    # testing data
    general_testloader = DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, shuffle=True
    )

    # create NUM_SPLITS local model instances
    def create_local_models(num_models, input_shape, hidden_units, output_shape):
        """
        Instantiates the CNNModel to produce the
            relevant client models and their optimizers

        Parameters
        ----------
        num_models: int, number of client models
        input_shape: int, number of channels in input data
        hidden_units: int, CNNModel layer's hidden units
        output_shape: int, number of labels

        Returns
        -------
        local_models_dict: dict, dictionary containing each
            client model instance and its respective optimizer
        """
        local_models_dict = dict()
        for _ in range(num_models):
            instance = CNNModel(
                input_shape=input_shape,
                hidden_units=hidden_units,
                output_shape=output_shape,
            ).to(device)
            local_models_dict[instance] = torch.optim.SGD(
                params=instance.parameters(), lr=learning_rate
            )
        return local_models_dict

    local_models_dict = create_local_models(
        num_models=NUM_MODELS,
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape,
    )

    global_results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    local_results = {}
    for i in range(len(local_models_dict)):
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        local_results[i] = results

    time_start = timer()

    for round in range(NUM_ROUNDS):
        print(f"Round: {round}\n--------")
        models_params = dict()

        # train each of the local models
        iteration = 0
        for client, client_params in local_models_dict.items():
            print("\nTraining client model \n-------------")

            train_loss, train_acc, test_loss, test_acc = run_model(
                model=client,
                train_dataloader=local_trainloader[iteration],
                test_dataloader=general_testloader,
                optimizer=client_params,
                loss_fn=loss_fn,
                device=device,
                epochs=EPOCHS,
            )

            local_results[iteration]["train_loss"].append(train_loss)
            local_results[iteration]["train_acc"].append(train_acc)
            local_results[iteration]["test_loss"].append(test_loss)
            local_results[iteration]["test_acc"].append(test_acc)

            models_params[iteration] = client.state_dict()
            iteration += 1

        # average their parameters
        global_params = average(models_params)

        # test the global model with the averaged parameters
        global_model_instance.load_state_dict(global_params)

        print("\nTesting global model \n----------------")

        test_loss, test_acc = test_step(
            model=global_model_instance,
            data_loader=general_testloader,
            loss_fn=loss_fn,
            device=device,
        )

        global_results["test_loss"].append(test_loss)
        global_results["test_acc"].append(test_acc)

        # update parameters of the local models
        for client in local_models_dict.keys():
            client.load_state_dict(global_params)

    time_end = timer()
    total_time = print_train_time(start=time_start, end=time_end, device=device)

    record_experiments(
        global_model=global_model_instance,
        learning_rate=learning_rate,
        num_local_models=NUM_MODELS,
        split_proportions=split_proportions,
        n_rounds=NUM_ROUNDS,
        n_epochs=EPOCHS,
        train_time=total_time,
        global_results=global_results,
        local_results=local_results,
    )

    # print result figures
    for client, results in local_results.items():
        plot_loss_curves(results)
    plot_loss_curves(global_results)

    return global_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Federated TinyVGG Model training on CIFAR10",
        description="Runs and experiments on a federated model to explore data "
        "distribution and branching factor among other",
    )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate of the model's optimizer"
    )
    parser.add_argument(
        "--num_models",
        type=int,
        help="number of local models involved in the federating process",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        help="number of rounds that the global model will aggregate on",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="number of epochs that each local model will train for",
    )
    parser.add_argument(
        "--equal_data_sizes",
        action="store_true",
        help="choose if data should be split in equal " "sizes",
    )
    parser.add_argument(
        "--unequal_data_sizes",
        action="store_false",
        help="choose if data should be split randomly",
    )

    # command line parameters
    args = vars(parser.parse_args())
    lr = args["learning_rate"]
    num_models = args["num_models"]
    rounds = args["num_rounds"]
    epochs = args["num_epochs"]
    data_sizing = args["equal_data_sizes"]
    data_sizing = args["unequal_data_sizes"]

    federate_model(
        global_model_instance=global_model,
        train_data=train_data,
        test_data=test_data,
        learning_rate=lr,
        equal_sizes=data_sizing,
        NUM_MODELS=num_models,
        NUM_ROUNDS=rounds,
        EPOCHS=epochs)

    # in case of running pre-specified parameters
    federate_model(
            global_model_instance=global_model,
            train_data=train_data,
            test_data=test_data,
            learning_rate=0.00001,
            equal_sizes=True,
            NUM_MODELS=2,
            NUM_ROUNDS=1,
            EPOCHS=1)