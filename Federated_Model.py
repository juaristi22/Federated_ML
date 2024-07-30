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
import logging

train_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=ToTensor())

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class_names = train_data.classes
class_to_idx = train_data.class_to_idx
class_to_idx


class CNNModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 14 * 14, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.classifier(x)
        return x

class NewModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

global_model = CNNModel(input_shape=3, hidden_units=10, output_shape=10).to(
    device)

accuracy_fn = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

def train_step(model, data_loader, loss_fn, optimizer, device):
    """
    Performs the training step for the machine learning model

    Parameters
    ----------
    model: CNNModel instance
    data_loader: dataloader object, train data
    loss_fn: nn.CrossEntropyLoss instance, loss function
    optimizer: torch,optim,SGD instance, learning optimizer
    device: str, device in which to train

    Returns
    -------
    train_loss: float, average training loss for the dataloader at hand
    train_acc: float, average training accuracy for the dataloader at hand
    """
    train_loss, train_acc = 0, 0
    model.train()
    num_steps = 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        #print(f"train loss: {loss}")
        train_loss += loss.item()
        train_acc += accuracy_fn(target=y, preds=y_pred.argmax(dim=1)).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        num_steps += 1

    train_loss /= num_steps
    train_acc /= num_steps
    train_acc *= 100

    # print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

    return train_loss, train_acc

def test_step(model, data_loader, loss_fn, device):
    """
    Performs the testing step for the machine learning model

    Parameters
    ----------
    model: CNNModel instance
    data_loader: dataloader object, test data
    loss_fn: nn.CrossEntropyLoss instance, loss function
    device: str, device in which to train

    Returns
    -------
    test_loss: float, average testing loss for the dataloader at hand
    test_acc: float, average testing accuracy for the dataloader at hand
    """
    test_loss, test_acc = 0, 0
    model.eval()
    num_steps = 0

    with ((torch.inference_mode())):
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            # print(f"test loss: {loss}")
            test_loss += loss.item()
            test_acc += accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
            num_steps += 1

        test_loss /= num_steps
        test_acc /= num_steps
        test_acc *= 100

    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

    return test_loss, test_acc

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
        # print(f"Epoch: {epoch} \n--------")

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

def average(local_models_params):
    """
    Averages the parameters of given models to perform federation

    Parameters
    ----------
    local_models_params: list[tensor], parameters of all client models to be averaged

    Returns
    -------
    averaged_params: tensor, average of all models' parameters
    """
    with torch.no_grad():
        averaged_params = local_models_params[0]
        for parameter in averaged_params:
            for model_state in local_models_params:
                if model_state == local_models_params[0]:
                    continue
                else:
                    parameter_value = local_models_params[model_state][parameter]
                    parameter_value += averaged_params[parameter]
                    parameter_value = (1 / 2) * torch.clone(parameter_value)
                    averaged_params[parameter] = parameter_value
    return averaged_params

def print_train_time(start, end, device=None):
    """
    Prints the time taken to train the model

    Parameters
    ----------
    start: float, starting time
    end: float, ending time
    device: str, device in which the training was performed

    Returns
    -------
    total_time: float, total time taken to train the model
    """
    total_time = end - start
    return total_time

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

def plot_loss_curves(results, config, filename=None):
    """
    Plots the loss curves of a model's performance

    Parameters
    ----------
    results: dict, model's performance results
    filename: str, default:None, directory path on which to save figures
    """
    test_accuracy = results["test_acc"]
    test_loss = results["test_loss"]
    if len(results) > 2:
        loss = results["train_loss"]
        accuracy = results["train_acc"]
    else:
        loss = None
        accuracy = None

    rounds = range(len(results["test_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.grid()
    ax2.grid()
    if config:
        fig.suptitle(config)

    if loss and len(loss) > 0:
        ax1.plot(rounds, loss, label="train_loss", color="blue")
    ax1.plot(rounds, test_loss, label="test_loss", color="orange")
    ax1.set_title("Loss")
    ax1.set_xlabel("Rounds")
    ax1.legend()


    if accuracy and len(accuracy) > 0:
        ax2.plot(rounds, accuracy, label="train_accuracy", color="blue")
    ax2.plot(rounds, test_accuracy, label="test_accuracy", color="orange")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Rounds")
    ax2.legend()
    if filename:
        plt.savefig(fname=filename)
    else:
        plt.show()

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
            split = random.randrange(1, total_size)
            split_sizes.append(split)
            total_size -= split
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

    general_trainloader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
    )

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

    # print("Local model results:")
    for client, results in local_results.items():
        plot_loss_curves(results)
    # print("Global model results:")
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

    args = vars(parser.parse_args())
    #lr = args["learning_rate"]
    #num_models = args["num_models"]
    #rounds = args["num_rounds"]
    #epochs = args["num_epochs"]
    #data_sizing = args["equal_data_sizes"]
    #data_sizing = args["unequal_data_sizes"]

    #federate_model(
        #global_model_instance=global_model,
        #train_data=train_data,
        #test_data=test_data,
        #learning_rate=lr,
        #equal_sizes=data_sizing,
        #NUM_MODELS=num_models,
        #NUM_ROUNDS=rounds,
        #EPOCHS=epochs)

    federate_model(
            global_model_instance=global_model,
            train_data=train_data,
            test_data=test_data,
            learning_rate=0.000001,
            equal_sizes=True,
            NUM_MODELS=2,
            NUM_ROUNDS=1,
            EPOCHS=1)