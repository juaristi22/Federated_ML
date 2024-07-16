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

train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor())

device = "mps"

class_names = train_data.classes
class_to_idx = train_data.class_to_idx
class_to_idx


class FashionMNISTModel(nn.Module):
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


global_model = FashionMNISTModel(input_shape=1, hidden_units=10, output_shape=10).to(
    device
)
print(next(global_model.parameters()).device)

accuracy_fn = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

def train_step(model, data_loader, loss_fn, optimizer, device):
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


def run_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs
):
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
        results_json = json.dump(results_dict, f)

    return results_json

def plot_loss_curves(results, filename=None):
    test_accuracy = results["test_acc"]
    test_loss = results["test_loss"]
    if len(results) > 2:
        loss = results["train_loss"]
        accuracy = results["train_acc"]
    else:
        loss = None
        accuracy = None

    rounds = range(len(results["test_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)

    if loss and len(loss) > 0:
        plt.plot(rounds, loss, label="train_loss", color="blue")
    plt.plot(rounds, test_loss, label="test_loss", color="orange")
    plt.title("Loss")
    plt.xlabel("Rounds")
    plt.legend()

    plt.subplot(1, 2, 2)
    if accuracy and len(accuracy) > 0:
        plt.plot(rounds, accuracy, label="train_accuracy", color="blue")
    plt.plot(rounds, test_accuracy, label="test_accuracy", color="orange")
    plt.title("Accuracy")
    plt.xlabel("Rounds")
    plt.legend()
    if filename:
        plt.savefig(fname=filename)
    else:
        plt.show()


def split_data(data, n_splits, batch_size, equal_sizes):
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
    BATCH_SIZE = 32
    loss_fn = nn.CrossEntropyLoss()

    input_shape = 1
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
        local_models_dict = dict()
        for _ in range(num_models):
            instance = FashionMNISTModel(
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
            learning_rate=0.0001,
            equal_sizes=True,
            NUM_MODELS=2,
            NUM_ROUNDS=1,
            EPOCHS=1)