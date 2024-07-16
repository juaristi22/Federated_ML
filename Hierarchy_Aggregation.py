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

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
class Client(FM.FashionMNISTModel):
    def __init__(self, input_shape, hidden_units, output_shape,
                 epochs=None, data=None, learning_rate=0.0001, device=device):
        super().__init__(input_shape, hidden_units, output_shape)
        self.lr = learning_rate
        self.optimizer = None
        self.epochs = epochs
        self.data = data
        self.parent = None
        self.name = None

    def train_step(self, data_loader, loss_fn, optimizer, device=device):
        for epoch in range(self.epochs):
            train_loss, train_acc = 0, 0

            self.train()
            num_steps = 0

            for batch, (X, y) in enumerate(data_loader):
                X, y = X.to(device), y.to(device)
                y_pred = self.forward(X)
                loss = loss_fn(y_pred, y)
                train_loss += loss.item()
                train_acc += FM.accuracy_fn(target=y, preds=y_pred.argmax(dim=1)).item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                num_steps += 1

            train_loss /= num_steps
            train_acc /= num_steps
            train_acc *= 100

        # print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

        return train_loss, train_acc

    def test_step(self, data_loader, loss_fn, device=device):
        test_loss, test_acc = 0, 0
        self.eval()
        num_steps = 0

        with ((torch.inference_mode())):
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                test_pred = self.forward(X)
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += FM.accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
                num_steps += 1

            test_loss /= num_steps
            test_acc /= num_steps
            test_acc *= 100

        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

        return test_loss, test_acc
    def named(self, n):
        self.name = (f"{self.__class__.__name__}_{n}")
    def __str__(self):
        return self.name

class Aggregator(FM.FashionMNISTModel):
    def __init__(self, input_shape, hidden_units, output_shape, device=device):
        super().__init__(input_shape, hidden_units, output_shape)
        self.parent = None
        self.children_nodes = []
        self.name = None

    def add_child(self, child):
        self.children.append(child)
    def test_step(self, data_loader, loss_fn, device=device):
        test_loss, test_acc = 0, 0
        self.eval()
        num_steps = 0

        with ((torch.inference_mode())):
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                test_pred = self.forward(X)
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += FM.accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
                num_steps += 1

            test_loss /= num_steps
            test_acc /= num_steps
            test_acc *= 100

        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

        return test_loss, test_acc

    def average(self, clients_params):
        with torch.no_grad():
            averaged_params = clients_params.popitem()[1]
            for parameter in averaged_params:
                for model_state in clients_params:
                    parameter_value = clients_params[model_state][parameter]
                    parameter_value += averaged_params[parameter]
                    parameter_value = (1 / 2) * torch.clone(parameter_value)
                    averaged_params[parameter] = parameter_value
        return averaged_params

    def named(self, n):
        self.name = (f"{self.__class__.__name__}_{n}")

    def __str__(self):
        return self.name

def initialize_models(
    NUM_MODELS, device=device, epochs=2, lr=0.0001):
    input_shape = 1
    hidden_units = 10
    output_shape = 10

    # create NUM_MODELS client model instances
    def create_local_models(num_models, input_shape, hidden_units, output_shape):
        clients_list = []
        naming_dict = dict()
        for i in range(num_models):
            instance = Client(
                        input_shape=input_shape,
                        hidden_units=hidden_units,
                        output_shape=output_shape,
                        learning_rate=lr,
                        epochs=epochs).to(device)
            instance.named(i)
            naming_dict[instance.name] = instance
            instance.optimizer = torch.optim.SGD(params=instance.parameters(), lr=instance.lr)
            clients_list.append(instance)
        return clients_list, naming_dict

    local_models_list, naming_dict = create_local_models(
        num_models=NUM_MODELS,
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape,
    )
    return local_models_list, naming_dict


BATCH_SIZE = 32
NUM_MODELS = 3
equal_sizes = True
NUM_ROUNDS = 3
BRANCH_FACTOR = 2

general_trainloader = DataLoader(
    dataset=FM.train_data, batch_size=BATCH_SIZE, shuffle=True)

general_testloader = DataLoader(
    dataset=FM.test_data, batch_size=BATCH_SIZE, shuffle=True)

# split training data for the local models
local_trainloader, split_proportions = FM.split_data(
    data=FM.train_data,
    n_splits=NUM_MODELS,
    batch_size=BATCH_SIZE,
    equal_sizes=equal_sizes)

local_models_list, naming_dict = initialize_models(NUM_MODELS)

def record_experiments(
    model,
    num_clients,
    split_proportions,
    n_rounds,
    branching_factor,
    client_results,
    aggregator_results):

    results_dict = {
        "model_name": "HierarchicalBranchingFactor",
        "learning_rate": model.lr,
        "num_clients": num_clients,
        "data_split_proportions": split_proportions,
        "n_rounds": n_rounds,
        "max_branching_factor": branching_factor,
        "client_results": client_results,
        "aggregator_results": aggregator_results,
    }

    experiment = 0

    while os.path.exists(
        os.path.join(
            os.getcwd(),
            "Hierarchy_results/experiment_"
            + str(experiment)
            + "_HierarchicalBranchingFactor"
            + "_lr"
            + str(model.lr)
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_nrounds_"
            + str(n_rounds)
            + ".json",
        )
    ):
        experiment += 1

    with open(
        os.path.join(
            os.getcwd(),
            "Hierarchy_results/experiment_"
            + str(experiment)
            + "_HierarchicalBranchingFactor"
            + "_lr"
            + str(model.lr)
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_nrounds_"
            + str(n_rounds)
            + ".json",
        ),
        "w+",
    ) as f:
        results_json = json.dump(results_dict, f)

    filename = os.path.join(
            os.getcwd(),
            "Hierarchy_results/experiment_"
            + str(experiment)
            + "_HierarchicalBranchingFactor"
            + "_lr"
            + str(model.lr)
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_nrounds_"
            + str(n_rounds)
            + ".png",
        )

    FM.plot_loss_curves(aggregator_results["Global_Model"], filename=filename)

    return results_json

def run_local_models(model, train_data, test_data, client_results, loss_fn):
    train_loss, train_acc = model.train_step(data_loader=train_data, loss_fn=loss_fn,
                                                   optimizer=model.optimizer)
    test_loss, test_acc = model.test_step(data_loader=test_data, loss_fn=loss_fn)

    client_results[model.name]["train_loss"].append(train_loss)
    client_results[model.name]["train_acc"].append(train_acc)
    client_results[model.name]["test_loss"].append(test_loss)
    client_results[model.name]["test_acc"].append(test_acc)

    return client_results

def evaluate_aggregator(model, test_data, agg_results, loss_fn):
    test_loss, test_acc = model.test_step(data_loader=test_data, loss_fn=loss_fn)
    agg_results[model.name]["test_loss"].append(test_loss)
    agg_results[model.name]["test_acc"].append(test_acc)
    return agg_results

def create_hierarchy(local_models_list, naming_dict, NUM_ROUNDS, split_proportions,
                     local_trainloader, general_testloader,
                     device=device, branch_f=BRANCH_FACTOR):
    loss_fn = nn.CrossEntropyLoss()
    client_results = {}
    for i in local_models_list:
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        client_results[i.name] = results
    aggregator_results = {}
    results = {"test_loss": [], "test_acc": []}
    random.shuffle(local_models_list)

    for round in range(NUM_ROUNDS):
        print(f"Round: {round}:")
        client_models = copy.copy(local_models_list)
        aggregators = []
        iteration = 0
        while client_models:
            model_num = 0
            if len(client_models) > (branch_f - 1):
                trained_models = []
                #choose two random models to aggregate
                for i in range(branch_f):
                    client = client_models.pop(0)

                    #train and test client models
                    print(f"\nTraining client model: {client.name} \n--------------")
                    client_results = run_local_models(model=client,
                                                    train_data=local_trainloader[model_num],
                                                    test_data=general_testloader,
                                                    client_results=client_results,
                                                    loss_fn=loss_fn)
                    trained_models.append(client)
                    model_num += 1

                if round == 0:
                    #create an aggregator for the trained models
                    agg = Aggregator(input_shape=1, hidden_units=10, output_shape=10).to(device)
                    agg.named(iteration)
                    naming_dict[agg.name] = agg
                    aggregator_results[agg.name] = copy.deepcopy(results)
                    iteration += 1

                    #assign hierarchy relationships
                    for client in trained_models:
                        client.parent = agg.name
                        agg.children_nodes.append(client.name)
                else:
                    agg_name = client.parent
                    agg = naming_dict[agg_name]

                #average the client model parameters
                clients_params = dict()
                for client in trained_models:
                    clients_params[client.name] = client.state_dict()
                averaged_params = agg.average(clients_params)
                agg.load_state_dict(averaged_params)
                aggregators.append(agg)

            else:
                trained_models = []
                #choose the remaining client model
                for i in range(len(client_models)):
                    client = client_models.pop(0)

                    #train and test client model
                    print(f"Training client model: {client.name} \n--------------")
                    client_results = run_local_models(model=client,
                                                      train_data=local_trainloader[model_num],
                                                      test_data=general_testloader,
                                                      client_results=client_results,
                                                      loss_fn=loss_fn)
                    trained_models.append(client)
                    model_num += 1

                if round == 0:
                    agg = Aggregator(input_shape=1, hidden_units=10, output_shape=10).to(device)
                    agg.named(iteration)
                    naming_dict[agg.name] = agg
                    aggregator_results[agg.name] = copy.deepcopy(results)
                    iteration += 1

                    #assign hierarchy relationships
                    for client in trained_models:
                        client.parent = agg.name
                        agg.children_nodes.append(client.name)
                else:
                    agg_name = client.parent
                    agg = naming_dict[agg_name]

                if len(trained_models) > 1:
                    clients_params = dict()
                    for client in trained_models:
                        clients_params[client.name] = client.state_dict()
                    averaged_params = agg.average(clients_params)
                    agg.load_state_dict(averaged_params)
                else:
                    agg.load_state_dict(trained_models[0].state_dict())
                aggregators.append(agg)

        print("\nCreating aggregator hierarchies \n-----------")
        while aggregators:
            if len(aggregators) > (branch_f - 1):
                tested_aggs = []
                #access aggregators
                for i in range(branch_f):
                    agg = aggregators.pop(0)

                    #evaluate their performance
                    aggregator_results = evaluate_aggregator(model=agg,
                                                             test_data=general_testloader,
                                                             agg_results=aggregator_results,
                                                             loss_fn=loss_fn)
                    tested_aggs.append(agg)

                if round == 0:
                    new_agg = Aggregator(input_shape=1, hidden_units=10, output_shape=10).to(device)
                    new_agg.named(iteration)
                    naming_dict[new_agg.name] = new_agg
                    aggregator_results[new_agg.name] = copy.deepcopy(results)
                    iteration += 1

                    # assign hierarchy relationships
                    for agg in tested_aggs:
                        agg.parent = new_agg.name
                        new_agg.children_nodes.append(agg.name)
                else:
                    new_agg_name = agg.parent
                    new_agg = naming_dict[new_agg_name]

                #average evaluated aggregators' parameters
                aggs_params = dict()
                for agg in tested_aggs:
                    aggs_params[agg.name] = agg.state_dict()
                averaged_params = new_agg.average(aggs_params)
                new_agg.load_state_dict(averaged_params)
                aggregators.append(new_agg)

            else:
                if len(aggregators) > 1:
                    tested_aggs = []
                    # access aggregators
                    for i in range(len(aggregators)):
                        agg = aggregators.pop(0)

                        # evaluate their performance
                        aggregator_results = evaluate_aggregator(model=agg,
                                                                 test_data=general_testloader,
                                                                 agg_results=aggregator_results,
                                                                 loss_fn=loss_fn)
                        tested_aggs.append(agg)

                    if round == 0:
                        new_agg = Aggregator(input_shape=1, hidden_units=10, output_shape=10).to(device)
                        new_agg.named(iteration)
                        naming_dict[new_agg.name] = new_agg
                        aggregator_results[new_agg.name] = copy.deepcopy(results)
                        iteration += 1

                        # assign hierarchy relationships
                        for agg in tested_aggs:
                            agg.parent = new_agg.name
                            new_agg.children_nodes.append(agg.name)
                    else:
                        new_agg_name = agg.parent
                        new_agg = naming_dict[new_agg_name]

                    # average evaluated aggregators' parameters
                    aggs_params = dict()
                    for agg in tested_aggs:
                        aggs_params[agg.name] = agg.state_dict()
                    averaged_params = new_agg.average(aggs_params)
                    new_agg.load_state_dict(averaged_params)
                    aggregators.append(new_agg)

                else:
                    #the last remaining aggregator contains our global model
                    global_model = aggregators.pop(0)
                    # evaluate global models' performance
                    aggregator_results = evaluate_aggregator(model=global_model,
                                                             test_data=general_testloader,
                                                             agg_results=aggregator_results,
                                                             loss_fn=loss_fn)

    global_results = aggregator_results[global_model.name]
    del aggregator_results[global_model.name]
    del naming_dict[global_model.name]
    global_model.name = "Global_Model"
    aggregator_results[global_model.name] = global_results
    naming_dict[global_model.name] = global_model

    record_experiments(
        model=client,
        num_clients=NUM_MODELS,
        split_proportions=split_proportions,
        n_rounds=NUM_ROUNDS,
        branching_factor=branch_f,
        client_results=client_results,
        aggregator_results=aggregator_results)

    return client_results, aggregator_results, naming_dict

if __name__ == "__main__":
    client_results, aggregator_results, naming_dict = create_hierarchy(local_models_list,
                                                                       split_proportions=split_proportions,
                                                                       local_trainloader=local_trainloader,
                                                                       general_testloader=general_testloader,
                                                                       naming_dict=naming_dict,
                                                                       NUM_ROUNDS=NUM_ROUNDS)