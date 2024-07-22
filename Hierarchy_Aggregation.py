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
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
class Client(FM.CNNModel):
    def __init__(self, input_shape, hidden_units, output_shape,
                 epochs=None, data=None, learning_rate=0.000001, device=device):
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

class Aggregator(FM.CNNModel):
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
    NUM_MODELS, device=device, epochs=5, lr=0.000001):
    input_shape = 3
    hidden_units = 10
    output_shape = 10
    """
    Initializes all client models keeping track of their names
    
    Parameters
    ----------
    NUM_MODELS: int, number of client models
    device: str, device computation will take place on
    epochs: int, number of epochs each client model will train for
    lr: float, models' learning rate
    
    Returns 
    -------
    local_models_list: list[Client instances], list of all client models
    naming_dict: dict, dictionary of the client models' names and their instances
    """

    # create NUM_MODELS client model instances
    def create_local_models(num_models, input_shape, hidden_units, output_shape):
        """
        Instantiates Client model instances

        Parameters
        ----------
        num_models: int, number of client models
        input_shape: int, number of channels in input data
        hidden_units: int, Client model layers' hidden units
        output_shape: int, number of labels

        Returns
        -------
        clients_list: list[Client objects], list with all client model instances
        naming_dict: dict, dictionary of all client models' names and their instances
        """
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


BATCH_SIZE = 256
NUM_MODELS = 2
equal_sizes = True
NUM_ROUNDS = 10
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
    height,
    client_results,
    aggregator_results,
    experiment_config=None):
    """
    Saves configuration and model performance results to json file,
        and loss curve figures to image

    Parameters
    ----------
    model: Client instance
    num_clients: int, number of client models
    split_proportions: list[int], list containing the
        data distribution across client models
    n_rounds: int, number of rounds for which the federated model runs
    branching_factor: int, maximum branching factor per node
    height: int, maximum height of the hierarchy tree
    client_results: dict, results of the client models
    aggregator_results, dict, results of each of the aggregator models
    experiment_config: st, default:None, configuration of the current experiment run if any

    Returns
    -------
    filename: str, directory of the saved results
    """

    results_dict = {
        "model_name": "HierarchicalBranchingFactor",
        "learning_rate": model.lr,
        "num_clients": num_clients,
        "data_split_proportions": split_proportions,
        "n_rounds": n_rounds,
        "max_branching_factor": branching_factor,
        "height": height,
        "client_results": client_results,
        "aggregator_results": aggregator_results,
        "experiment_config": experiment_config
    }

    experiment = 0

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'Hierarchy_results')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    while os.path.exists(
        os.path.join(
            os.getcwd(),
            "Hierarchy_results/experiment_"
            + str(experiment)
            + "_HierarchicalBranchingFactor"
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_specifiedheight_"
            + str(height)
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
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_specifiedheight_"
            + str(height)
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
            + "_clientmodels_"
            + str(num_clients)
            + "_branchingfactor_"
            + str(branching_factor)
            + "_specifiedheight_"
            + str(height)
            + ".png",
        )

    FM.plot_loss_curves(aggregator_results["Global_Model"], filename=filename, config=experiment_config)

    return filename

def run_local_models(model, train_data, test_data, client_results, loss_fn):
    """
    Performs training and testing steps on all local models

    Parameters
    ----------
    model: Client instance
    train_data: dataloader instance, training data
    test_data: dataloader instance, testing data
    client_results: dict, performance results of client models
    loss_fn: nn.CrossEntropyLoss instance, loss function

    Returns
    -------
    client_results: dict, performance results of all client models
    """
    train_loss, train_acc = model.train_step(data_loader=train_data, loss_fn=loss_fn,
                                                   optimizer=model.optimizer)
    test_loss, test_acc = model.test_step(data_loader=test_data, loss_fn=loss_fn)

    client_results[model.name]["train_loss"].append(train_loss)
    client_results[model.name]["train_acc"].append(train_acc)
    client_results[model.name]["test_loss"].append(test_loss)
    client_results[model.name]["test_acc"].append(test_acc)

    return client_results

def evaluate_aggregator(model, test_data, agg_results, loss_fn):
    """
    Performs the testing step on all aggregator models

    Parameters
    ----------
    model: Aggregator instance
    test_data: dataloader instance, testing data
    agg_results: dict, aggregator performance results
    loss_fn: nn.CrossEntropyLoss instance, loss function

    Returns
    -------
    agg_results: dict, all aggregator performance results
    """
    test_loss, test_acc = model.test_step(data_loader=test_data, loss_fn=loss_fn)
    agg_results[model.name]["test_loss"].append(test_loss)
    agg_results[model.name]["test_acc"].append(test_acc)
    return agg_results

def compute_bf(n_leaves, height):
    """
    Calculate the maximum branching factor of the hierarchical
        aggregation based on the desired depth of the tree

    Parameters
    ----------
    n_leafs: int, number of leaves in the tree
        ie. number of client models
    height: int, desired height of the tree

    Returns
    -------
    bf: int, maximum branching factor
    """
    bf = n_leaves * (1/height)
    bf = round(bf)
    return bf
def create_hierarchy(local_models_list, naming_dict, NUM_ROUNDS, split_proportions,
                     local_trainloader, general_testloader,
                     device=device, branch_f=None, height=None, experiment_config=None):

    """
    Creates a federated machine learning model hierarchy with intermediate aggregator nodes
        based on either the desired maximum branching factor or the desired height of the tree

    Parameters
    ----------
    local_models_list: list[Client instances], all client instances to include in the hierarchy
    naming_dict: dict, name of all client nodes' names and their respective instances
    NUM_ROUNDS: int, number of rounds for which all nodes will run
    split_proportions: list[int], distribution of training data across client nodes
    local_trainloader: dataloader instance, training data
    general_testloader: dataloader instance, testing data
    device: str, device on which to perform computation
    branch_f: int, default:None, desired maximum branching factor
    height: int, default:None, desired height of the tree
    experiment_config: st, default:None, configuration of the current experiment run if any

    Returns
    -------
    client_results: dict, performance results of all client models
    aggregator_results: dict, performance results of all aggregator models
    naming_dict: dict, dictionary of all nodes' names and their respective model instances
    genealogy: list[model instances], list containing all nodes to retrieve their genealogy
    filename: str, directory of the saved results
    """

    if (branch_f == None) and (height == None):
        raise ValueError("Please choose either a branching factor or "
                         "height hyperparameter to create the aggregation hierarchy")
    elif (branch_f != None) and (height != None):
        raise ValueError("Please choose only either a branching factor or "
                         "height hyperparameter to create the aggregation hierarchy, "
                         "setting the other to None")
    elif (branch_f != None) and (height == None):
        pass
    elif (branch_f == None) and (height != None):
        branch_f = compute_bf(len(local_models_list), height)

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
        genealogy = []
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
                    agg = Aggregator(input_shape=3, hidden_units=10, output_shape=10).to(device)
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
                    agg = Aggregator(input_shape=3, hidden_units=10, output_shape=10).to(device)
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
                    genealogy.append(agg)

                    #evaluate their performance
                    aggregator_results = evaluate_aggregator(model=agg,
                                                             test_data=general_testloader,
                                                             agg_results=aggregator_results,
                                                             loss_fn=loss_fn)
                    tested_aggs.append(agg)

                if round == 0:
                    new_agg = Aggregator(input_shape=3, hidden_units=10, output_shape=10).to(device)
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
                        genealogy.append(agg)

                        # evaluate their performance
                        aggregator_results = evaluate_aggregator(model=agg,
                                                                 test_data=general_testloader,
                                                                 agg_results=aggregator_results,
                                                                 loss_fn=loss_fn)
                        tested_aggs.append(agg)

                    if round == 0:
                        new_agg = Aggregator(input_shape=3, hidden_units=10, output_shape=10).to(device)
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
                    genealogy.append(global_model)
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

    for i in genealogy:
        print(f"{i.name}´s children: {i.children_nodes}")

    filename = record_experiments(
        model=client,
        num_clients=len(local_models_list),
        split_proportions=split_proportions,
        n_rounds=NUM_ROUNDS,
        branching_factor=branch_f,
        height=height,
        client_results=client_results,
        aggregator_results=aggregator_results,
        experiment_config=experiment_config)

    return client_results, aggregator_results, naming_dict, genealogy, filename

if __name__ == "__main__":
    client_results, aggregator_results, naming_dict, genealogy, filename = create_hierarchy(local_models_list,
                                                                       split_proportions=split_proportions,
                                                                       local_trainloader=local_trainloader,
                                                                       general_testloader=general_testloader,
                                                                       branch_f=BRANCH_FACTOR,
                                                                       #height=4,
                                                                       naming_dict=naming_dict,
                                                                       NUM_ROUNDS=NUM_ROUNDS)