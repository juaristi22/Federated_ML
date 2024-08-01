import torch
from torch import nn
from torch.utils.data import DataLoader

import copy
import random
import json
import os
from Federated_Learning.Federate.Federate_Model import federate_model, train_data, test_data
from Federated_Learning.Learn.Model import Client, Aggregator
from Federated_Learning.helper_functions import plot_loss_curves, compute_bf
from Federated_Learning.Federate.federate_data import split_data
from initialization import initialize_models

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

BATCH_SIZE = 256
NUM_MODELS = 2
equal_sizes = True
NUM_ROUNDS = 10
BRANCH_FACTOR = 2

general_trainloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

general_testloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# split training data for the local models
local_trainloader, split_proportions = split_data(
    data=train_data,
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

    plot_loss_curves(aggregator_results["Global_Model"], filename=filename, config=experiment_config)

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
    input_shape=3
    hidden_units=10
    output_shape=10

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
                    agg = Aggregator(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape).to(device)
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
                    agg = Aggregator(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape).to(device)
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
                    new_agg = Aggregator(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape).to(device)
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
                        new_agg = Aggregator(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape).to(device)
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

    return client_results, aggregator_results, naming_dict, genealogy

if __name__ == "__main__":
    client_results, aggregator_results, naming_dict, genealogy, filename = create_hierarchy(local_models_list,
                                                                       split_proportions=split_proportions,
                                                                       local_trainloader=local_trainloader,
                                                                       general_testloader=general_testloader,
                                                                       branch_f=BRANCH_FACTOR,
                                                                       #height=4,
                                                                       naming_dict=naming_dict,
                                                                       NUM_ROUNDS=NUM_ROUNDS)