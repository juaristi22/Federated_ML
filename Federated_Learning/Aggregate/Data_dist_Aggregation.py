import torch
from torch import nn
from torch.utils.data import DataLoader

import copy
from Federated_Learning.Federate.Federate_Model import federate_model, train_data, test_data
from Federated_Learning.Federate.federate_data import split_data
from initialization import map_epochs_to_data

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
NUM_MODELS = 5
MAX_EPOCHS = 10

local_models_list, naming_dict, data_prop_dict, sorted_split_proportions = map_epochs_to_data(10, 10)

BATCH_SIZE = 256

general_testloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

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

