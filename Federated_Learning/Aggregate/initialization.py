from Federated_Learning.Learn.Model import Client
from Federated_Learning.Federate.federate_data import split_data
from Federated_Learning.Federate.Federate_Model import train_data, test_data


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

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


def map_epochs_to_data(NUM_MODELS, MAX_EPOCHS):
    """
    Randomly distributes data across all client models and
        calculates the number of epochs that each client model
        should run given its data load

    Parameters
    ----------
    NUM_MODELS: int, number of client models
    MAX_EPOCHS: int, number of epochs that the model with largest
        data load should run for
    """
    BATCH_SIZE = 256
    local_models_list, naming_dict = initialize_models(NUM_MODELS)

    local_trainloader, split_proportions = split_data(
        data=train_data,
        n_splits=NUM_MODELS,
        batch_size=BATCH_SIZE,
        equal_sizes=False)

    sorted_split_proportions = copy.deepcopy(split_proportions)
    sorted_split_proportions.sort()
    highest_dataload = sorted_split_proportions[-1]

    data_prop_dict = {}
    for i in range(len(local_trainloader)):
        model = local_models_list[i]
        data_prop_dict[split_proportions[i]] = model
        model.data = local_trainloader[i]
        current_dataload = split_proportions[i]
        if current_dataload == highest_dataload:
            model.epochs = MAX_EPOCHS
        else:
            proportion = current_dataload / highest_dataload
            if proportion >= 0.75:
                model.epochs = MAX_EPOCHS
            elif proportion >= 0.5 and proportion < 0.75:
                EPOCHS = MAX_EPOCHS // (4/3)
                model.epochs = EPOCHS
            elif proportion >= 0.25 and proportion < 0.5:
                EPOCHS = MAX_EPOCHS // (4/2)
                model.epochs = EPOCHS
            elif proportion < 0.25:
                EPOCHS = MAX_EPOCHS // (4/1)
                model.epochs = EPOCHS

    return local_models_list, naming_dict, data_prop_dict, sorted_split_proportions