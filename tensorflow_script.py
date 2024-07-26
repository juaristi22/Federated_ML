import copy
import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.utils import to_categorical as tc
import numpy as np
import matplotlib.pyplot as plt
import os
import json

###set parameters###
num_models = 8
epochs_per_round = 5
rounds = 10
bs = 128

###load data###
(x,y),(x_test,y_test) = cifar10.load_data()
x  = x/255.0
y = tc(y)
x_test = x_test/255.0
y_test = tc(y_test)

def split_data(x, y, n_splits, equal_sizes):
    if equal_sizes:
        split_sizes = [len(x) // n_splits for _ in range(n_splits)]
    else:
        total_size = len(x)
        split_sizes = []
        for i in range(n_splits - 1):
            split = random.randrange(1, total_size)
            split_sizes.append(split)
            total_size -= split
        split_sizes.append(total_size)
    x_splits = []
    y_splits = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        subset_x = x[start_idx:end_idx]
        subset_y = y[start_idx:end_idx]
        x_splits.append(subset_x)
        y_splits.append(subset_y)
        start_idx = end_idx
    return x_splits, y_splits, split_sizes

###define model###
def get_model(classes=10,input_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(32,3,1,padding='same',activation='relu',input_shape=input_shape))
    model.add(Dropout(.1))
    model.add(Conv2D(32,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(64,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(Conv2D(64,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(Conv2D(128,3,1,padding='same',activation='relu'))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(classes,activation='softmax'))
    model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['acc'])
    #model.summary()
    return(model)

def run_federation(num_models, rounds, x, y, x_test, y_test):
    epochs_per_round = 5
    bs = 128

    ###federated learning###
    global_model = get_model()
    weights_list = [global_model.get_weights()]*num_models

    x_splits, y_splits, split_sizes = split_data(x, y, num_models, True)

    for r in range(rounds):
        print('Starting round '+str(r+1))
        for m in range(num_models):
            print('training model ' + str(m+1))
            global_model.set_weights(weights_list[m])
            global_model.fit(x_splits[m],y_splits[m],
                             epochs=epochs_per_round,batch_size=bs,validation_data=(x_test,y_test),verbose=0)
            weights_list[m]=global_model.get_weights()
        global_weights=np.mean(np.array(weights_list,dtype='object'),axis=0)
        weights_list = [global_weights]*num_models
        global_model.set_weights(global_weights)
        print('global model performance after '+str(r+1)+' round:')
        global_model.evaluate(x_test,y_test,batch_size=1024)


def compute_bf(n_leaves, height):
    bf = n_leaves * (1/height)
    bf = round(bf)
    return bf


def create_hierarchy(NUM_ROUNDS, num_models, x, y, x_test, y_test,
                     branch_f=None, height=None):
    epochs_per_round = 2
    bs = 128

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
        branch_f = compute_bf(num_models, height)

    test_acc = []

    global_model = get_model()
    client_weights = [global_model.get_weights()] * num_models
    x_splits, y_splits, split_sizes = split_data(x, y, num_models, True)

    for round in range(NUM_ROUNDS):
        print(f"Round: {round}:")
        weights_list = copy.deepcopy(client_weights)
        for m in range(num_models):
            print('training model ' + str(m + 1))
            global_model.set_weights(client_weights[m])
            global_model.fit(x_splits[m],y_splits[m], epochs=epochs_per_round,
                             batch_size=bs, validation_data=(x_test, y_test), verbose=0)
            weights_list[m] = global_model.get_weights()
            client_weights[m] = global_model.get_weights()

        aggregator_weights = []
        print('aggregating client models')
        while weights_list:
            if len(weights_list) > (branch_f - 1):
                avg = []
                for i in range(branch_f):
                    current_weights = weights_list.pop(0)
                    avg.append(current_weights)
                averaged_weights = np.mean(np.array(avg,dtype='object'),axis=0)
                global_model.set_weights(averaged_weights)
                global_model.evaluate(x_test, y_test, batch_size=1024)
                aggregator_weights.append(averaged_weights)
                print(f"averaging weights for branching factor of {branch_f}")
            else:
                avg = []
                for i in range(len(weights_list)):
                    current_weights = weights_list.pop(0)
                    avg.append(current_weights)
                averaged_weights = np.mean(np.array(avg, dtype='object'), axis=0)
                global_model.set_weights(averaged_weights)
                global_model.evaluate(x_test, y_test, batch_size=1024)
                aggregator_weights.append(averaged_weights)
                print(f"averaging weights for branching factor of {len(weights_list)}")

        print('completing hierarchy with aggregator nodes')
        while aggregator_weights:
            if len(aggregator_weights) > (branch_f - 1):
                avg = []
                for i in range(branch_f):
                    current_weights = weights_list.pop(0)
                    avg.append(current_weights)
                averaged_weights = np.mean(np.array(avg, dtype='object'), axis=0)
                global_model.set_weights(averaged_weights)
                global_model.evaluate(x_test, y_test, batch_size=1024)
                aggregator_weights.append(averaged_weights)
                print(f"averaging weights for branching factor of {branch_f}")
            else:
                if len(aggregator_weights) == 1:
                    current_weights = aggregator_weights.pop(0)
                    global_model.set_weights(current_weights)
                    print('global model performance after ' + str(round + 1) + ' rounds:')
                    metrics = global_model.evaluate(x_test, y_test, batch_size=1024)
                    test_acc.append(metrics[1])
                else:
                    avg = []
                    for i in range(branch_f):
                        current_weights = weights_list.pop(0)
                        avg.append(current_weights)
                    averaged_weights = np.mean(np.array(avg, dtype='object'), axis=0)
                    global_model.set_weights(averaged_weights)
                    global_model.evaluate(x_test, y_test, batch_size=1024)
                    aggregator_weights.append(averaged_weights)
                    print(f"averaging weights for branching factor of {len(weights_list)}")

    return test_acc, split_sizes


def plot_loss_curves(accuracies, config, filename=None):
    rounds = range(len(accuracies))

    plt.plot(rounds, accuracies, color="blue")
    plt.set_title(config)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.grid()

    if filename:
        plt.savefig(fname=filename)
    else:
        plt.show()

def record_experiments(
    num_clients,
    split_proportions,
    n_rounds,
    branching_factor,
    height,
    accuracy_results,
    experiment_config=None):

    results_dict = {
        "model_name": "HierarchicalBranchingFactor",
        "num_clients": num_clients,
        "data_split_proportions": split_proportions,
        "n_rounds": n_rounds,
        "max_branching_factor": branching_factor,
        "height": height,
        "accuracy_results": accuracy_results,
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
        json.dump(results_dict, f)

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


def experiment_configs(max_n_models, max_bf=None, max_height=None):
    NUM_MODELS = [i for i in range(2, max_n_models+1)]
    if max_bf:
        BRANCHING_FACTOR = [i for i in range(2, max_bf+1)]
    if max_height:
        HEIGHT = [i for i in range(1, max_height+1)]
    equal_data_dist = [True, False]

    configurations = []
    config_descriptions = []
    if max_bf:
        for bf in BRANCHING_FACTOR:
            for n_models in NUM_MODELS:
                #for data_dist in equal_data_dist:
                if n_models >= bf:
                    configs_dict = {}
                    configs_dict["n_models"] = n_models
                    configs_dict["bf"] = bf
                    configs_dict["height"] = None
                    configs_dict["data_dist"] = True
                    configurations.append(configs_dict)
                    config_descriptions.append(f"n_models_{n_models}_bf_{bf}_equal_data_dist{True}")
    elif max_height:
        for height in HEIGHT:
            for n_models in NUM_MODELS:
                if n_models >= height + 1:
                    #for data_dist in equal_data_dist:
                    configs_dict = {}
                    configs_dict["n_models"] = n_models
                    configs_dict["bf"] = None
                    configs_dict["height"] = height
                    configs_dict["data_dist"] = True
                    configurations.append(configs_dict)
                    config_descriptions.append(f"n_models_{n_models}_height_{height}_equal_data_dist{True}")

    return configurations, config_descriptions


def experiment_running(max_n_models, max_bf=None, max_height=None, experiments=3):
    ROUNDS = 3

    (x, y), (x_test, y_test) = cifar10.load_data()
    x = x / 255.0
    y = tc(y)
    x_test = x_test / 255.0
    y_test = tc(y_test)

    configurations, config_descriptions = experiment_configs(max_n_models=max_n_models,
                                                             max_bf=max_bf, max_height=max_height)
    print(configurations)
    trial = 0
    for configuration in configurations:
        print(f"Running experimental configuration {trial} on configuration: {configurations[trial]}")
        for experiment in range(experiments):
            print(f"running experiment {experiment}")
            test_acc, split_sizes = create_hierarchy(NUM_ROUNDS=ROUNDS, num_models=configuration["n_models"],
                             x=x, y=y, x_test=x_test, y_test=y_test,
                             branch_f=configuration["bf"], height=configuration["height"])
            print(f"Experiment {experiment} produces accuracy: {test_acc}")

            if trial == 0:
                total_accuracy = copy.deepcopy(test_acc)
            else:
                for i in range(len(total_accuracy)):
                    total_accuracy[i] += test_acc[i]

        for i in range(len(total_accuracy)):
            total_accuracy[i] /= experiments

        print(f"total accuracy for current configuration: {total_accuracy}")

        record_experiments(
        num_clients=configuration["n_models"],
        split_proportions=split_sizes,
        n_rounds=ROUNDS,
        branching_factor=configuration["bf"],
        height=configuration["height"],
        accuracy_results=total_accuracy,
        experiment_config=config_descriptions[trial])

        trial += 1

experiment_running(max_n_models=10, max_bf=None, max_height=3)