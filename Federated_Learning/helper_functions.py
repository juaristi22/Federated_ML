import torch
import matplotlib.pyplot as plt

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


def print_train_time(start, end):
    """
    Prints the time taken to train the model

    Parameters
    ----------
    start: float, starting time
    end: float, ending time

    Returns
    -------
    total_time: float, total time taken to train the model
    """
    total_time = end - start
    return total_time

def compute_bf(n_leaves, height):
    """
    Calculate the maximum branching factor of a hierarchical
        aggregation based on the desired depth of the tree structure

    Parameters
    ----------
    n_leaves: int, number of leaves in the tree
        ie. number of client models
    height: int, desired height of the tree

    Returns
    -------
    bf: int, maximum branching factor
    """
    bf = n_leaves * (1/height)
    bf = round(bf)
    return bf


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