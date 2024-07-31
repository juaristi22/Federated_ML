import Federated_Model as FM
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
def run_experiments(learning_rate, n_models, num_experiments, epochs, rounds, device=device, filename=None):
    """
    Runs experiments for the Federated_Model.py
        script with different hyperparameter configurations

    Parameters
    ----------
    learning_rate: float, models' learning rate
    n_models: int, number of local models to experiment on
    num_experiments: int, number of experiments to run
    epochs: int, number of epochs that each model should train for
    rounds: int, number of rounds of federations
    device: str, default:device, device on which to perform computation
    filename: str, default:None, path to folder in which to save figures
    """

    EPOCHS = epochs
    ROUNDS = rounds

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor())

    results_dict = dict()

    for local_models in range(1, n_models+1):
        experiment_results = 0
        for i in range(num_experiments):
            global_model = FM.CNNModel(input_shape=3, hidden_units=10, output_shape=10).to(device)

            global_results = FM.federate_model(global_model_instance=global_model,
                                               train_data=train_data,
                                               test_data=test_data,
                                               learning_rate=learning_rate,
                                               equal_sizes=True,
                                               NUM_MODELS=local_models,
                                               NUM_ROUNDS=ROUNDS,
                                               EPOCHS=EPOCHS)

            if experiment_results != 0:
                current_test_loss = global_results["test_loss"]
                current_test_acc = global_results["test_acc"]

                avg_test_loss = experiment_results["test_loss"]
                avg_test_acc = experiment_results["test_acc"]

                for i in range(len(avg_test_acc)):
                    avg_test_loss[i] += current_test_loss[i]
                    avg_test_loss[i] /= 2

                    avg_test_acc[i] += current_test_acc[i]
                    avg_test_acc[i] /= 2

                    experiment_results["test_loss"] = avg_test_loss
                    experiment_results["test_acc"] = avg_test_acc

            else:
                experiment_results = global_results

        results_dict[f"equal data dist, {local_models} clients"] = experiment_results

    for local_models in range(1, n_models+1):
        experiment_results = 0
        for i in range(num_experiments):
            global_model = FM.CNNModel(input_shape=3, hidden_units=10, output_shape=10).to(device)

            global_results = FM.federate_model(global_model_instance=global_model,
                                               train_data=train_data,
                                               test_data=test_data,
                                               learning_rate=learning_rate,
                                               equal_sizes=True,
                                               NUM_MODELS=local_models,
                                               NUM_ROUNDS=ROUNDS,
                                               EPOCHS=EPOCHS)

            if experiment_results != 0:
                current_test_loss = global_results["test_loss"]
                current_test_acc = global_results["test_acc"]

                avg_test_loss = experiment_results["test_loss"]
                avg_test_acc = experiment_results["test_acc"]

                for i in range(len(avg_test_acc)):
                    avg_test_loss[i] += current_test_loss[i]
                    avg_test_loss[i] /= 2

                    avg_test_acc[i] += current_test_acc[i]
                    avg_test_acc[i] /= 2

                    experiment_results["test_loss"] = avg_test_loss
                    experiment_results["test_acc"] = avg_test_acc

            else:
                experiment_results = global_results

        results_dict[f"random data dist, {local_models} clients"] = experiment_results

        if filename:
            FM.plot_loss_curves(experiment_results, filename=filename)

    return results_dict

run_experiments(learning_rate=0.0001, n_models=2, num_experiments=2, epochs=2, rounds=5)
