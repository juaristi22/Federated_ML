import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 12

    fig, ax = plt.subplots()
    ax.plot(rounds, height_1, color="blue", label="height: 1")
    ax.plot(rounds, height_2, color="red", label="height: 2")
    ax.plot(rounds, height_3, color="green", label="height: 3")
    ax.plot(rounds, height_4, color="orange", label="height: 4")
    ax.plot(rounds, height_5, color="purple", label="height: 5")
    ax.set_title(f"Model performance for {n_models} client models")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid()

    filename = os.path.join(
        os.getcwd(),
        "FM_results/HierarchicalBranchingFactor"
        + "_clientmodels_"
        + str(n_models)
        + ".png",
    )

    if filename:
        fig.savefig(fname=filename)
    else:
        plt.show()



height_1 = [10.009765625, 9.469401041666666, 9.466145833333334, 9.654947916666666, 10.328776041666666, 10.045572916666666, 11.396484375, 15.748697916666666, 14.811197916666666, 12.40234375]
height_2 = [10.107421875, 10.13671875, 10.188802083333334, 10.328776041666666, 9.934895833333334, 9.840494791666666, 10.029296875, 9.908854166666666, 11.559244791666666, 11.565755208333334]
height_3 = [9.9609375, 9.944661458333334, 10.188802083333334, 10.514322916666666, 9.860026041666666, 10.169270833333334, 9.853515625, 10.358072916666666, 11.067708333333334, 11.321614583333334]
height_4 = [9.912109375, 10.009765625, 9.791666666666666, 8.649088541666666, 9.241536458333334, 9.427083333333334, 9.609375, 9.957682291666666, 10.087890625, 10.188802083333334]
height_5 = [10.009765625, 10.055338541666666, 9.990234375, 9.6484375, 9.505208333333334, 9.723307291666666, 10.237630208333334, 10.462239583333334, 10.755208333333334, 10.771484375]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)