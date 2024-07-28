import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 20

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



height_1 = [9.84375, 10.009765625, 10.384114583333334, 10.319010416666666, 8.125, 10.159505208333334, 12.190755208333334, 14.654947916666666, 14.0625, 11.539713541666666]
height_2 = [10.05859375, 9.973958333333334, 10.013020833333334, 10.003255208333334, 9.615885416666666, 10.045572916666666, 10.426432291666666, 9.583333333333334, 10.703125, 10.947265625]
height_3 = [9.765625, 10.152994791666666, 10.071614583333334, 10.732421875, 8.84765625, 9.970703125, 11.136067708333334, 10.712890625, 11.416015625, 10.533854166666666]
height_4 = [9.912109375, 9.90234375, 11.526692708333334, 9.762369791666666, 9.332682291666666, 10.397135416666666, 10.8203125, 10.133463541666666, 10.283203125, 10.494791666666666]
height_5 = [9.912109375, 10.110677083333334, 10.05859375, 10.042317708333334, 10.777994791666666, 10.016276041666666, 9.606119791666666, 9.690755208333334, 10.0390625, 9.765625]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)