import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 18

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



height_1 = [10.107421875, 10.888671875, 9.244791666666666, 10.201822916666666, 10.224609375, 10.46875, 11.272786458333334, 14.117838541666666, 13.899739583333334, 10.579427083333334]
height_2 = [9.86328125, 9.9609375, 9.765625, 10.208333333333334, 9.488932291666666, 9.811197916666666, 9.915364583333334, 10.126953125, 9.772135416666666, 11.686197916666666]
height_3 = [9.9609375, 9.905598958333334, 10.680338541666666, 9.713541666666666, 9.964192708333334, 10.595703125, 10.813802083333334, 10.777994791666666, 12.057291666666666, 10.296223958333334]
height_4 = [10.009765625, 9.967447916666666, 9.899088541666666, 10.286458333333334, 9.664713541666666, 9.055989583333334, 9.638671875, 10.963541666666666, 10.992838541666666, 11.110026041666666]
height_5 = [10.05859375, 9.957682291666666, 9.931640625, 9.794921875, 9.684244791666666, 10.0390625, 10.46875, 10.771484375, 11.276041666666666, 11.669921875]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)