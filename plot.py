import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 24

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



height_1 = [10.4296875, 10.107421875, 10.458984375, 10.455729166666666, 9.86328125, 10.68359375, 12.438151041666666, 13.610026041666666, 13.102213541666666, 10.465494791666666]
height_2 = [9.814453125, 10.009765625, 9.993489583333334, 9.469401041666666, 10.970052083333334, 10.3125, 10.126953125, 10.061848958333334, 10.439453125, 10.152994791666666]
height_3 = [10.15625, 9.9609375, 11.15234375, 7.760416666666667, 8.160807291666666, 8.49609375, 8.798828125, 9.912109375, 12.444661458333334, 11.819661458333334]
height_4 = [10.05859375, 10.009765625, 10.227864583333334, 10.022786458333334, 9.908854166666666, 9.53125, 9.915364583333334, 10.143229166666666, 10.432942708333334, 10.166015625]
height_5 = [9.912109375, 10.009765625, 9.94140625, 10.143229166666666, 10.436197916666666, 9.895833333333334, 10.169270833333334, 10.771484375, 10.771484375, 10.843098958333334]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)