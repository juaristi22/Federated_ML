import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 28

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



height_1 = [9.990234375, 9.514973958333334, 9.000651041666666, 8.681640625, 9.9609375, 9.261067708333334, 10.094401041666666, 12.340494791666666, 12.493489583333334, 11.311848958333334]
height_2 = [9.9609375, 10.05859375, 10.498046875, 10.738932291666666, 9.847005208333334, 8.880208333333334, 9.527994791666666, 10.540364583333334, 11.145833333333334, 11.285807291666666]
height_3 = [9.912109375, 9.912109375, 10.205078125, 9.661458333333334, 9.938151041666666, 11.298828125, 9.798177083333334, 9.772135416666666, 11.780598958333334, 10.081380208333334]
height_4 = [10.009765625, 10.009765625, 10.78125, 11.331380208333334, 10.250651041666666, 9.820963541666666, 9.534505208333334, 9.6875, 10.60546875, 10.419921875]
height_5 = [10.3515625, 10.15625, 9.563802083333334, 10.322265625, 9.912109375, 10.787760416666666, 10.628255208333334, 10.237630208333334, 10.514322916666666, 9.9609375]


plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)