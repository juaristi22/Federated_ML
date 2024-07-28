import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 32

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



height_1 = [10.133463541666666, 9.840494791666666, 9.498697916666666, 9.410807291666666, 8.776041666666666, 9.7265625, 11.58203125, 14.534505208333334, 14.84375, 11.715494791666666]
height_2 = [9.807942708333334, 9.912109375, 9.938151041666666, 9.612630208333334, 10.25390625, 10.107421875, 9.886067708333334, 10.960286458333334, 10.042317708333334, 10.91796875]
height_3 = [9.912109375, 10.15625, 10.068359375, 10.048828125, 10.436197916666666, 10.423177083333334, 11.145833333333334, 10.6640625, 11.282552083333334, 10.572916666666666]
height_4 = [9.86328125, 10.107421875, 9.879557291666666, 9.091796875, 9.837239583333334, 10.419921875, 10.44921875, 10.673828125, 9.921875, 9.345703125]
height_5 = [10.05859375, 9.86328125, 10.240885416666666, 10.029296875, 10.240885416666666, 10.234375, 10.032552083333334, 10.875651041666666, 10.901692708333334, 11.012369791666666]


plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)