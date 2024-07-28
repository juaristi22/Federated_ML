import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 14

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



height_1 = [10.05859375, 10.100911458333334, 10.247395833333334, 11.002604166666666, 10.445963541666666, 10.589192708333334, 12.802734375, 14.544270833333334, 13.675130208333334, 10.572916666666666]
height_2 = [9.9609375, 10.1171875, 10.227864583333334, 10.5859375, 10.25390625, 9.918619791666666, 10.686848958333334, 11.54296875, 12.093098958333334, 12.01171875]
height_3 = [9.86328125, 10.348307291666666, 10.0390625, 9.807942708333334, 9.8828125, 9.713541666666666, 9.833984375, 9.703776041666666, 10.065104166666666, 10.481770833333334]
height_4 = [9.814453125, 10.05859375, 10.100911458333334, 9.84375, 9.5703125, 10.042317708333334, 9.013671875, 9.479166666666666, 10.602213541666666, 10.5078125]
height_5 = [9.86328125, 9.860026041666666, 9.661458333333334, 10.283203125, 10.628255208333334, 10.885416666666666, 11.129557291666666, 11.302083333333334, 11.201171875, 10.872395833333334]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)