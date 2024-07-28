import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 16

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



height_1 = [9.905598958333334, 10.299479166666666, 11.7578125, 10.579427083333334, 10.693359375, 10.413411458333334, 12.337239583333334, 14.375, 15.152994791666666, 11.087239583333334]
height_2 = [10.084635416666666, 9.9609375, 9.908854166666666, 10.8203125, 10.387369791666666, 9.736328125, 9.713541666666666, 10.107421875, 10.335286458333334, 9.596354166666666]
height_3 = [10.05859375, 10.05859375, 9.700520833333334, 10.231119791666666, 9.853515625, 10.390625, 9.895833333333334, 9.625651041666666, 9.583333333333334, 10.76171875]
height_4 = [10.179036458333334, 10.05859375, 10.035807291666666, 9.417317708333334, 9.957682291666666, 9.713541666666666, 9.74609375, 10.638020833333334, 11.110026041666666, 11.389973958333334]
height_5 = [10.107421875, 10.205078125, 10.751953125, 10.364583333333334, 10.44921875, 10.3515625, 10.6640625, 10.72265625, 10.72265625, 10.517578125]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)