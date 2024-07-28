import matplotlib.pyplot as plt
import os
import json

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 22

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



height_1 = [10.15625, 10.032552083333334, 9.339192708333334, 9.166666666666666, 10.520833333333334, 9.391276041666666, 9.895833333333334, 13.06640625, 12.291666666666666, 9.983723958333334]
height_2 = [10.15625, 9.8828125, 8.977864583333334, 8.854166666666666, 9.684244791666666, 10.869140625, 10.638020833333334, 10.270182291666666, 11.061197916666666, 9.313151041666666]
height_3 = [9.912109375, 9.8828125, 10.029296875, 9.388020833333334, 9.505208333333334, 8.629557291666666, 8.889973958333334, 11.064453125, 12.268880208333334, 11.702473958333334]
height_4 = [9.912109375, 9.814453125, 10.33203125, 10.960286458333334, 10.276692708333334, 10.621744791666666, 10.992838541666666, 10.992838541666666, 10.91796875, 11.630859375]
height_5 = [9.814453125, 9.583333333333334, 9.9609375, 9.163411458333334, 8.896484375, 8.987630208333334, 9.514973958333334, 9.814453125, 10.01953125, 10.716145833333334]

plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)