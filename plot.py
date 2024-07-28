import matplotlib

def plot_loss_curves(height_1, height_2, height_3, height_4, height_5):
    rounds = range(len(height_1))
    n_models = 8

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



 height_1 = [9.915364583333334, 10.436197916666666, 9.524739583333334, 8.531901041666666, 8.580729166666666, 8.9453125, 13.736979166666666, 17.503255208333332, 18.430989583333332, 13.655598958333334]
 height_2 = [10.009765625, 10.026041666666666, 10.022786458333334, 10.091145833333334, 10.670572916666666, 10.869140625, 10.891927083333334, 10.504557291666666, 10.699869791666666, 10.885416666666666]
 height_3 = [10.009765625, 9.947916666666666, 9.378255208333334, 9.593098958333334, 9.661458333333334, 9.986979166666666, 10.21484375, 10.44921875, 10.589192708333334, 10.810546875]
 height_4 = [9.86328125, 9.86328125, 9.602864583333334, 10.078125, 9.811197916666666, 8.815104166666666, 9.010416666666666, 8.717447916666666, 8.746744791666666, 8.96484375]
 height_5 = [10.009765625, 10.1171875, 9.651692708333334, 9.127604166666666, 8.938802083333334, 9.244791666666666, 8.779296875, 8.671875, 8.850911458333334, 8.766276041666666]

 plot_loss_curves(height_1=height_1, height_2=height_2, height_3=height_3, height_4=height_4, height_5=height_5)