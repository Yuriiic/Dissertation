import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def manage_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plotLearning(x, scores, epsilons, filename, sell_list=None, buy_list=None, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", color="C0")
    ax.tick_params(axis="y", color="C0")

    N = len(scores)
    running_avg = np.empty(N)
    window_size = 20
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window_size) : (t + 1)])

    ax2.scatter(x, running_avg, color="C1")

    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()

    ax2.set_ylabel("Scores", color="C1")

    ax2.yaxis.set_label_position("right")

    ax2.tick_params(axis="y", color="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    if sell_list != None or buy_list != None:
        # sell_normalise = zscore(sell_list)
        # buy_normalise = zscore(buy_list)

        import pickle

        buy_path = 'buy_saved_data.pkl'
        sell_path = 'sell_saved_data.pkl'

        # Save data to a pickle file
        with open(buy_path, 'wb') as file:
            pickle.dump(buy_list, file)

        with open(sell_path, 'wb') as file:
            pickle.dump(sell_list, file)

        # sns.histplot(sell_normalise, bins=100, kde=True, color='blue', label='Data1')
        sns.histplot(buy_list, bins=100, kde=True, color='orange', label='Data2')

        # Add labels and title
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title('Distribution of Two Datasets')

        # Show legend
        plt.legend()

        # Show the plot
        plt.show()