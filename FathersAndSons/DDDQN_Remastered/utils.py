import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def manage_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plotLearning(x, scores, epsilons, filename, lines=None):
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
    for t in range(N):
        running_avg[t] = np.mean(scores[max(9, t - 20) : (t + 1)])

    ax2.scatter(x, running_avg, color="C1")

    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()

    ax2.set_ylabel("Score", color="C1")

    ax2.yaxis.set_label_position("right")

    ax2.tick_params(axis="y", color="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
