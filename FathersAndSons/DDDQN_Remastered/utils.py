import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle

def manage_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plotLearning(x, scores, epsilons, filename, avg_scores, uncertainties, lines=None):
    with open('avg_scores.pkl', 'wb') as file:
        pickle.dump(avg_scores, file)

    with open('uncertainties.pkl', 'wb') as file:
        pickle.dump(uncertainties, file)

    with open('raw_scores.pkl', 'wb') as file:
        pickle.dump(scores, file)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", color="C0")
    ax.tick_params(axis="y", color="C0")

    N = len(scores)
    running_avg = np.empty(N)
    window_size = 10
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window_size) : (t + 1)])

    ax.scatter(x, running_avg, color="C1")
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    ax.set_ylabel("Score", color="C1")

    ax.yaxis.set_label_position("right")

    ax.tick_params(axis="y", color="C1")
    ax.set_title('Epsilon vs Running Score (Last 10) Comparison')

    ax2.plot(x, avg_scores, "o", label="Average scores and their uncertainties")
    ax2.set_title('Robustness of the Average Score (Game Average)')
    ax2.legend()

    plt.tight_layout()

    plt.show()

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    fig.savefig(filename)
