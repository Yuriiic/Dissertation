import numpy as np
from agent import Agent
from env import TradeEnv
from gym import wrappers
from tqdm import tqdm
from utils import plot_learning_curve, manage_memory

def run(train=True):
    # my gpus aren't being recognised
    manage_memory()

    env = TradeEnv(trading_cost=0, train=train)

    best_score = np.inf
    load_checkpoint = False

    n_games = 1
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        lr=0.0001,
        input_dims=[6],
        n_actions=env.action_space.n,
        mem_size=1000,
        eps_min=0.1,
        batch_size=32,
        replace=100,
        eps_dec=1e-5,
    )
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    fname = "lr" + str(agent.lr) + "_" + str(n_games) + "games"
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, observation)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(
            "episode {} score {:.1f} avg score {:.1f} "
            "best score {:.1f} epsilon {:.2f} steps {}".format(
                i, score, avg_score, best_score, agent.epsilon, n_steps
            )
        )

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)


if __name__ == "__main__":
