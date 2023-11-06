from TradingEnv import TradeEnv
from Agent import Agent
import numpy as np
from utils import manage_memory, plotLearning
from tqdm import tqdm


def run(filename, train=True, n_games=1):
    manage_memory()

    env = TradeEnv(trading_cost=0, train=train)

    agent = Agent(
        gamma=0.999,
        epsilon=1.0,
        learning_rate=1e-3,
        input_dims=[5],
        epsilon_dec=1e-3,
        mem_size=1000,
        batch_size=64,
        eps_end=0.01,
        fc1_dims=128,
        fc2_dims=128,
        replace=100,
        n_actions=3,
    )

    if not train:
        n_steps = 0
        while n_steps < agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action, observation)
            agent.store_transition(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_model()

    scores, eps_history = [], []

    for i in tqdm(range(n_games)):
        done = False
        EpRewards = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, observation)
            EpRewards += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            if train:
                agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(EpRewards)

        avg_score = np.mean(scores)
        print(
            "episode: ",
            i,
            "score %.3f" % EpRewards,
            "average score %.3f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )

    if train:
        agent.save_model()
    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)


if __name__ == "__main__":
    np.random.seed(42)
    run("ThirdTry_100.png", True, n_games=100)
    run("Test.png", False, n_games=1)
