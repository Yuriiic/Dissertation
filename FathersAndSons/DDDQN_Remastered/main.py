import numpy as np
from Agent import Agent
from TradingEnv import TradeEnv
from tqdm import tqdm
from utils import plotLearning


def run(filename, train=True, n_games=1):
    env = TradeEnv(trading_cost=0, train=train)

    best_score = -np.inf

    agent = Agent(
        gamma=0.999,
        epsilon=1.0,
        learning_rate=1e-3,
        input_dims=[5],
        mem_size=100000,
        batch_size=64,
        eps_end=0.01,
        fc1_dims=128,
        fc2_dims=128,
        replace=100,
        n_actions=3,
    )

    if not train:
        agent.load_model()
        agent.epsilon = agent.eps_end

    scores, eps_history = [], []

    for i in tqdm(range(n_games)):
        Start_Quantity = 1000000
        done = False
        EpRewards = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, observation)
            EpRewards += reward
            if train:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn(reward)
            observation = observation_
        eps_history.append(agent.epsilon)
        scores.append(EpRewards)

        avg_score = np.mean(scores)
        uncertainty = np.std(scores)
        print(
            f"episode {i} score {EpRewards} avg score {avg_score} +/- {uncertainty}"
            f"best score {best_score} epsilon {agent.epsilon}"
            f"Started episode with {Start_Quantity}, finished episode with {Start_Quantity + EpRewards}"
            f"Average Perspective: {Start_Quantity + avg_score} +/- {uncertainty}"
        )

        if EpRewards > best_score:
            if train:
                agent.save_model()
            best_score = EpRewards
            print(f"The best score : {best_score}")

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)


if __name__ == "__main__":
    np.random.seed(42)
    run("DynamicEpsilon.png", True, n_games=25)
    # run("Test.png", False, n_games=1)
