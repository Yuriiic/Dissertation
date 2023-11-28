import numpy as np
from Agent import Agent
from TradingEnv import TradeEnv
from tqdm import tqdm
from utils import plotLearning


def run(filename, train=True, n_games=1):
    env = TradeEnv(trading_cost=0, train=train, timeframe="15M")

    best_score = -np.inf

    # First Agent, 15/30 minutes
    agent = Agent(
        gamma=0.999,
        epsilon=0,
        learning_rate=1e-3,
        input_dims=[5],
        mem_size=100000,
        batch_size=64,
        fc1_dims=128,
        fc2_dims=128,
        replace=100,
        n_actions=2,
    )

    # Second Agent, 1 hour
    # agent = Agent(
    #     gamma=0.999,
    #     epsilon=1.0,
    #     learning_rate=1e-3,
    #     input_dims=[5],
    #     mem_size=100000,
    #     batch_size=64,
    #     eps_end=0.01,
    #     fc1_dims=128,
    #     fc2_dims=128,
    #     replace=100,
    #     n_actions=2,
    # )

    # Third Agent that makes the decision, 1 day
    # agent = Agent(
    #     gamma=0.999,
    #     epsilon=1.0,
    #     learning_rate=1e-3,
    #     input_dims=[5],
    #     mem_size=100000,
    #     batch_size=64,
    #     eps_end=0.01,
    #     fc1_dims=128,
    #     fc2_dims=128,
    #     replace=100,
    #     n_actions=2,
    # )

    if not train:
        agent.load_model()
        # agent.epsilon = agent.eps_end

    scores, eps_history = [], []

    all_actions = []
    for i in tqdm(range(n_games)):
        Start_Quantity = 1000000
        done = False
        EpRewards = 0
        observation = env.reset()
        all_EpRewards = []
        while not done:
            action, actions_list = agent.choose_action(observation)
            all_actions.append(actions_list.numpy()[0])
            observation_, reward, done, info = env.step(action, observation)
            EpRewards += reward
            all_EpRewards.append(reward)
            if train:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn(reward)
            observation = observation_
        eps_history.append(agent.epsilon)
        scores.append(EpRewards)

        avg_score = np.mean(scores)
        uncertainty = np.std(scores)
        print(
            f"episode {i} score {round(EpRewards, 2)} avg score {round(avg_score, 2)} +/- {round(uncertainty, 2)}; "
            f"best score {round(best_score, 2)} epsilon {round(agent.epsilon, 5)}; "
            f"Started episode with {round(Start_Quantity, 2)}, finished episode with {round((Start_Quantity + EpRewards), 2)}; "
            f"Average Perspective: {round((Start_Quantity + avg_score), 2)} +/- {round(uncertainty, 2)}; "
        )

        if EpRewards > best_score:
            if train:
                agent.save_model()
            best_score = EpRewards
            print(f"The best score : {round(best_score, 2)}")

        if i == (n_games - 1):
            if train:
                agent.save_model_latest()

    all = np.array(all_actions)
    tp = np.transpose(all)
    sell_list = tp[0].tolist()
    buy_list = tp[1].tolist()

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename, sell_list, buy_list)


def normalise_rewards(rewards):
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    normalized_rewards = 2 * (rewards - min_reward) / (max_reward - min_reward) - 1

    return normalized_rewards


if __name__ == "__main__":
    # The agent does not have a hold option. The hold comes from the timeframe you select.
    # In the main paper, the agent observes N previous time steps as well as the current position of its higher timeframe.
    # For now the look back is 1.
    np.random.seed(42)
    run("TrainTest.png", True, n_games=1)
    # run("TrainTest.png", False, n_games=1)
