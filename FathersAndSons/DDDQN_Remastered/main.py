import numpy as np
from Agent import Agent
from TradingEnv import TradeEnv
from tqdm import tqdm
from utils import plotLearning


def run(filename, train=True, n_games=1):
    env = TradeEnv(trading_cost=0, train=train, timeframe="1D")

    best_score = -np.inf

    # First Agent, 15/30 minutes
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
        agent.epsilon = agent.eps_end

    scores, eps_history = [], []

    for i in tqdm(range(n_games)):
        Start_Quantity = 1000000
        done = False
        EpRewards = 0
        observation = env.reset()
        all_EpRewards = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, observation)
            EpRewards += reward
            all_EpRewards.append(reward)
            if train:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn(reward)
            observation = observation_
        # normalised_rewards = normalise_rewards(all_EpRewards)
        # agent.epsilon_decrease(np.mean(normalised_rewards))
        # if len(all_EpRewards) > 2:
        #     if all_EpRewards[-1] < EpRewards:
        #         # maybe multiply by how many rewards there are?
        #         normalised_rewards = normalise_rewards(all_EpRewards)
        #         agent.epsilon_decrease(np.mean(normalised_rewards))
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

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)


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
    run("500Games.png", True, n_games=100)
    # run("Test.png", False, n_games=1)
