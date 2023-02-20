import gym
import numpy as np
from yt_agent import Agent
from utils import plot_learning_curve

#if __name__ == 'yt_main':
if True:
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n,  # number of actions possible (4)
                  batch_size = batch_size,  # number of sequential pieces learned from in a single epoch?
                  alpha = alpha,  # learning rate
                  n_epochs = n_epochs,  # number sampled to learn from??
                  input_dims = env.observation_space.shape  # neural network "x" input?
                  )
    n_games = 50

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i}, score {score}, avg score {avg_score}, time_steps {n_steps}, learning_steps {learn_iters}')
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)




