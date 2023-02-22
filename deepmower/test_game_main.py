import gym
import numpy as np
from mow_memory import Agent
from utils import plot_learning_curve
import test_game_base

from logger import logger

# change LR:
# - took longer to die
#
# print sum of weights:
# - doesn't reveal anything
#
# discriminate between terminal state:
# - think its ok!!
#
#
# cosine restart:
#


debug = False

run_id = 4
lawn_num = 3

#if __name__ == 'yt_main':
if True:
    #env = gym.make('CartPole-v0')


    auto = False
    sticky_probability = 0.2

    env = test_game_base.test_game(lawn_num, no_print=True)


    #N = 128
    N = 128
    batch_size = 32
    n_epochs = 3
    alpha = 0.0003
    agent = Agent(n_actions= 4,  # number of actions possible (4)
                  batch_size = batch_size,  # number of sequential pieces learned from in a single epoch?
                  alpha = alpha,  # learning rate
                  n_epochs = n_epochs,  # number sampled to learn from??
                  input_dims = 11  # env.observation_space.shape  # neural network "x" input?
                  )
    n_games = 10000

    figure_file = 'plots/test_game.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    logger = logger(run_id = run_id)

    for i in range(n_games):
        # TODO: add run logger that exports action array, reward, % mowed, fuel picked up
        env.reset()
        #observation = env.save_state()
        observation = env.state.permute(2,0,1).float()  # oops, needed to change order
        observation_numericals = env.state_numericals.float()  # oops, needed to change order
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation, observation_numericals)



            observation_, observation_numericals_, reward, done, info = env.step(action)
            logger.log(action, reward)
            observation_ = observation_.permute(2,0,1).float()
            observation_numericals_ = observation_numericals_.float()
            n_steps += 1
            score += reward
            agent.remember(observation, observation_numericals, action, prob, val, reward, done)
            if n_steps % N == 0:
                if debug is True:
                    agent.debug()
                agent.learn()
                if debug is True:
                    agent.debug()

                learn_iters += 1
            observation = observation_
            observation_numericals = observation_numericals_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            x = [i + 1 for i in range(len(score_history))]
            plot_learning_curve(x, score_history, figure_file)

        print(f'episode {i}, score {score}, avg score {avg_score}, time_steps {n_steps}, learning_steps {learn_iters}')
        logger.write(score)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)




