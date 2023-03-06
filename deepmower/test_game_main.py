import gym
import numpy as np
from mow_memory import Agent
from utils import plot_learning_curve
import test_game_base

from logger import logger
import os
import torch

from PPO import PPO

from datetime import datetime




#
#  https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
#
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
# add entropy
#
# add "done" critic
#
# train over each run instead of batch (in tandem with above)
#
# add scheduler on LR, etc


debug = False

run_id = 2
lawn_num = 1

random_seed = 0         # set random seed if required (0 = no random seed)

n_actions = 4
input_dims = 13

env_name = f"lawn{lawn_num}"

#if __name__ == 'yt_main':
if True:
    #env = gym.make('CartPole-v0')


    auto = False
    sticky_probability = 0.2

    env = test_game_base.test_game(lawn_num, no_print=True)

    ####### initialize environment hyperparameters ######


    max_ep_len = 5000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = 1000        # print avg reward in the interval (in num timesteps)
    log_freq = 1000           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 128      # update policy every n timesteps
    K_epochs = 64               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    #####################################################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    logger = logger(run_num, path=log_dir)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)



    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", input_dims)
    print("action space dimension : ", n_actions)
    print("--------------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")



    #T = 128
    # T = 128  # time horizon
    # batch_size = 32
    # n_epochs = 3
    # #alpha = 0.0003
    # a = 1
    # alpha = 5e-4 * a
    # policy_clip = 0.2 * a
    # agent = Agent(n_actions= n_actions,  # number of actions possible (4)
    #               batch_size = batch_size,  # number of sequential pieces learned from in a single epoch?
    #               alpha = alpha,  # learning rate
    #               n_epochs = n_epochs,  # number sampled to learn from??
    #               policy_clip = policy_clip,
    #               input_dims = input_dims  # env.observation_space.shape  # neural network "x" input?
    #               )
    # n_games = 100000
    #
    # figure_file = f'plots/{lawn_num}-{run_id}.png'
    #
    # best_score = 0
    # score_history = []
    #
    # learn_iters = 0
    # avg_score = 0
    # n_steps = 0
    #
    # logger = logger(run_id = f"{str(lawn_num)}-{str(run_id)}")

    ppo_agent = PPO(n_actions, input_dims, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        observation = env.state.permute(2,0,1).float()  # oops, needed to change order
        observation_numericals = env.state_numericals.float()

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(observation, observation_numericals)
            observation, observation_numericals, reward, done, info = env.step(action)
            observation = observation.permute(2, 0, 1)  # oops, needed to change order

            # saving reward and is_terminals
            logger.log(action, reward)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        logger.write(sum(logger.rewards))

        i_episode += 1

    log_f.close()
    #env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

# if __name__ == '__main__':
#     train()

