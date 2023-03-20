import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import test_game_base


from PPO_ikostrikov import PPO
from a2c_ppo_acktr import utils

#from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
#from a2c_ppo_acktr.envs import make_vec_envs
#from a2c_ppo_acktr.envs import make_vec_env
from model import Policy
from storage import RolloutStorage
from evaluation import evaluate

from pca import get_go_paths


from datetime import datetime

debug = False


n_actions = 4
input_dims = 17

# SIL/hypers stuff: https://arxiv.org/pdf/2004.12919.pdf p31


args = get_args()

# args.run_id = 10000023
# args.lawn_num = 21
# args.go_explore_frequency = 16
# args.go_explore = True






run_id = args.run_id
lawn_num = args.lawn_num





random_seed = lawn_num + run_id + lawn_num * run_id        # set random seed if required (0 = no random seed)
args.seed = random_seed
np.random.seed(random_seed)

env_name = f"lawn{lawn_num}"


print(f"Beginning run on Lawn {lawn_num}.  Run ID: {run_id}.  Random Seed: {random_seed}")




def main():
    #args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)



    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_env(args.env_name,
    #                 args.seed,
    #                 args.num_processes,
    #                 args.log_dir,
    #                 #device,
    #                 allow_early_resets = False)

    env = test_game_base.test_game(lawn_num, args.reward_type, no_print=True)


    actor_critic = Policy(
        input_dims,
        n_actions,
        base_kwargs={'recurrent': args.recurrent_policy,
                     'depth_dim': args.depth_dim,
                     'hidden_size': args.hidden_size,
                     'hidden_num': args.hidden_num})
    actor_critic.to(device)

    print(f'-- Number of hyperparameters: {sum(p.numel() for p in actor_critic.parameters())}')


    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.ridge_lambda,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)



    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    sub_dir = 'go_explore_' + str(args.go_explore) + '/reward_function' + str(args.reward_type) + "/"

    log_dir = log_dir + '/' + env_name + '/' + sub_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    #run_num = len(current_num_files)

    run_num = run_id

    import logger
    logger = logger.logger(run_num, env = env, path=log_dir)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"''

    log_p1_name = log_dir + f'/score-{lawn_num}-{run_id}.png'
    log_p2_name = log_dir + f'/loss-{lawn_num}-{run_id}.png'

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    obs = env.state
    obs = obs.permute(2, 0, 1)  # oops, needed to change order
    input_dims_tensor = obs.shape
    input_dims_num = input_dims


    rollouts = RolloutStorage(args.num_steps,
                              input_dims_tensor,
                              input_dims_num,
                              n_actions,
                              actor_critic.recurrent_hidden_state_size)

    log_freq = 100

    env.reset()

    run_num = 1

    obs = env.state
    obs = obs.permute(2, 0, 1)  # oops, needed to change order
    obs_num = env.state_numericals
    rollouts.obs[0].copy_(obs)
    rollouts.obs_num[0].copy_(obs_num)
    rollouts.to(device)

    # episode_rewards = deque(maxlen=100)
    # episode_total_losses = deque(maxlen=100)
    # episode_entropies = deque(maxlen=100)
    # episode_values = deque(maxlen=100)
    # episode_actions = deque(maxlen=100)

    episode_rewards = []
    episode_perc_dones = []
    episode_total_losses = []
    episode_entropies = []
    episode_values = []
    episode_actions = []

    if args.go_explore is True:
        go_queue = []
    go_path = []

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps
    for j in range(num_updates):

        current_ep_reward = 0
        score = 0

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)


        for step in range(args.num_steps):

            with torch.no_grad():
                if len(go_path) == 0 and env.frames == 0 and args.go_explore is True:
                    if len(go_queue) > 0:
                        go_path = go_queue.pop(0)
                    elif run_num > 0 and run_num % args.go_explore_frequency == 0:
                        go_queue = get_go_paths(logger.filename)
                        go_path = go_queue.pop(0)
                    else:
                        # do runs as normal
                        pass


            while len(go_path) > 0:
                # Using go_explore


                with torch.no_grad():
                    action = torch.tensor([[go_path.pop(0)]]).to(device)
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs.to(device),
                        obs_num.to(device).float(),
                        recurrent_hidden_states,  # I THINK this is irrelevant (maybe not)
                        torch.FloatTensor([1.0]).to(device),  # done should never appear with go-explore
                        action = action)
                    obs, obs_num, reward, done, infos = env.step(action)
                    obs = obs.permute(2, 0, 1)  # oops, needed to change order

                    logger.log(action.item(), reward, 1)
                    score += reward
                    time_step += 1
                    current_ep_reward += reward

                    print_running_reward += current_ep_reward
                    print_running_episodes += 1

                    log_running_reward += current_ep_reward
                    log_running_episodes += 1

                if len(go_path) == 0:
                    masks = torch.FloatTensor([1.0])
                    bad_masks = torch.FloatTensor([1.0])
                    # # TODO: hacky.  fix!
                    # rollouts.obs[step] = obs
                    # rollouts.obs_num[step] = obs_num
                    # rollouts.recurrent_hidden_states[step] = recurrent_hidden_states
                    # rollouts.masks[step] = masks
                    # rollouts.bad_masks[step] = bad_masks
                    rollouts.insert(obs, obs_num, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.obs_num[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, obs_num, reward, done, infos = env.step(action)
            obs = obs.permute(2, 0, 1)  # oops, needed to change order

            with torch.no_grad():
                logger.log(action.item(), reward, 0)
                score += reward
            time_step +=1
            current_ep_reward += reward

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # masks = torch.FloatTensor(
            #     [[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks = torch.FloatTensor(
            #     [[0.0] if done_ else [0.0] for done_ in done])
            if done is True:
                masks = torch.FloatTensor([0.0])
                bad_masks = torch.FloatTensor([1.0])  # bad_masks is 0 only if there is a forced end run

                logger.write(score, run_num)
                episode_rewards.append(score)
                episode_perc_dones.append(env.perc_done)
                logger.write_rewards(run_num, score, env.perc_done)

                score = 0

                env.reset()

                run_num += 1

            else:
                masks = torch.FloatTensor([1.0])
                bad_masks = torch.FloatTensor([1.0])  # bad_masks is 0 only if there is a forced end run
            # bad_masks = torch.FloatTensor(
            #     [[0.0] if 'bad_transition' in info.keys() else [1.0]
            #      for info in infos])


            rollouts.insert(obs, obs_num, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.obs_num[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()
        #
        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)
        #
        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        with torch.no_grad():
            total_loss = action_loss + value_loss - args.entropy_coef * dist_entropy

            episode_total_losses.append(total_loss)
            episode_values.append(value_loss * args.value_loss_coef)
            episode_actions.append(action_loss)
            episode_entropies.append(-args.entropy_coef * dist_entropy)

            logger.write_loss(j, total_loss, value_loss * args.value_loss_coef, action_loss, -args.entropy_coef * dist_entropy)

        rollouts.after_update()



        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                #getattr(utils.get_vec_normalize(env), 'obs_rms', None)
                None
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_steps
            end = time.time()
            print(
                "Run_num {}, Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\nentropy: {:.4f}, value: {:.4f}, action: {:.4f}"
                    .format(run_id, j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards),
                            - args.entropy_coef * dist_entropy,
                            value_loss * args.value_loss_coef,
                            action_loss))
            x = [i + 1 for i in range(len(episode_rewards))]

            # if len(episode_rewards) > 100:
            #     plot_learning_curve(j, x, episode_rewards,
            #                         episode_perc_dones,
            #                         episode_total_losses,
            #                         episode_entropies,
            #                         episode_values,
            #                         episode_actions,
            #                         log_p1_name,
            #                         log_p2_name,
            #                         lawn_num,
            #                         run_id)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            #obs_rms = utils.get_vec_normalize(env).obs_rms
            obs_rms = None
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     1, eval_log_dir, device)


if __name__ == "__main__":
    main()