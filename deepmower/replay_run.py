import copy
import glob
import os
import time
from collections import deque
from utils import direction_dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import test_game_base

import pandas as pd


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

args = get_args()






def main():


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")




    verbose = True


    non_defaults = input('Non-defaults? ')
    if non_defaults:
        lawn_num = input('Lawn Number: ') or "22"
        reward_type = input('Reward Type: ') or "2"
        go_explore = input('Go_Explore?: ') or "True"
        #go_explore = True
        run_id = input('Run id: ') or "10001"
        sort_by = input('Sort By (one of "Perc_done", "Score", "Frames"): ') or 'Frames'
        top_n = input('Top n?: ') or 10
        filter = input('Filter Perc_done >= : ') or 100
    else:
        lawn_num = 22
        reward_type = 2
        go_explore = "True"
        run_id = 10001
        sort_by = 'Frames'
        top_n = 10
        filter = 100





    env = test_game_base.test_game(f"lawn{lawn_num}", reward_type, device=device, no_print=True)
    env.reset()

    score = 0

    run = True

    while run:




        log_dir = "PPO_logs"
        sub_dir = 'go_explore_' + str(go_explore) + '/reward_function' + str(reward_type) + "/"
        env_name = f"lawn{lawn_num}"
        log_dir = log_dir + '/' + env_name + '/' + sub_dir


        #### create new log file for each run
        log_f_name = log_dir + '/' + str(run_id) + ".csv"''

        #%%

        run_df = pd.read_csv(log_f_name)
        display_df = run_df[run_df['Perc_done'] >= filter].sort_values(sort_by, ascending=True).iloc[:top_n]
        print(display_df[['Perc_done', 'Frames', 'Score']])


        run_idx = int(input('Run Index (put -1 to go back): ')) or -1
        if run_idx == -1:
            break

        path = run_df['Path'].iloc[run_idx]
        path = path.strip('][').split(', ')
        path = list(map(int, path))

        while len(path) > 0:
            action = torch.tensor([[path.pop(0)]]).to(device)
            _, _, reward, _, _ = env.step(action)

            score += reward

            if verbose:
                print(f'Action: {direction_dict[action.item()]}  --  Perc Done: {env.perc_done:.2f}  --  Reward: {reward:.2f}  --  Score: {score:.2f}  --  Frames: {env.frames}')

        print(f'Final Score: {score:.2f}  --  Final Frames: {env.frames}')
        env.reset()
        score = 0





if __name__ == "__main__":
    main()