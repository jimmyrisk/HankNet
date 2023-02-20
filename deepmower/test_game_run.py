import torch
from collections import deque
import numpy as np
from copy import deepcopy
import csv
import warnings
from pathlib import Path
import load_lawn
import test_game_base
from hank import Hank
import sys
import datetime


from utils import memory_to_tensor, print_grid, cardinal_input, tracker, EpisodeLogger


lawn_num = 1
auto = False
sticky_probability = 0.2

env = test_game_base.test_game(lawn_num)

save_dir = Path("../test_game_checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
#checkpoint = Path('../test_game_checkpoints/2022-08-11T19-45-43/Hank_net_1.chkpt')
hank = Hank(state_dim=(4, 84, 84), action_dim=4, save_dir=save_dir, env = env)#, checkpoint=checkpoint)

### Set these if you want it to begin learning anew with the current nn
#hank.exploration_rate_min = 0.1
#hank.exploration_rate = 0.1
#hank.exploration_rate_decay = 0.9999975
#hank.exploration_rate_decay = 0.99999975

""" LOGGING """

#logger = EpisodeLogger(save_dir)
lawn1_clear_ep = []

""" BEGIN TRAINING """

debug = False
episodes = 1000000
best_propane_points = 0  # aka best_cumulative_reward


for e in range(episodes):
    # State reset between runs
    # need to fill queue.  pick top left, top right, bottom left, or bottom right
    #hist = tracker(env)

    # init tau

    state = env.init_state
    env.reset()

    #hist.reset()

    done = False
    frame_count = 0

    while done is False:

        if auto is True:
            action = hank.act(state)
        else:
            env.print()

            dir = input("Mow which direction?")
            action = cardinal_input(dir)
            if action == "mpc":
                n_ahead = 5
                print("SAVING STATE...")
                save_state = env.save_state()
                print("GENERATING RANDOM SET OF ACTIONS:")
                # TODO: implement sticky
                actions = np.random.choice([0,1,2,3], n_ahead)
                for action in actions:
                    done = env.step(action)
                    env.print()
                print("LOADING STATE...")
                env.load_state(save_state)

            else:
                _, _, done, _ = env.step(action, save = True)



        # TODO:
        # - set up numeric things to keep track of in state
        # - finish EpisodeLogger along with reward
        # - try to generate random trajectories?
        # - set up both NN's


        # first action is always EAST

        # wait for state change
        #  - save until x, y changes.  save current states
        # record s_{t+1}, s_t, a_t, r_t


        # get action from M_phi(s_t, a_t)
        # - look ahead horizon H=10
        # - find path that maximizes reward






        # rewind 2f
        # push action.  All done inside logger.
        # Also checks if done.

        # done = logger.step(action)



        # wait for state change
        #  - save until x, y changes.  save current states
        # record s_{t+1}, s_t, a_t, r_t

        # check if done


        if debug is True:
            #print(frame_since_act)

            ram = env.get_ram()
            ram_tensor = memory_to_tensor(ram)



            #input("Action made based on this state. Press any key to continue")
            #print(f"next_action={action}")






    #logger.log_episode()

    """ SAVING & CHANGING LAWNS"""

    # if e % 10 == 0:
    #     hank.save()
    #     logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
    #     if len(lawn1_clear_ep)>0:
    #         print(f"Lawn 1 cleared on episode {lawn1_clear_ep}")
    #     elif len(lawn1_clear_ep)>1:
    #         print(f"Lawn 1 cleared on episodes {lawn1_clear_ep}")
    #
    # if info["GRASS_LEFT"] < 1 and save_states:
    #     hank.save()
    #     lawn1_clear_ep.append(e)
    #     logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
    #     env.load_state(save_states.pop(), inttype = retro.data.Integrations.ALL)
    # elif not save_states:
    #     sys.exit("HANK, YOU DID IT! YOU RAN THE GAUNTLET! LAWN 1-10 COMPLETE.")
    #
