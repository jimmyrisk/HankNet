import retro
import torch
import numpy as np
import sys
import datetime
from pathlib import Path
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame, Discretizer
from metrics import MetricLogger
from hank import Hank

from retro import RetroEnv

from utils import memory_to_tensor, print_grid, cardinal_input, tracker, EpisodeLogger


LAWNMOWER_LOCATION = Path().parent.absolute()
retro.data.Integrations.add_custom_path(LAWNMOWER_LOCATION)

""" CHECK NVIDIA CUDA AVAILABILITY """

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

""" START ENVIRONMENT """



try:
    save_states = [f'lawn{x}.state' for x in range(10, 8, -1)]
    # env is a RetroEnv object https://github.com/openai/retro/blob/98fe0d328e1568836a46e5fce55a607f47a0c332/retro/retro_env.py

    env = RetroEnv(game='lawnmower',
                   state=save_states.pop(),  # pops off lawn1.state
                   inttype=retro.data.Integrations.ALL,
                   obs_type=retro.enums.Observations.RAM)
except FileNotFoundError:
    print(f"ERROR: lawnmower integration directory not found in the following location: {LAWNMOWER_LOCATION}")
    sys.exit()

""" OBSERVATION WRAPPERS """

action_space = [
    ['LEFT', 'B'],
    ['RIGHT', 'B'],
    ['DOWN', 'B'],
    ['UP', 'B']
]

env = Discretizer(env, combos=action_space)
#env = ResizeObservation(env, shape=84)
#env = GrayScaleObservation(env, keep_dim=False)
# env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

""" CHECKPOINT SAVING """

save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
checkpoint = Path('../checkpoints/2022-08-11T19-45-43/Hank_net_1.chkpt')
hank = Hank(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, env = env, checkpoint=checkpoint)

### Set these if you want it to begin learning anew with the current nn
#hank.exploration_rate_min = 0.1
hank.exploration_rate = 0.1
#hank.exploration_rate_decay = 0.9999975
#hank.exploration_rate_decay = 0.99999975

""" LOGGING """

logger = EpisodeLogger(save_dir)
lawn1_clear_ep = []

""" BEGIN TRAINING """

debug = False
episodes = 1000000
best_propane_points = 0  # aka best_cumulative_reward



for e in range(episodes):
    # State reset between runs
    # need to fill queue.  pick top left, top right, bottom left, or bottom right
    hist = tracker(env)

    # init tau

    hist.reset()
    env.render()

    done = False
    frame_count = 0

    while done is False:
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

        action = hank.act(init_state)

        # rewind 2f
        # push action.  All done inside logger.
        # Also checks if done.

        done = logger.step(action)



        # wait for state change
        #  - save until x, y changes.  save current states
        # record s_{t+1}, s_t, a_t, r_t

        # check if done


        if debug is True:
            #print(frame_since_act)

            ram = env.get_ram()
            ram_tensor = memory_to_tensor(ram)
            print_grid(ram_tensor)

            dir = input("Mow which direction?")

            action = int(int(cardinal_input(dir)))
            #input("Action made based on this state. Press any key to continue")
            #print(f"next_action={action}")






    logger.log_episode()

    """ SAVING & CHANGING LAWNS"""

    if e % 10 == 0:
        hank.save()
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
        if len(lawn1_clear_ep)>0:
            print(f"Lawn 1 cleared on episode {lawn1_clear_ep}")
        elif len(lawn1_clear_ep)>1:
            print(f"Lawn 1 cleared on episodes {lawn1_clear_ep}")

    if info["GRASS_LEFT"] < 1 and save_states:
        hank.save()
        lawn1_clear_ep.append(e)
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
        env.load_state(save_states.pop(), inttype = retro.data.Integrations.ALL)
    elif not save_states:
        sys.exit("HANK, YOU DID IT! YOU RAN THE GAUNTLET! LAWN 1-10 COMPLETE.")