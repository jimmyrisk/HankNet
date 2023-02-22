import torch
from collections import deque
import numpy as np
import copy
import csv
import warnings

import load_lawn

from utils import memory_to_tensor, print_grid, cardinal_input, tracker, EpisodeLogger

# https://github.com/aremath/sm_rando/blob/master/abstraction_validation/go_explore.py

#
# idea:
#  - add sampling value based on mower's end x+y position AND lawn mowed
#  - or, # mowed / manhattan distance]
#  - e.g. look at top #100 out of last 1000 trajectories (prob weighed).  pick 10 of them based on extremeties
#  - sample 10 of them, uniformly selecting 10 spots from which to begin new

# atlas: goes from cell (low dim rep of save state) to set of actual ram states
#  - doesn't matter for bare bones version
# cell_selector:
# for steps
#   select a cell at random (weighted eventually.  e.g. cut the most grass, or has the most fuel, etc.)
#   select random state from
#   select random action
#   set state.  run actions for some amount of frames (like sticky?)
#   get cell
#   if cell not ok, then discard it
#   if cell ok, then add to atlas
#     but only if it's not completely duplicated?  (but
#   can resimulate a path through
#   gets a set of paths with a graph
#   even though it's not policy-based, it can find based on a policy
# SELECT CELLS SHORTER IN FRAMES ELAPSED
#
# AREMATH line 118 is learning a policy to get to that state
#
# INCLUDE MANHATTAN DISTANCE TO FUEL AS REWARD / HEURISTIC
#
# instead of "load_state", have neural network tell you actions to get to that state
#
# when doing regular go-explore, you know for a given cell, the paths you take to get there
# want RL agent to visit same cells
# when you want to do "emu.set_state(state)" we just want RL agent to get to same cell
# do roll-outs, time limit, if it doesn't get a certain amount of reward, then...
# then you're getting reward based on whether you visited cells that are on the path to where you're trying to go
# more reward for going to more of those cells
# sampling from probability distribution to get to cells.  reward for that
#
# 1. implement go-explore without policy, and see what you get
# gives a window as to how good the heuristic is.  SNR ratio us huge on most RL algorithms


# We wrap the action
# space into 5 discrete joypad actions, none, walk right, jump
# right, run right and hyper jump right. We follow [45] to add
# a sticky action wrapper that repeats the last action with a
# probability of 20%. Besides this, we follow add the standard
# wrapper as in past work [1]

class test_game:
    def __init__(self, lawn, no_print = False):
        self.lawn = lawn
        self.init_state = load_lawn.csv_to_tensor(self.lawn)
        #self.state = self.init_state
        self.no_print = no_print

        self.flower_penalty = 10
        self.rock_penalty = 30



        # initialize
        self.reset()

    # 0. (16x32) 1's and 0's corresponding to grass
    # 1. (16x32) 1's and 0's flower
    # 2. (16x32) 1's and 0's rock
    # 3. (16x32) 1's and 0's mower position
    # 4. (16x32) 1's and 0's impassable
    # 5. (16x32) 1's and 0's fuel
    # 6. (16x32) 1's and 0's upcoming fuel

    def print(self):
        if self.dir == 1:
            facing = "EAST"
        elif self.dir == 2:
            facing = "SOUTH"
        elif self.dir == 3:
            facing = "NORTH"
        else:
            facing = "WEST"

        if self.no_print is False:
            print("--------------------------------")
            print(f"FRAMES: {self.frames}  --  FUEL: {self.fuel}  --  DONE: {self.perc_done}% ")
            print(f"MOMENTUM: {self.momentum}  --  FACING: {facing}")
            print_grid(self.state)


    def reset(self):
        self.state = copy.deepcopy(self.init_state)
        self.frames = 0
        self.player_coord = (self.state[:, :, 3] == 1.0).nonzero()[0]
        self.momentum = 0
        self.dir = 1  # east
        self.fuel = 100
        self.total_grass = self.state[:, :, 0].sum()
        self.mowed = torch.tensor([0])
        self.perc_done = np.round(self.mowed.item() / self.total_grass.item()*100,2)
        self.east = self.dir == 1
        self.west = self.dir == 0
        self.north = self.dir == 4
        self.south = self.dir == 3
        urgency_to_oof = 1 / (1 + self.fuel)
        urgency_to_finish = 1 / (101 - self.perc_done)

        self.state_numericals = torch.tensor(
            [self.frames / 1000, self.fuel / 100, self.momentum / 4, self.perc_done / 100,
             self.east, self.west, self.north, self.south, self.mowed / 100,
             urgency_to_oof,
             urgency_to_finish
             ]
        )
        self.actions = []
        self.states = []
        self.numerical_states = []

    def save_state(self):
        numericals = [self.frames, self.fuel, self.momentum, self.perc_done, self.dir, self.mowed]

        universe = {
            "state": copy.deepcopy(self.state),
            "numericals": numericals,
            'player_coord': copy.deepcopy(self.player_coord)
        }

        return universe

    def load_state(self, universe):
        self.state = universe['state']
        self.player_coord = universe['player_coord']
        self.frames, self.fuel, self.momentum, self.perc_done, self.dir, self.mowed = universe['numericals']

    def step(self, action, save=False):
        # for saving tau
        if save is True:
            self.actions.append(action)
            self.states.append(copy.deepcopy(self.state))
            numericals = [self.frames, self.fuel, self.momentum, self.perc_done]
            self.numerical_states.append(numericals)

        reward = 0.0
        #print("updating...")
        # update
        self.frames = self.frames + 6
        self.fuel = self.fuel - 1

        if self.dir == action:
            self.momentum = self.momentum + 1

            # if gone 4 squares in a row
            if self.momentum == 3:
                self.frames = self.frames - 1
                self.momentum = 0
        else:
            self.momentum = 0

        self.dir = action

        if action == 0:  # west
            coord_shift = torch.tensor([0, -1])

        elif action == 1:  # east
            coord_shift = torch.tensor([0, 1])

        elif action == 2:  # south
            coord_shift = torch.tensor([1, 0])

        else:  # north
            coord_shift = torch.tensor([-1, 0])

        coord = self.player_coord
        new_coord = coord + coord_shift

        if self.state[new_coord[0], new_coord[1], 0] == 1.0:
            # grass
            self.mowed = self.mowed + 1
            self.perc_done = np.round(self.mowed.item() / self.total_grass.item() * 100, 2)
            self.state[new_coord[0], new_coord[1], 0] = 0.0
            #reward += 1 / np.sqrt(self.frames)
            reward += 1

            if self.perc_done == 100:
                reward += 100 / np.sqrt(self.frames)

        elif self.state[new_coord[0], new_coord[1], 1] == 1.0:
            # flower
            self.fuel = self.fuel - self.flower_penalty
            self.state[new_coord[0], new_coord[1], 1] = 0.0



        elif self.state[new_coord[0], new_coord[1], 2] == 1.0:
            # rock
            self.fuel = self.fuel - self.rock_penalty

        elif self.state[new_coord[0], new_coord[1], 3] == 1.0:
            # Player
            raise NotImplementedError("Trying to move player on top of player")

        elif self.state[new_coord[0], new_coord[1], 4] == 1.0:
            # impassable
            new_coord = self.player_coord
            self.momentum = 0

        elif self.state[new_coord[0], new_coord[1], 5] == 1.0:
            # fuel
            self.fuel = 100.0
            self.state[new_coord[0], new_coord[1], 5] = 0.0
            print("TODO: Set Next Fuel Spawn")
            reward += 0.5

        elif self.state[new_coord[0], new_coord[1], 6] == 1.0:
            # next fuel
            pass

        # set fuel to 0
        if self.fuel < 0:
            self.fuel = 0.0

        self.state[coord[0], coord[1], 3] = 0.0
        self.state[new_coord[0], new_coord[1], 3] = 1.0
        self.player_coord = new_coord

        info = None
        self.east = self.dir == 1
        self.west = self.dir == 0
        self.north = self.dir == 4
        self.south = self.dir == 3

        urgency_to_oof = 1/(1+self.fuel)
        urgency_to_finish = 1/(101-self.perc_done)

        self.state_numericals = torch.tensor(
            [self.frames / 1000, self.fuel / 100, self.momentum / 4, self.perc_done / 100,
             self.east, self.west, self.north, self.south, self.mowed / 100,
             urgency_to_oof,
             urgency_to_finish
             ]
        )

        #return self.save_state(), reward, self.check_done(), info
        return self.state, self.state_numericals, reward, self.check_done(), info


    def check_done(self):
        if self.mowed == self.total_grass:
            done = True
            if self.no_print is False:
                print("--------------------------------")
                print("-- DONE!")
                print(f"-- FRAMES: {self.frames}  --  FUEL: {self.fuel}  --  DONE: {self.perc_done}% ")
                print("--------------------------------")
        elif self.fuel < 1:
            done = True
            if self.no_print is False:
                print("--------------------------------")
                print("-- OUT OF FUEL!")
                print(f"-- FRAMES: {self.frames}  --  FUEL: {self.fuel}  --  DONE: {self.perc_done}% ")
                print("--------------------------------")
        else:
            done = False

        return done

    def record(self):
        tau = [self.actions, self.states, self.numerical_states]
        return tau

