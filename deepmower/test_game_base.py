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
    def __init__(self, lawn, reward_type = 1, device = 'cpu', no_print = False, fuel_seed = 2147483647):
        self.lawn = lawn
        self.init_state = load_lawn.csv_to_tensor(self.lawn)
        self.total_grass = self.init_state[:, :, 0].sum()
        #self.state = self.init_state
        self.no_print = no_print

        self.reward_type = reward_type

        self.flower_penalty = 10
        self.rock_penalty = 20

        self.fuel_reward_max = 200
        self.frame_reward_max = 1000

        self.fuel_seed = fuel_seed
        self.device = device
        self.fuel_rng = torch.Generator(device=self.device)






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
        self.fuel_counter = 0

        self.fuel_rewards = 0
        self.grass_rewards = 0
        self.amt_fuel_obtained = 0

        self.frames_since_fuel = 0

        self.momentum = 0
        self.momentum_lost = 0
        self.dir = 1  # east
        self.fuel = 60
        self.fuel_rng.manual_seed(self.fuel_seed)
        self.mowed = torch.tensor([0])
        self.perc_done = np.round(self.mowed.item() / self.total_grass.item()*100,2)
        self.east = self.dir == 1
        self.west = self.dir == 0
        self.north = self.dir == 3
        self.south = self.dir == 2



        # if we start with a fuel
        if self.state[:, :, 5].sum() == 1:
            self.no_fuel = 0
            self.fuel_coord = (self.state[:, :, 5] == 1.0).nonzero()[0]
        # if we instead start with next fuel
        else:
            self.no_fuel = 1
            self.fuel_coord = (self.state[:, :, 6] == 1.0).nonzero()[0]
            self.frames_since_fuel = 0

        urgency_to_oof = 1 / (61 - self.fuel)
        urgency_to_finish = 1 / (101 - self.perc_done)

        player_y = 13 - self.player_coord[0]
        player_x = 32 - self.player_coord[1]

        fuel_y = 13 - self.fuel_coord[0]
        fuel_x = 32 - self.fuel_coord[1]

        self.fuel_manhattan = abs(player_x - fuel_x) + abs(player_y - fuel_y)

        self.state_numericals = torch.tensor(
            [self.frames / 1000, self.fuel / 100, self.momentum / 4, self.perc_done / 100,
             self.east, self.west, self.north, self.south, self.mowed / 100,
             urgency_to_oof,
             urgency_to_finish,
             player_x / 32,
             player_y / 13,
             fuel_x / 32,
             fuel_y / 13,
             self.fuel_manhattan / (32+13),
             self.fuel_counter / 10
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

        #reward = -0.1
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
            self.momentum_lost += self.momentum
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
            self.state[new_coord[0], new_coord[1], 7] = 1.0

            if self.reward_type == 0:
                reward += 10
            else:
                reward += (1 + self.perc_done)
            self.grass_rewards += reward
            #reward += 1

            if self.perc_done == 100:
                if self.reward_type == 0:
                    done_reward = (
                            10 +
                            10 / (self.frames - 5.5*self.total_grass.item()) * 10 * self.total_grass.item() +
                            np.maximum(self.fuel_reward_max - self.fuel_rewards, 0)
                    )
                else:
                    done_reward = (
                            (1 + self.perc_done) +
                            (1 + self.perc_done) / (self.frames - 5.5*self.total_grass.item()) * 10 * self.total_grass.item() +
                            np.maximum(self.fuel_reward_max - self.fuel_rewards, 0)
                    )

                if self.no_print is False:
                    print(
                        f"-- DONE!: reward: {done_reward:.2f}  --  fuel pickups: {self.fuel_counter}  --  %: {self.perc_done}  --  fr: {self.frames}  --  g-rew: {reward:.2f}")

                reward += done_reward

        elif self.state[new_coord[0], new_coord[1], 1] == 1.0:
            # flower
            self.fuel = self.fuel - self.flower_penalty
            self.state[new_coord[0], new_coord[1], 1] = 0.0
            self.state[new_coord[0], new_coord[1], 7] = 1.0



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

            self.state[new_coord[0], new_coord[1], 5] = 0.0

            # set next fuel spawn
            fuel_choice_num = (self.state[:, :, 7] == 1.0).nonzero().size()[0]
            fuel_choice_idx = torch.randperm(fuel_choice_num, generator = self.fuel_rng, device = self.device)[0]
            self.fuel_coord = (self.state[:, :, 7] == 1.0).nonzero()[fuel_choice_idx]
            self.state[self.fuel_coord[0], self.fuel_coord[1], 6] = 1.0


            self.fuel_counter += 1
            self.amt_fuel_obtained += 60.0 - self.fuel

            #fuel_reward = (3 / self.fuel_counter) * (1 + self.perc_done) / np.sqrt(self.frames) * 60 / (1+self.fuel)

            if self.reward_type == 3:
                fuel_reward = 0
            elif self.reward_type == 0:
                fuel_reward = 4 / self.fuel_counter * 20
            else:
                fuel_reward = 4 / self.fuel_counter * (1 + self.perc_done)

            if self.fuel_rewards + fuel_reward > self.fuel_reward_max:
                fuel_reward = max(self.fuel_reward_max - self.fuel_rewards, 0)

            self.fuel_rewards += fuel_reward

            reward += fuel_reward

            self.no_fuel = 1
            self.frames_since_fuel = 0

            self.fuel = 60.0

            if self.no_print is False:
                g_rew = (1 + self.perc_done)
                print(f"fuel pickup: {self.fuel_counter}  --  reward: {fuel_reward:.2f}  --  %: {self.perc_done}  --  fr: {self.frames}  --  g-rew: {g_rew:.2f}")

        elif self.state[new_coord[0], new_coord[1], 6] == 1.0:
            # next fuel
            pass

        if self.no_fuel == 1:
            self.frames_since_fuel += 1
            if self.frames_since_fuel == 24:
                self.state[self.fuel_coord[0], self.fuel_coord[1], 5] = 1.0
                self.state[:, :, 6] = 0.0
                self.no_fuel = 0

        # set fuel to 0
        if self.fuel <= 0 and self.perc_done < 100:
            self.fuel = 0.0

            if self.reward_type == 1:
                pass
            else:
                reward -= 200

        self.state[coord[0], coord[1], 3] = 0.0
        self.state[new_coord[0], new_coord[1], 3] = 1.0
        self.player_coord = new_coord

        info = None
        self.east = self.dir == 1
        self.west = self.dir == 0
        self.north = self.dir == 3
        self.south = self.dir == 2

        urgency_to_oof = 1/(61 - self.fuel)
        urgency_to_finish = 1/(101-self.perc_done)
        player_y = 13 - self.player_coord[0]
        player_x = 32 - self.player_coord[1]

        fuel_y = 13 - self.fuel_coord[0]
        fuel_x = 32 - self.fuel_coord[1]

        self.fuel_manhattan = abs(player_x - fuel_x) + abs(player_y - fuel_y)


        # print(f"player coord: [{player_x.item()}, {player_y.item()}")
        # print(f" fuel  coord: [{fuel_x.item()}, {fuel_y.item()}")
        # print(f"manhattan   : {fuel_manhattan}")

        self.state_numericals = torch.tensor(
            [self.frames / 1000, self.fuel / 100, self.momentum / 4, self.perc_done / 100,
             self.east, self.west, self.north, self.south, self.mowed / 100,
             urgency_to_oof,
             urgency_to_finish,
             player_x / 32,
             player_y / 13,
             fuel_x / 32,
             fuel_y / 13,
             self.fuel_manhattan / (32+13),
             self.fuel_counter / 10
             ]
        )

        # scale if needed
        reward = reward / 100

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

