import torch
from collections import deque
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# 0. (16x32) 1's and 0's corresponding to grass
# 1. (16x32) 1's and 0's flower
# 2. (16x32) 1's and 0's rock
# 3. (16x32) 1's and 0's mower position
# 4. (16x32) 1's and 0's impassable
# 5. (16x32) 1's and 0's fuel
# 6. (16x32) 1's and 0's upcoming fuel
# 7. fuel%
# 8. %done
# 9. momentum

def print_grid(tensor):
    print("-"*32)
    for i in range(13):
        for j in range(32):

            # player
            if tensor[i,j,3] == 1.0:
                print("P", end='')

            # impassable
            elif tensor[i,j,4] == 1.0:
                print("X", end='')

            # grass
            elif tensor[i,j,0] == 1.0:
                print("1", end='')

            # next
            elif tensor[i,j,6] == 1.0:
                print("N", end='')

            # gas
            elif tensor[i,j,5] == 1.0:
                print("G", end='')

            # rock
            elif tensor[i,j,2] == 1.0:
                print("R", end='')

            # flower
            elif tensor[i,j,1] == 1.0:
                print("F", end='')

            # else
            else:
                print("0", end='')

        print("")

def state_to_tensor(state):
    frame_0 = memory_to_tensor(state[0])
    frame_1 = memory_to_tensor(state[1])
    frame_2 = memory_to_tensor(state[2])
    frame_3 = memory_to_tensor(state[3])

    tensor = torch.stack((frame_0, frame_1, frame_2, frame_3)).double()
    return(tensor)


def memory_to_tensor(memory):
    state = torch.zeros(13,32,7).double()

    # player location
    player_i = memory[0x00E8] - 2
    player_j = memory[0x00EA]  # -2(?)

    state[player_i, player_j, 3] = 1.0

    # current fuel:
    if memory[0x00C1] == 0:
        pass  # no fuel
    else:
        curr_fuel_i = memory[0x00C0] - 2  # CurrFuelYAd
        curr_fuel_j = memory[0x00C1]      # CurrFuelXAd
        state[curr_fuel_i, curr_fuel_j, 5] = 1.0

    # next fuel
    next_fuel_i = memory[0x00B7] - 2      # NextFuelYAd
    next_fuel_j = memory[0x00B8]          # NextFuelXAd

    state[next_fuel_i, next_fuel_j, 6] = 1.0

    # borders
    state[:, 0, 4] = 1.0
    state[:, 31, 4] = 1.0
    state[0, :, 4] = 1.0
    state[12, :, 4] = 1.0

    addr = 0x340

    for i in range(0,13):

        for j in range(0,32):
            if memory[addr] <= 0x09:
                # impassable
                state[i,j,4] = 1.0

            elif memory[addr] <= 0x0D:
                # mowed
                pass

            elif memory[addr] <= 0x11:
                # grass
                state[i,j,0] = 1.0

            elif memory[addr] == 0x12:
                # flower
                state[i,j,1] = 1.0

            elif memory[addr] == 0x14:
                # rock
                state[i,j,2] = 1.0

            addr += 1

    return state

def plot_learning_curve(j, x, scores, total_loss, entropy, value, action, figure_file_score, figure_file_loss):
    running_avg_score = np.zeros(len(x))

    for i in range(len(running_avg_score)):
        running_avg_score[i] = np.mean(scores[max(0, i - 100):(i + 1)])



    fig, ax1 = plt.subplots()
    ax1.set_ylabel('score', color='red')
    ax1.plot(x, running_avg_score, color='red', label = 'Score')
    ax1.tick_params(axis='y', labelcolor='red')

    ax1.legend()

    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file_score)
    plt.close()

    if j > 10:
        running_avg_total_loss = np.zeros(j)
        running_avg_entropy = np.zeros(j)
        running_avg_value = np.zeros(j)
        running_avg_action = np.zeros(j)
        y = [i + 1 for i in range(len(total_loss))]
        for i in range(len(running_avg_entropy)):
            running_avg_total_loss[i] = np.mean(total_loss[max(0, i - 10):(i + 1)])
            running_avg_entropy[i] = np.mean(entropy[max(0, i - 10):(i + 1)])
            running_avg_value[i] = np.mean(value[max(0, i - 10):(i + 1)])
            running_avg_action[i] = np.mean(action[max(0, i - 10):(i + 1)])

        fig, ax2 = plt.subplots()
        ax2.set_ylabel('loss', color='blue')
        ax2.plot(y[1:], running_avg_total_loss, label='Total Loss')
        ax2.plot(y[1:], running_avg_entropy, label='Entropy')
        ax2.plot(y[1:], running_avg_value, label='Value')
        ax2.plot(y[1:], running_avg_action, label='Action')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax2.legend()


        plt.title('Running average of previous 10 losses')
        plt.savefig(figure_file_loss)
        plt.close()












def cardinal_input(str):
    while str != "w" and str != "e" and str != "s" and str != "n" and str != "mpc":
        print("error.  input" + str + "not in one of 'w', 'e', 's', 'n'.\n")
        str = input("Mow which direction?")
    if str == "w":
        return 0
    elif str == "e":
        return 1
    elif str == "s":
        return 2
    elif str == "n":
        return 3
    elif str == "mpc":
        return "mpc"


# transform_observation wrapper
# https://github.com/openai/gym/blob/master/gym/wrappers/transform_observation.py
#
#
# get_tile:
# - takes in x and y (in map_width and map_height)
#   - tile_offset = ...
#   - tile_id = tile_data[tile_offset]
# - returns tile_id
#
# row 1 (1+1):
# 000360
#
# row 5 (5+1):
# 0003E0
#
#
# 00-09: impassable
# 0A-0D: mowed
# 0E-11: grass
# 12: flower
# 13: mowed flower
# 14: rock
# 2E-2F: fuel

class tracker:
    def __init__(self, env):
        self.env = env
        self.momentum = torch.tensor([0.0])

    def reset(self):
        # For dynamics model
        self.action_hist = deque(maxlen=4)
        self.state_hist = deque(maxlen=4)
        self.shift_hist = deque(maxlen=3)
        self.coord_hist = deque(maxlen=4)

        self.env.reset()
        self.init_ram = self.env.get_ram()
        self.init_tensor = memory_to_tensor(self.init_ram)
        self.init_info = self.env.data.lookup_all()
        tensor = memory_to_tensor(self.env.get_ram())
        self.coord_init = (tensor[:,:,3] == 1.0).nonzero()[0]

        # For dynamics model -- this doesn't actually start the mower at a different spot.
        pos = np.random.randint(3)
        if pos == 0:
            self.init_top()
        elif pos == 1:
            self.init_bot()
        else:
            self.init_right()

    def init_top(self):
        tensor = self.init_tensor
        tensor[self.coord_init[0], self.coord_init[1], 3] = 0.0

        coord = self.coord_init
        coord[0] = coord[0] - 1
        self.coord_hist.append(coord)

        tensor[coord[0], coord[1], 3] = 1.0
        self.state_hist.append(tensor)
        self.move_down_init()
        self.move_up_init()
        self.move_down_init()

    def init_bot(self):
        tensor = self.init_tensor
        tensor[self.coord_init[0], self.coord_init[1], 3] = 0.0

        coord = self.coord_init
        coord[0] = coord[0] + 1
        self.coord_hist.append(coord)

        tensor[coord[0], coord[1], 3] = 1.0
        self.state_hist.append(tensor)
        self.move_up_init()
        self.move_down_init()
        self.move_up_init()


    def init_right(self):
        tensor = self.init_tensor
        tensor[self.coord_init[0], self.coord_init[1], 3] = 0.0

        coord = self.coord_init
        coord[1] = coord[1] + 1
        self.coord_hist.append(coord)

        tensor[coord[0], coord[1], 3] = 1.0
        self.state_hist.append(tensor)
        self.move_left_init()
        self.move_right_init()
        self.move_left_init()

    # 0: left
    # 1: right
    # 2: down
    # 3: up
    def move_left_init(self):
        # move left
        coord_shift = torch.tensor([0, -1])

        # old info
        coord = self.coord_hist[-1]
        tensor = self.state_hist[-1]
        tensor[coord[0], coord[1], 3] = 0.0

        # update old info
        coord = coord + coord_shift
        tensor[coord[0], coord[1], 3] = 1.0

        # append
        self.state_hist.append(tensor)
        self.coord_hist.append(coord)
        self.action_hist.append(torch.tensor([1, 0, 0, 0]))

        shift = self.state_hist[-1] - self.state_hist[-2]
        self.shift_hist.append(shift)

    def move_right_init(self):
        # move right
        coord_shift = torch.tensor([0, 1])

        # old info
        coord = self.coord_hist[-1]
        tensor = self.state_hist[-1]
        tensor[coord[0], coord[1], 3] = 0.0

        # update old info
        coord = coord + coord_shift
        tensor[coord[0], coord[1], 3] = 1.0

        # append
        self.state_hist.append(tensor)
        self.coord_hist.append(coord)
        self.action_hist.append(torch.tensor([0, 1, 0, 0]))

        shift = self.state_hist[-1] - self.state_hist[-2]
        self.shift_hist.append(shift)

    def move_down_init(self):
        # move down
        coord_shift = torch.tensor([1, 0])

        # old info
        coord = self.coord_hist[-1]
        tensor = self.state_hist[-1]
        tensor[coord[0],coord[1],3] = 0.0

        # update old info
        coord = coord + coord_shift
        tensor[coord[0], coord[1], 3] = 1.0

        # append
        self.state_hist.append(tensor)
        self.coord_hist.append(coord)
        self.action_hist.append(torch.tensor([0,0,1,0]))

        shift = self.state_hist[-1] - self.state_hist[-2]
        self.shift_hist.append(shift)

    def move_up_init(self):
        # move up
        coord_shift = torch.tensor([-1, 0])

        # old info
        coord = self.coord_hist[-1]
        tensor = self.state_hist[-1]
        tensor[coord[0], coord[1], 3] = 0.0

        # update old info
        coord = coord + coord_shift
        tensor[coord[0], coord[1], 3] = 1.0

        # append
        self.state_hist.append(tensor)
        self.coord_hist.append(coord)
        self.action_hist.append(torch.tensor([0, 0, 0, 1]))

        shift = self.state_hist[-1] - self.state_hist[-2]
        self.shift_hist.append(shift)

    def step(self, action):
        # env.step() until x,y change
        # update action_hist
        self.action_hist.append(action)

        # update momentum
        if self.action_hist[-1] == self.action_hist[-2]:
            self.momentum = self.momentum + 1.0
        else:
            self.momentum = torch.tensor([0.0])

        # update state_hist


        # update shift_hist
        shift = self.state_hist[-1] - self.state_hist[-2]
        self.shift_hist.append(shift)





class EpisodeLogger:
    def __init__(self, save_dir, tracker):
        self.save_dir = save_dir
        self.tracker = tracker

        # for dynamics model
        self.actions_hist = []
        self.shifts_hist = []
        self.state_hist = []

    def update(self):
        self.actions_hist.append(deepcopy(tracker.action_hist))
        self.shifts_hist.append(deepcopy(tracker.shift_hist))
        self.state_hist.append(deepcopy(tracker.state_hist[-1]))

    def step(self, action):
        self.tracker.step(action)
        self.update()

        done = False
        return done

    def log_episode(self):
        pass


