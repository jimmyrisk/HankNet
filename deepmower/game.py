import torch
from collections import deque
import numpy as np
from copy import deepcopy

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


def cardinal_input(str):
    while str != "w" and str != "e" and str != "s" and str != "n":
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

