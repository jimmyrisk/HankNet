import torch
from collections import deque
import numpy as np
from copy import deepcopy
import csv
import warnings


def csv_to_tensor(lawn_name):
    if isinstance(lawn_name, int):
        lawn_name = "lawn" + str(lawn_name)
    state = torch.zeros(13,32,8)
    with open('test_game/lawns/' + lawn_name + ".csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in spamreader:
            for j in range(32):
                entry = row[j]
                if entry == "P":
                    state[i,j,3] = 1.0
                    state[i, j, 7] = 1.0
                elif entry == "X":
                    state[i,j,4] = 1.0
                elif entry == "1":
                    state[i,j,0] = 1.0
                elif entry == "N":
                    state[i,j,6] = 1.0
                    state[i, j, 7] = 1.0
                elif entry == "G":
                    state[i,j,5] = 1.0
                elif entry == "R":
                    state[i,j,2] = 1.0
                elif entry == "F":
                    state[i,j,1] = 1.0
                elif entry == "0":
                    state[i,j,7] = 1.0
                else:
                    warnings.warn("Found " + entry + ", which is not a valid choice...")

            i += 1  # next row of lawn

    return state




