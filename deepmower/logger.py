import csv

class logger:
    def __init__(self, run_id, env, path = "results/", ):
        self.actions = []
        self.rewards = []
        self.done = 0
        self.score = 0
        self.fuel = 0
        self.path = path + str(run_id) + ".csv"
        self.env = env

        with open(self.path, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = ["Run", "Path", "Rewards", "Score", "Fuel_Score", "Grass_Score", "Num_Fuel_Obtained",
                   "Amt_Fuel_Obtained", "End_Fuel", "Frames", "End_x", "End_y", "Perc_done", "Frames_Since_Fuel",
                   "Momentum Lost"]
            wr.writerow(row)
            csv_output.close()

    def log(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

    def write(self, score, run_num = 0):
        with open(self.path, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = [run_num,
                self.actions, self.rewards, score,
                   self.env.fuel_rewards,
                   self.env.grass_rewards,
                   self.env.fuel_counter,
                   self.env.amt_fuel_obtained,
                   self.env.fuel,
                   self.env.frames,
                   self.env.player_coord[0].item(),
                   self.env.player_coord[1].item(),
                   self.env.perc_done,
                   self.env.frames_since_fuel,
                   self.env.momentum_lost
                   ]
            wr.writerow(row)
            csv_output.close()
        self.actions = []
        self.rewards = []
        self.score = 0
        self.fuel = 0