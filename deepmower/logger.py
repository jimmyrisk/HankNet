import csv

class logger:
    def __init__(self, run_id, env, path = "results/", ):
        self.actions = []
        self.rewards = []
        self.copied = []
        self.done = 0
        self.fuel = 0
        self.path = path
        self.filename = path + str(run_id) + ".csv"
        self.filename_loss = path + str(run_id) + "_loss_plots.csv"
        self.filename_rewards = path + str(run_id) + "_reward_plots.csv"
        self.env = env

        with open(self.filename, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = ["Run", "Deterministic", "Path", "Rewards", "Score", "Fuel_Score", "Grass_Score", "Num_Fuel_Obtained",
                   "Amt_Fuel_Obtained", "End_Fuel", "Frames", "End_x", "End_y", "Perc_done", "Frames_Since_Fuel",
                   "Momentum Lost", "Fuel_Manhattan", "Go_Explore_Copied"]
            wr.writerow(row)
            csv_output.close()

        with open(self.filename_loss, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = ["iter", 'total_loss', 'value_loss', 'action_loss', 'entropy_loss']
            wr.writerow(row)
            csv_output.close()


    def log(self, action, reward, copied):
        self.actions.append(action)
        self.rewards.append(reward)
        self.copied.append(copied)

    def write(self, score, deterministic, run_num = 0):
        with open(self.filename, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = [run_num, deterministic,
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
                   self.env.momentum_lost,
                   self.env.fuel_manhattan.item(),
                   self.copied
                   ]
            wr.writerow(row)
            csv_output.close()
        self.actions = []
        self.rewards = []
        self.copied = []
        self.fuel = 0



    def write_loss(self, iter, total_loss, value_loss, action_loss, entropy_loss):
        with open(self.filename_loss, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = [iter,
                   total_loss,
                   value_loss,
                   action_loss,
                   entropy_loss
                   ]
            wr.writerow(row)
            csv_output.close()

    def write_rewards(self, iter, reward, perc_done):
        with open(self.filename_rewards, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = [iter,
                   reward,
                   perc_done
                   ]
            wr.writerow(row)
            csv_output.close()