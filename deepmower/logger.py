import csv

class logger:
    def __init__(self, run_id, path = "results/", ):
        self.actions = []
        self.rewards = []
        self.done = 0
        self.score = 0
        self.fuel = 0
        self.path = path + str(run_id) + ".csv"

        with open(self.path, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = ["Path", "Rewards", "Score", "Fuel"]
            wr.writerow(row)
            csv_output.close()

    def log(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

    def write(self, score):
        with open(self.path, mode='a', newline='') as csv_output:
            wr = csv.writer(csv_output)
            row = [self.actions, self.rewards, score, self.fuel]
            wr.writerow(row)
            csv_output.close()
        self.actions = []
        self.rewards = []
        self.score = 0
        self.fuel = 0