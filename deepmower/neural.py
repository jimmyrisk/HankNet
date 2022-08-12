from torch import nn
import copy

class NeuralNet(nn.Module):
    """
    # 0. (16x32) 1's and 0's corresponding to grass
    # 1. (16x32) 1's and 0's flower
    # 2. (16x32) 1's and 0's rock
    # 3. (16x32) 1's and 0's mower position
    # 4. (16x32) 1's and 0's impassable
    # 5. (16x32) 1's and 0's fuel
    # 6. (16x32) 1's and 0's upcoming fuel
    # 7. fuel%
    # 8. %done
    # 9. momentum (frames since input?)
    # is there a way to manually give reward based on momentum and grid structure, instead of using built in frames?
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()


        # playing around with size in neural
        self.online = self.nn = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(3,3,3), stride=4),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,1), stride=2),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1,3,1), stride=1),
            nn.ReLU(),
            nn.Flatten(1,4),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        # TODO: make this less hacky
        while len(input.shape) < 5:
            input = input.unsqueeze(dim=0)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)