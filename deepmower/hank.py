import torch
import random, numpy as np
from neural import NeuralNet
from collections import deque
from utils import memory_to_tensor


class Hank:
    def __init__(self, state_dim, action_dim, save_dir, env, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.env = env
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.95
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.use_cuda = torch.cuda.is_available()

        # Hank's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = NeuralNet(self.state_dim, self.action_dim).double()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        #self.exploration_rate_decay = 0.999999
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e4  # no. of experiences between saving Hank Net
        
        if checkpoint:
            try:
                self.load(checkpoint)
            except:
                print(f"{checkpoint} not found! Initializing Hank anyway...")

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Hank will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:

            frame_0 = memory_to_tensor(state[0])
            frame_1 = memory_to_tensor(state[1])
            frame_2 = memory_to_tensor(state[2])
            frame_3 = memory_to_tensor(state[3])

            state = torch.stack((frame_0, frame_1, frame_2, frame_3))

            if self.use_cuda:
                state = torch.tensor(state).cuda()
            #state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            #action_idx = torch.argmax(action_values, axis=1).item()
            action_idx = torch.argmax(action_values).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """

        # state is 4x10240
        frame_0 = memory_to_tensor(state[0])
        frame_1 = memory_to_tensor(state[1])
        frame_2 = memory_to_tensor(state[2])
        frame_3 = memory_to_tensor(state[3])

        state = torch.stack((frame_0,frame_1,frame_2,frame_3))

        frame_0 = memory_to_tensor(next_state[0])
        frame_1 = memory_to_tensor(next_state[1])
        frame_2 = memory_to_tensor(next_state[2])
        frame_3 = memory_to_tensor(next_state[3])

        next_state = torch.stack((frame_0,frame_1,frame_2,frame_3))


        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
                self.save_dir / f"Hank_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"HankNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action).double()

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done).double()

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()