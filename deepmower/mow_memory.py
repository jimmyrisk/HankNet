import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class MowMemory:
    def __init__(self, batch_size):
        self.states = []
        self.states_num = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):

        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)

        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.states_num), \
               np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, state_numericals, action, probs, vals, reward, done):
        self.states.append(state)
        self.states_num.append(state_numericals)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.states_num = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256,
                 fc2_num_dims=32,
                 chkpt_dir='../checkpoints'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.hidden_state = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(7 * 9 * 28, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        ).float()

        self.hidden_num = nn.Sequential(
            nn.Linear(input_dims, fc2_num_dims),
            nn.ReLU(),
        ).float()


        self.actor = nn.Sequential(
            nn.Linear(fc2_num_dims+fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        ).float()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, state_num):
        if state.shape[0] == 7:
            state = state[None, :]
            state_num = state_num[None, :]
        hidden1 = self.hidden_state(state)
        hidden2 = self.hidden_num(state_num)
        x = torch.cat((hidden1, hidden2), dim=1)
        dist = self.actor(x)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims = 256, fc2_dims = 256,
                 fc2_num_dims = 32,
                 chkpt_dir = 'tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.hidden_state = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(7 * 9 * 28, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        ).float()

        self.hidden_num = nn.Sequential(
            nn.Linear(input_dims, fc2_num_dims),
            nn.ReLU(),
        ).float()

        self.critic = nn.Sequential(
            nn.Linear(fc2_num_dims+fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        ).float()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, state_num):
        if state.shape[0] == 7:
            state = state[None, :]
            state_num = state_num[None, :]
        hidden1 = self.hidden_state(state)
        hidden2 = self.hidden_num(state_num)
        x = torch.cat((hidden1, hidden2), dim=1)
        value = self.critic(x)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    # default values from paper
    def __init__(self, n_actions, input_dims,
                 gamma=0.99, alpha=0.0003, gae_lambda = 0.95,
                 policy_clip=0.1, batch_size = 64, N=2048, n_epochs = 10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = MowMemory(batch_size)

    def remember(self, state, state_numericals, action, probs, vals, reward, done):
        self.memory.store_memory(state, state_numericals, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, observation_num):
        #state = torch.tensor([observation], dtype = torch.float).to(self.actor.device)
        state = observation.to(self.actor.device)
        state_num = observation_num.to(self.actor.device)

        dist = self.actor(state, state_num)
        value = self.critic(state, state_num)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, state_num_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()


            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    # a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                    #                    (1 - int(dones_arr[k])) - values[k])
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states = torch.stack(list(state_arr[batch])).to(self.actor.device)
                states_num = torch.stack(list(state_num_arr[batch])).to(self.actor.device)
                #states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)

                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states, states_num)
                critic_value = self.critic(states, states_num)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()


                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                                                     1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]

                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def debug(self):
        print("Debug info")
        actor_wt_sum = sum(p.sum() for p in self.actor.parameters())
        print(f"Sum of NN weights: {actor_wt_sum}")
        print(f"- {sum(p.sum() for p in self.actor.hidden_state.parameters())}")
        print(f"- {sum(p.sum() for p in self.actor.hidden_num.parameters())}")
        print(f"- {sum(p.sum() for p in self.actor.actor.parameters())}")
        critic_wt_sum = sum(p.sum() for p in self.critic.parameters())
        print(f"Sum of NN weights: {critic_wt_sum}")
        print(f"- {sum(p.sum() for p in self.critic.hidden_state.parameters())}")
        print(f"- {sum(p.sum() for p in self.critic.hidden_num.parameters())}")
        print(f"- {sum(p.sum() for p in self.critic.critic.parameters())}")