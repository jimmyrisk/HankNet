import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.state_numericals = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.state_numericals[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, n_actions, input_dims,
            fc1_dims=256, fc2_dims=256,
                 fc2_num_dims=32,
                 chkpt_dir='../checkpoints'):
        super(ActorCritic, self).__init__()


        # actor

        self.actor_hidden_state = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(8 * 9 * 28, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        ).float()

        self.actor_hidden_num = nn.Sequential(
            nn.Linear(input_dims, fc2_num_dims),
            nn.ReLU(),
        ).float()

        self.actor = nn.Sequential(
            nn.Linear(fc2_num_dims+fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        ).float()

        # critic
        self.critic_hidden_state = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(8 * 9 * 28, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        ).float()

        self.critic_hidden_num = nn.Sequential(
            nn.Linear(input_dims, fc2_num_dims),
            nn.ReLU(),
        ).float()

        self.critic = nn.Sequential(
            nn.Linear(fc2_num_dims + fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        ).float()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, state_num):
        if state.shape[0] == 8:
            state = state[None, :]
            state_num = state_num[None, :]
        hidden1 = self.actor_hidden_state(state)
        hidden2 = self.actor_hidden_num(state_num)
        x = torch.cat((hidden1, hidden2), dim=1)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        critic_hidden1 = self.actor_hidden_state(state)
        critic_hidden2 = self.actor_hidden_num(state_num)
        critic_x = torch.cat((critic_hidden1, critic_hidden2), dim=1)
        state_val = self.critic(critic_x)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, state_num, action):
        if state.shape[0] == 8:
            state = state[None, :]
            state_num = state_num[None, :]
        hidden1 = self.actor_hidden_state(state)
        hidden2 = self.actor_hidden_num(state_num)
        x = torch.cat((hidden1, hidden2), dim=1)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        critic_hidden1 = self.actor_hidden_state(state)
        critic_hidden2 = self.actor_hidden_num(state_num)
        critic_x = torch.cat((critic_hidden1, critic_hidden2), dim=1)
        state_values = self.critic(critic_x)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, action_dim, state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(action_dim, state_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'eps': 1e-5},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'eps': 1e-5}
        ])

        self.policy_old = ActorCritic(action_dim, state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def select_action(self, state, state_numerical):

        with torch.no_grad():
            #state = torch.FloatTensor(state).to(device)
            state = state.float().to(device)
            state_numerical = state_numerical.float().to(device)
            action, action_logprob, state_val = self.policy_old.act(state, state_numerical)

        self.buffer.states.append(state)
        self.buffer.state_numericals.append(state_numerical)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()



    def update(self):
        # Rewards
        rewards = []
        discounted_reward = 0



        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        # https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
        # https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
        # https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/4.PPO-discrete/ppo_discrete.py

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_state_numericals = torch.squeeze(torch.stack(self.buffer.state_numericals, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()



        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_state_numericals, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values, rewards)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




