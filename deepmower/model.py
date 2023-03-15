import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = HybridBase


        self.base = base(obs_shape, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            num_outputs = 4
            self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, inputs_num, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, inputs_num, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, inputs_num, rnn_hxs, masks):
        value, _, _ = self.base(inputs, inputs_num, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, inputs_num, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, inputs_num, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            # hxs = hxs.squeeze(0)
            hxs = hxs.squeeze(0).squeeze(0)  # need to do twice because of hacky thing in forward
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))), nn.ReLU(),
            init_(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))), nn.ReLU(),
            #init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(),
            Flatten(),
            init_(nn.Linear(8 * 9 * 28, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs




class HybridBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, hidden_num=512):
        super(HybridBase, self).__init__(recurrent, hidden_size, hidden_num)

        # if recurrent:
        #     num_inputs = hidden_size

        init_cnn_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))



        self.main = nn.Sequential(
            init_cnn_(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))), nn.ReLU(),
            init_cnn_(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))), nn.ReLU(),
            #init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(),
            Flatten(),
            init_cnn_(nn.Linear(8 * 9 * 28, hidden_size)), nn.ReLU())

        # if recurrent:
        #     num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.combined_hidden_1 = nn.Sequential(
            init_(nn.Linear(num_inputs+hidden_size, hidden_num)), nn.Tanh()
        )

        self.actor_hidden = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_num)), nn.Tanh()
        )

        self.critic_hidden = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_num)), nn.Tanh()
        )

        self.actor_output = nn.Sequential(
            init_(nn.Linear(hidden_num+hidden_size, hidden_size)), nn.Tanh()
        )

        self.critic_output = nn.Sequential(
            init_(nn.Linear(hidden_num+hidden_size, hidden_size)), nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))


        self.train()

    def forward(self, inputs, inputs_num, rnn_hxs, masks):
        ### TODO: Consider having separate CNN for actor and critic
        if inputs.shape[0] == 8:
            inputs = inputs[None, :]
            inputs_num = inputs_num[None, :]
            rnn_hxs = rnn_hxs[None, :]
        x = self.main(inputs)



        # attempt 1:
        #  - concat -> GRU -> split -> reuse x_num

        x_concat = torch.cat((x, inputs_num), dim=1)
        x = self.combined_hidden_1(x_concat)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x_num_actor = self.actor_hidden(inputs_num)
        x_num_critic = self.critic_hidden(inputs_num)

        x_actor = torch.cat((x, x_num_actor), dim=1)
        x_critic = torch.cat((x, x_num_critic), dim=1)

        y_actor = self.actor_output(x_actor)
        y_critic = self.critic_output(x_critic)

        return self.critic_linear(y_critic), y_actor, rnn_hxs