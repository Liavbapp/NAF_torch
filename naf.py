import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


def mse_loss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()

def update_fixed_network(target, source, tau=1):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QNetwork(nn.Module):

    def __init__(self, hidden_size, state_features_size, action_space):
        super(QNetwork, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.bn0 = nn.BatchNorm1d(state_features_size) # batch network, layer 0 , size num_inputs = state featu
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(state_features_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1) # linear function, receives hidden_size and output the estimated value function
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, num_outputs) # linear function, returns the best action for a state
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, num_outputs ** 2) # in order to calculate the Advantage function
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        # lower trinagular matrix (without the diagonal), for calculate the advantage function
        self.tril_mask = Variable(torch.tril(torch.ones(num_outputs, num_outputs), diagonal=-1).unsqueeze(0))
        # for advantage function calculation
        self.diag_mask = Variable(torch.diag(torch.diag(torch.ones(num_outputs,  num_outputs))).unsqueeze(0))

    def forward(self, inputs):
        x, u = inputs  # state, action
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        V = self.V(x)  # applying linear1, linear2 on x and finally V.
        mu = F.tanh(self.mu(x))  # applying linear1, linear2 on x and finally F.than(mu(x)).

        Q = None
        # calculating the advantage function
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)

            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)

            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF:
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = QNetwork(hidden_size, num_inputs, action_space)
        self.target_model = QNetwork(hidden_size, num_inputs, action_space)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        update_fixed_network(self.target_model, self.model)

    # returns action normalized to range of [-1,1]
    def select_action(self, state, action_noise=None):
        self.model.eval()
        mu, _, _ = self.model((Variable(state), None))
        self.model.train()
        mu = mu.data
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        _, _, next_state_values = self.target_model((next_state_batch, None))  # V' (of theta - target model)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        # expected_state_action_values = reward_batch + (self.gamma * mask_batch + next_state_values) bug?
        expected_state_action_values = reward_batch + (self.gamma * mask_batch * next_state_values)

        _, state_action_values, _ = self.model((state_batch, action_batch))

        loss = mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        update_fixed_network(self.target_model, self.model, self.tau)

        return loss.item()

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/naf_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(model_path))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
