import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """Policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = normal.rsample().to(state.device) # reparameterization trick
        action = torch.tanh(z) # [-inf, inf] -> [-1, 1]
        log_prob = (normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        return action, log_prob
    
    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = dist.rsample().to(state.device)
        action = torch.tanh(z)
        return action.detach().cpu()
    
    def get_deterministic_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Q-network"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Q-value
        self.apply(_weights_init)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
