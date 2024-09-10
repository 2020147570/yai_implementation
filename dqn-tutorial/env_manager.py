import torch.optim as optim
from dqn import DQN
from replay_memory import ReplayMemory 


class EnvManager():
    def __init__(self, env, lr, device):
        self.device=device
        self.env = env

        self.n_actions = env.action_space.n
        self.policy_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
    
    def reset(self):
        return self.env.reset()
