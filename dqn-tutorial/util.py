import math
import random
import torch
import torch.optim as optim

import init # env
import load_hyperparameters # STEPS_DONE
from dqn import DQN
from init import device
from load_hyperparameters import EPS_DECAY, EPS_END, EPS_START, LR
from replay_memory import ReplayMemory

env = init.env
STEPS_DONE = load_hyperparameters.STEPS_DONE

###

def __env_init():
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    return policy_net, target_net, optimizer, memory


def select_action(state):
    global STEPS_DONE
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * STEPS_DONE / EPS_DECAY)

    STEPS_DONE = STEPS_DONE + 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


if __name__ == 'util':
    policy_net, target_net, optimizer, memory = __env_init()
