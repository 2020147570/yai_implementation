import math
import random
import torch
from load_hyperparameters import get_steps_done, update_steps_done, EPS_DECAY, EPS_END, EPS_START


def select_action(env, state, policy_net, device):
    steps_done = get_steps_done()

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)

    update_steps_done()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
