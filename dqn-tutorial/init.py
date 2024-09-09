import gymnasium as gym
import torch

###

if __name__ == 'init':
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
