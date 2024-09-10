import gymnasium as gym
import torch


if __name__ == 'init':
    env = gym.make('ALE/MsPacman-v5', frameskip=1, render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
    env = gym.wrappers.FrameStack(env, 4)
    env.metadata['render_fps'] = 30 # video

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
