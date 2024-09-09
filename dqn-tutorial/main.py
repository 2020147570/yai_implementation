import matplotlib.pyplot as plt
import torch
from itertools import count
from tqdm import tqdm

import init # env
import load_hyperparameters # EPISODE_DURATIONS
import util # memory, policy_net, target_net
from init import device
from load_hyperparameters import NUM_EPISODES, TAU
from plot import plot_durations
from train_loop import optimize_model
from util import select_action

env = init.env
EPISODE_DURATIONS = load_hyperparameters.EPISODE_DURATIONS
memory = util.memory
policy_net = util.policy_net
target_net = util.target_net


if __name__ == '__main__':
    for i_episode in tqdm(range(NUM_EPISODES)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = state
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                EPISODE_DURATIONS.append(t + 1)
                plot_durations()
                break
    
    plot_durations(show_result=True)
    plt.ioff()
    # plt.show()
    plt.savefig('result.png')
