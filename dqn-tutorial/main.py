import cv2
import init
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import count
from env_manager import EnvManager
from load_hyperparameters import get_episode_durations, get_steps_done, update_episode_durations, update_steps_done, LR, NUM_EPISODES, TAU
from plot import plot_durations
from select_action import select_action
from optimize_model import optimize_model
from tqdm import tqdm


def main(device, env, memory, optimizer, policy_net, target_net):
    # video
    frame_width, frame_height = (160, 210)
    result_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    for i_episode in tqdm(range(NUM_EPISODES)):
        state, info = env.reset()
        state = torch.tensor(np.array(state, dtype=np.float32, copy=None), device=device).unsqueeze(0)
        for t in count():
            result_video.write(cv2.resize(env.render(), (frame_width, frame_height))) # video

            action = select_action(env=env, state=state, policy_net=policy_net, device=device)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(np.array(next_state, dtype=np.float32, copy=None), device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(memory=memory, optimizer=optimizer, policy_net=policy_net, target_net=target_net, device=device)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                update_episode_durations(t + 1)
                plot_durations()
                break
    
    plot_durations(show_result=True)
    plt.ioff()
    # plt.show()
    plt.savefig('result.png')
    
    result_video.release() # video


if __name__ == '__main__':
    env_manager = EnvManager(env=init.env, lr=LR, device=init.device)
    main(
        device=env_manager.device,
        env=env_manager.env,
        memory=env_manager.memory,
        optimizer=env_manager.optimizer,
        policy_net=env_manager.policy_net,
        target_net=env_manager.target_net
        )
