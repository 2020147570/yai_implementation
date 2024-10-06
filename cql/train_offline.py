import gym
import d4rl.gym_mujoco
###
import imageio
import numpy as np
import mujoco
import torch
from agent import CQLAgent
from torch.utils.data import DataLoader, TensorDataset


def prep_dataloader(env_id, batch_size):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    
    tensors = dict()
    for k, v in dataset.items():
        if k in ['actions', 'observations', 'next_observations', 'rewards', 'terminals']:
            if k != 'terminals':
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors['observations'],
                               tensors['actions'],
                               tensors['rewards'][:, None],
                               tensors['next_observations'],
                               tensors['terminals'][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader, env


def evaluate(env, agent, info, eval_runs=5):
    reward_batch = []
    frames = []

    for i in range(eval_runs):
        state = env.reset()
        
        done = False
        rewards = 0
        while not done:
            action = agent.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward

            frame = env.render(mode='rgb_array')
            frames.append(frame)
        
        reward_batch.append(rewards)
    
    env_id, idx = info
    video_writer = imageio.get_writer(f"{env_id}-{str(idx)}-run.mp4", fps=30)

    for frame in frames:
        video_writer.append_data(frame)

    video_writer.close()
    
    return np.mean(reward_batch)


def main(env_id, batch_size, episodes):
    dataloader, env = prep_dataloader(env_id=env_id, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = CQLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device
    )

    batches = 0
    for i in range(1, episodes + 1):
        for batch_idx, experiences in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experiences
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            policy_loss, _, _ = agent.update((states, actions, rewards, next_states, dones))
            batches += 1
        
        eval_reward = evaluate(env, agent, (env_id, i))
        print(f"Episode: {i} | Reward: {eval_reward} | Policy loss: {policy_loss} | Batches: {batches}")

    env.close()

if __name__ == '__main__':
    main(
        env_id="halfcheetah-medium-v2",
        batch_size=256,
        episodes=1
    )
