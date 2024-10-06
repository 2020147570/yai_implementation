import numpy as np
import random
import torch
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device='cpu'):
        self.device = device
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            self.experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        )
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.from_numpy(np.stack(states)).float().to(self.device),
            torch.from_numpy(np.stack(actions)).float().to(self.device),
            torch.from_numpy(np.stack(rewards)).float().to(self.device),
            torch.from_numpy(np.stack(next_states)).float().to(self.device),
            torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        )
    
    def __len__(self):
        return len(self.memory)
