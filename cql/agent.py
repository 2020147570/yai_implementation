import torch
import torch.nn.functional as F
import torch.optim as optim
from networks import Actor, Critic


class CQLAgent:
    def __init__(self,
                    state_dim,
                    action_dim,
                    hidden_dim=256,
                    gamma=0.99,
                    tau=0.005,
                    alpha=0.2,
                    lr=3e-4,
                    buffer_dim=1e6,
                    device='cpu'
                ):
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic network (w/ target network)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)

        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
    
    def get_action(self, state, eval=False):
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor.get_deterministic_action(state)
            else:
                action = self.actor.get_action(state)
        
        return action.numpy()
        
    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update actor network
        new_actions, log_probs = self.actor.sample(states)
        new_q1 = self.critic1(states, new_actions)
        new_q2 = self.critic2(states, new_actions)
        min_new_q = torch.min(new_q1, new_q2)

        actor_loss = torch.mean(self.alpha * log_probs - min_new_q)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic networks
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2) - self.alpha * log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_next_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()
    
    def soft_update(self, local_net, target_net):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
