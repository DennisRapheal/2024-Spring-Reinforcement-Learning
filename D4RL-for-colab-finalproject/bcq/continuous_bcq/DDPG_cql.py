import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		# self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi

		self.log_std_min = -20
		self.log_std_max = 2
        
		self.mu = nn.Linear(300, action_dim)
		self.log_std_linear = nn.Linear(300, action_dim)
        
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = self.mu(a)

		log_std = self.log_std_linear(a)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mu, log_std

	def evaluate(self, state, epsilon=1e-6):
		mu, log_std = self.forward(state)
		std = log_std.exp()
		dist = Normal(mu, std)
		e = dist.rsample().to(state.device)
		action = torch.tanh(e)
		log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

		return action, log_prob


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.discount = discount
		self.tau = tau
		self.device = device

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
  
        ###Calculating CQL Loss
		q1_pred_actions = self.critic.forward(state, action)
		q1_curr_actions = self.critic.forward(state, self.actor.forward(state, action))
		q2_curr_actions = self.critic.forward(state, self.actor.forward(next_state, action))
		cat_q1=torch.cat(
		    [q1_pred_actions, 
		        q1_curr_actions,
		        q2_curr_actions], 1
		)
		alpha = torch.tensor(0.2,dtype=torch.float32)
		min_qf1_loss=torch.logsumexp(cat_q1,dim=1).mean()*alpha#logsumexp()
		min_qf1_loss=min_qf1_loss-((q1_pred_actions).mean()*alpha)
		critic_loss=(torch.tensor(0.5,dtype=torch.float32)*critic_loss)+min_qf1_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		