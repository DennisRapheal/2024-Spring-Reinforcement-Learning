import copy
import time
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 800)
		self.l1_1 = nn.Linear(800, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l1_1(a))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)

	def evaluate(self, state, action, epsilon=1e-6):
		mu, log_std = self.forward(state, action)
		std = log_std.exp()
		dist = Normal(mu, std)
		e = dist.rsample()
		action = torch.tanh(e)
		log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
		return action, log_prob


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 800)
		self.l1_1 = nn.Linear(800, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 800)
		self.l4_1 = nn.Linear(800, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l1_1(q1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l4_1(q2))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l1_1(q1))
		q1 = F.relu(self.l1_2(q1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		# CQL
		learning_rate = 0.001
		self.target_entropy = -action_dim  # -dim(A)

		self.log_alpha = torch.tensor([0.0], requires_grad=True)
		self.alpha = self.log_alpha.exp().detach()
		self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
		# CQL params
		self.with_lagrange = False
		self.temp = 0
		self.cql_weight = 0.5
		self.target_action_gap = 0
		self.cql_log_alpha = torch.zeros(1, requires_grad=True)
		self.cql_alpha_optimizer = torch.optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

	def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
		local_time = time.localtime()
		timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
		if not os.path.exists('preTrained/'):
			os.makedirs('preTrained/')

		if actor_path is None:
			actor_path = "preTrained/bcq_actor_{}_{}_{}".format(env_name, timestamp, suffix)
		if critic_path is None:
			critic_path = "preTrained/bcq_critic_{}_{}_{}".format(env_name, timestamp, suffix)
		print('Saving models to {} and {}'.format(actor_path, critic_path))
		torch.save(self.actor.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()

	# CQL need
	def calc_policy_loss(self, states, alpha):
		actions_pred, log_pis = self.actor.evaluate(states)
		q1, q2 = self.critic(states, actions_pred.squeeze(0))   
		min_Q = torch.min(q1,q2)
		actor_loss = ((alpha * log_pis - min_Q )).mean()
		return actor_loss, log_pis

	def _compute_policy_values(self, obs_pi, obs_q):
		#with torch.no_grad():
		actions_pred, log_pis = self.actor.evaluate(obs_pi)
		qs1, qs2 = self.critic(obs_q, actions_pred)
        
		return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
	def _compute_random_values(self, obs, actions, critic):
		random_values1, random_values2 = critic(obs, actions)
		random_log_probs = torch.log(0.5 ** self.action_size)
		return random_values1 - random_log_probs, random_values2 - random_log_probs

	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()
   

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))
    
				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
    
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q
				
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
   
		 	# Compute alpha loss
			alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy)).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
   
			self.alpha_optimizer.step()
			self.alpha = self.log_alpha.exp().detach()
   
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
   
			# 更新
			random_actions = torch.FloatTensor(current_Q1.shape[0] * 10, action.shape[-1]).uniform_(-1, 1)
			num_repeat = int (random_actions.shape[0] / state.shape[0])
			temp_states = state.unsqueeze(1).repeat(1, num_repeat, 1).view(state.shape[0] * num_repeat, state.shape[1])
			temp_next_states = next_state.unsqueeze(1).repeat(1, num_repeat, 1).view(next_state.shape[0] * num_repeat, next_state.shape[1])
			
			current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
			next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
			
			random_values1, random_values2 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(state.shape[0], num_repeat, 1)
			
			current_pi_values1 = current_pi_values1.reshape(state.shape[0], num_repeat, 1)
			current_pi_values2 = current_pi_values2.reshape(state.shape[0], num_repeat, 1)

			next_pi_values1 = next_pi_values1.reshape(state.shape[0], num_repeat, 1)
			next_pi_values2 = next_pi_values2.reshape(state.shape[0], num_repeat, 1)
			
			cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
			cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
			
			assert cat_q1.shape == (state.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
			assert cat_q2.shape == (state.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
			

			cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - current_Q1.mean()) * self.cql_weight
			cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - current_Q1.mean()) * self.cql_weight
			
			cql_alpha_loss = torch.FloatTensor([0.0])
			cql_alpha = torch.FloatTensor([0.0])
			if self.with_lagrange:
				cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0)
				cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
				cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

				self.cql_alpha_optimizer.zero_grad()
				cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
				cql_alpha_loss.backward(retain_graph=True)
				self.cql_alpha_optimizer.step()
			
			total_c1_loss = critic1_loss + cql1_scaled_loss
			total_c2_loss = critic2_loss + cql2_scaled_loss
			# 更新

			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)