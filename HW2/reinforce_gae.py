# Spring 2024, 535514 Reinforcement Learning
# HW2: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_gae")

class Policy(nn.Module):
    """
        Implement both "policy network" and the "value network" in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Shared layer
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.drop_1 = nn.Dropout(0.1)
        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        # Action layer for policy
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        # Value layer
        self.value_layer = nn.Linear(self.hidden_size, 1)


        # Initialize weights
        self.apply(self.init_weights)
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.returns = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Forward pass through shared layer
        shared_output = self.shared_layer(state)
        shared_output = F.relu( self.drop_1(shared_output) )
        shared_output = F.relu( self.linear_layer(shared_output) )

        # Forward pass through action layer
        action_logits = self.action_layer(shared_output)
        action_prob = F.softmax(action_logits, dim=-1)

        # Forward pass through value layer
        state_value = self.value_layer(shared_output)
        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Forward pass to obtain action probabilities and state value
        action_prob, state_value = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def get_returns(self, gamma=0.999):
        # Calculate rewards-to-go required by REINFORCE
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def calculate_loss(self, gamma=0.999, returns=[], gae=0):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # Calculate the policy loss

        
        t = 0
        g = 1
        for (log_prob, _), ret in zip(saved_actions, returns):
            p_loss = -log_prob * gae[t] * (gamma ** t )
            policy_losses.append(p_loss)
            t+=1
            

        # Calculate the value loss
        for (_, value), ret in zip(saved_actions, returns):
            value_losses.append(F.mse_loss(value, ret.unsqueeze(0)))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        ########## END OF YOUR CODE ##########
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma   = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########
        #print(done.shape)
        advantages = []
        nxt_value = 0
        advantage = 0
        
        for t in reversed(range(self.num_steps)):
            delta = rewards[t] + self.gamma * nxt_value - values[t]
            advantage = delta + advantage * self.gamma * self.lambda_
            advantages.insert(0, advantage)
            nxt_value = values[t]
            
        advantages = torch.tensor(advantages)

        # normalized
        # advantages = (advantages - advantages.mean()) / advantages.std()
        
        return advantages
        ########## END OF YOUR CODE ##########

def train(lr=0.03):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode,
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """

    # Instantiate the policy model and the optimizer
    lambda_ = 0.8
    gae_gamma = 0.9
    
    model = Policy()
    gae = GAE(gamma=gae_gamma, lambda_=lambda_, num_steps=100)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########
        # TODO (1): In each episode,
        # 1. run the policy till the end of the episode and keep the sampled trajectory
        # 2. update both the policy and the value network at the end of episode
        # store trajectory information
        dones = []
        for t in range(10000):
            # Forward pass through the policy network
            action = model.select_action(torch.FloatTensor(state))

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            dones.append(done)
            model.rewards.append(reward)
            ep_reward += reward
            state = next_state
            if reward < -400:
                break
            if done:
                break
      
        returns = model.get_returns()
        gae.num_steps = len(dones)
        values = [sa.value for sa in model.saved_actions]
        gae_param = gae.__call__(rewards=model.rewards, values=values, done=dones)
        # Backpropagation
        loss = model.calculate_loss(returns=returns, gae=gae_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.clear_memory()
        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('Train/length', len(model.rewards), i_episode)
        writer.add_scalar('Train/loss', loss, i_episode)
        writer.add_scalar('Train/lr', lr, i_episode)
        writer.add_scalar('Train/reward', ep_reward, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120 or i_episode > 5000:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './LunarLander-v2_gae.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break
        
def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load('./LunarLander-v2_gae.pth'.format(name)))

    render = True
    max_episode_len = 10000

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.001
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test('LunarLander-v2_gae.pth')