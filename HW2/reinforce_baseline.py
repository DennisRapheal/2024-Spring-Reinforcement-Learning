import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_baseline")

class Policy(nn.Module):
    """
    Implement both "policy network" and the "value network" in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128

        # Shared layer
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        # Action layer for policy
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        # Value layer
        self.value_layer = nn.Linear(self.hidden_size, 1)

        # Initialize weights
        self.apply(self.init_weights)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        """
        Forward pass of both policy and value networks
        """
        # Forward pass through shared layer
        shared_output = F.relu(self.shared_layer(state))

        # Forward pass through action layer
        action_logits = self.action_layer(shared_output)
        action_prob = F.softmax(action_logits, dim=-1)

        # Forward pass through value layer
        state_value = self.value_layer(shared_output)

        return action_prob, state_value


    def select_action(self, state):
        """
        Select the action given the current state
        """
        # Forward pass to obtain action probabilities and state value
        action_prob, state_value = self.forward(state)
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
        Calculate the loss (= policy loss + value loss) to perform backprop later
        """
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        # Calculate rewards-to-go required by REINFORCE
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # Calculate the policy loss -> baseline
        t = 0
        for (log_prob, value), ret in zip(saved_actions, returns):
            with torch.no_grad():
                advantage = ret - value
            policy_losses.append( -log_prob * advantage * gamma ** t)
            t+=1

        # Calculate the value loss
        # for (_, value), ret in zip(saved_actions, returns):
        #     value_losses.append(F.mse_loss(value, ret.unsqueeze(0)))

        loss = torch.stack(policy_losses).sum()  # + torch.stack(value_losses).sum()

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    """
    Train the model using SGD (via backpropagation)
    """
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run infinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

         # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        for t in range(10000):
            # Forward pass through the policy network
            action = model.select_action(torch.FloatTensor(state))

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            state = next_state

            if done:
                break

        # Backpropagation
        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.clear_memory()

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # Try to use Tensorboard to record the behavior of your implementation
        writer.add_scalar('Train/length', t, i_episode)
        writer.add_scalar('Train/loss', loss.item(), i_episode)
        writer.add_scalar('Train/lr', lr, i_episode)
        writer.add_scalar('Train/reward', ep_reward, i_episode)

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_baseline.pth')
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    """
    Test the learned model
    """
    model = Policy()
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    render = True
    max_episode_len = 10000

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(torch.FloatTensor(state))
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
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test('CartPole_baseline.pth')
