import argparse
import gym
import numpy as np
import os
import torch

import d4rl
import uuid
import json

import bcq.continuous_bcq.BCQ
import continuous_bcq.DDPG as DDPG
import continuous_bcq.utils as utils


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f"{args.env_name}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = DDPG.DDPG(state_dim, action_dim, max_action, device)#, args.discount, args.tau)
	if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	
	evaluations = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action with noise
		if (
			(args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
			(args.train_behavioral and t < args.start_timesteps)
		):
			action = env.action_space.sample()
		else: 
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if args.train_behavioral and (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env_name, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env_name, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, args):
    # For saving files
    setting = f"{args.env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = continuous_bcq.BCQ_original.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    print('Loading buffer!')
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        # Don't apply terminals on the last step of an episode
        if episode_step == env._max_episode_steps - 1:
            episode_step = 0
            continue  
        if done_bool:
            episode_step = 0
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
        episode_step += 1
    print('Loaded buffer')
    #replay_buffer.load(f"./buffers/{buffer_name}")
    
    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
    
    while training_iters < args.max_timesteps: 
            print('Train step:', training_iters)
            pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

            evaluations.append(eval_policy(policy, args.env_name, args.seed))
            np.save(os.path.join(output_dir, f"BCQ_{setting}"), evaluations)

            training_iters += args.eval_freq
            print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                    action = policy.select_action(np.array(state))
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-random-v0")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    d4rl.set_dataset_path('/datasets')

    print("---------------------------------------")	
    if args.train_behavioral:
            print(f"Setting: Training behavioral, Env: {args.env_name}, Seed: {args.seed}")
    elif args.generate_buffer:
            print(f"Setting: Generating buffer, Env: {args.env_name}, Seed: {args.seed}")
    else:
            print(f"Setting: Training BCQ, Env: {args.env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    results_dir = os.path.join(args.output_dir, 'BCQ', str(uuid.uuid4()))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': args.env_name,
            'seed': args.seed,
        }, params_file)

    if args.train_behavioral and args.generate_buffer:
            print("Train_behavioral and generate_buffer cannot both be true.")
            exit()

    if not os.path.exists("./results"):
            os.makedirs("./results")

    if not os.path.exists("./models"):
            os.makedirs("./models")

    if not os.path.exists("./buffers"):
            os.makedirs("./buffers")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_behavioral or args.generate_buffer:
            interact_with_environment(env, state_dim, action_dim, max_action, device, args)
    else:
            train_BCQ(env, state_dim, action_dim, max_action, device, results_dir, args)
