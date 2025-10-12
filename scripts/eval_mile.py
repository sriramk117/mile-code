import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V3_ENVIRONMENTS_GOAL_HIDDEN # type: ignore

import pickle
import numpy as np
from collections import deque
import argparse
import os

import torch

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

from mile.algorithm import generate_rollout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rand = np.random.randint(0, 1000)   

def eval(args):
    if not os.path.isdir(args.trained_model):
        print(f"Error: {args.trained_model} is not a directory.")
        return
    config_file = args.trained_model + '/config.pkl'
    with open(config_file, 'rb') as f:
        config = pickle.load(f)

    env_name = config['experiment']['env_name']
    if env_name+'-goal-observable' in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env.reward_function_version = "v2" # Use the reward function from v2
        env._freeze_rand_vec = False
        env = FrameStackObservation(env, 4)
        env = FlattenObservation(env)
    else:
        env = gym.make(env_name)
    env = Monitor(env)
    env.reset()

    if config['experiment']['policy_type'] == 'sac':
        policy_cls = SACPolicy
    elif config['experiment']['policy_type'] == 'qnetwork':
        policy_cls = QNetwork
    elif config['experiment']['policy_type'] == 'bc':
        policy_cls = ActorCriticPolicy

    policy = policy_cls.load(args.trained_model + '/policy')
    policy = policy.eval()

    score_window = deque(maxlen=100)
    score_window, success_rate = generate_rollout(agent=policy,
                                                  env=env,
                                                  env_name=env_name,
                                                  num_episodes=args.num_episodes,
                                                  scores_window=score_window,
                                                  )
    
    print(f"Average score over {args.num_episodes} episodes: {np.mean(score_window)}")
    print(f"Success rate: {success_rate}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    eval(args)