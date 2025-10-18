import numpy as np
import pickle
import argparse
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import FlattenObservation

import torch
import torch.distributions as D

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.distributions import TanhBijector
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V3_ENVIRONMENTS_GOAL_HIDDEN # type: ignore

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import policies
from stable_baselines3.common.distributions import TanhBijector

from mile.computational_model import computational_intervention_model, COST_LOOKUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def collect_synthetic_data(env:gym.Env, 
                           n_episodes:int, 
                           cost:int, 
                           cdf_scale:float, 
                           rollout_policy:Union[SACPolicy, policies.ActorCriticPolicy, QNetwork], 
                           intervention_policy:Union[SACPolicy, policies.ActorCriticPolicy, QNetwork],
                           mental_model:Optional[policies.BasePolicy]=None,
                           max_t:int=1000):
    rollout_policy.eval()
    intervention_policy.eval()
    if mental_model is not None:
        mental_model.eval()

    if isinstance(env.action_space, gym.spaces.Box):
        dataset = dict(state=[], 
                    rollout_action=[], 
                    action=[], 
                    intervention_prob=[], 
                    intervention=[], 
                    reward=[], 
                    next_state=[], 
                    done=[])
    elif isinstance(env.action_space, gym.spaces.Discrete):
        dataset = dict(state=[], 
                rollout_action=[],
                intervention=[],
                human_action=[],
                human_action_prob=[], 
                next_state=[], 
                reward=[], 
                done=[])
    scores = []
    successes = []

    with torch.no_grad():
        for eps in range(n_episodes):               
            state, _ = env.reset()
            score = 0
            success = 0
            for t in range(max_t):
                rollout_action, _ = rollout_policy.predict(state, deterministic=True)
                tensor_state = torch.from_numpy(state).float().to(device)
                if isinstance(env.action_space, gym.spaces.Box):
                    rollout_action = TanhBijector.inverse(torch.from_numpy(rollout_action)).numpy() if isinstance(rollout_policy, SACPolicy) else rollout_action
                    final_mu, final_log_std, intervention_prob, _, _ = computational_intervention_model(tensor_state, mental_model, intervention_policy, cost=cost, cdf_scale=cdf_scale)
                    intervention_prob = intervention_prob.squeeze(0).detach().cpu().numpy()

                    intervene = np.random.choice([0, 1], p=intervention_prob)
                    if intervene:
                        final_action_dist = D.Normal(final_mu, final_log_std.exp())
                        action = final_action_dist.sample()
                        action = action.squeeze(0).detach().cpu().numpy()
                    else:
                        action = rollout_action

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    dataset['state'].append(state)
                    dataset['rollout_action'].append(rollout_action)
                    dataset['action'].append(action)
                    dataset['intervention_prob'].append(intervention_prob)
                    dataset['intervention'].append(intervene)

                elif isinstance(env.action_space, gym.spaces.Discrete):
                    final_prob, intervention_prob = computational_intervention_model(state=tensor_state,
                                                                                     mental_model=mental_model,
                                                                                     policy=intervention_policy,
                                                                                     cost=cost,
                                                                                     cdf_scale=cdf_scale)
                    human_action = final_prob.argmax().item()
                    if rollout_action == human_action or human_action == env.action_space.n:
                        action = rollout_action
                        dataset['intervention'].append(0)
                    else:
                        action = human_action
                        dataset['intervention'].append(1)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    dataset['state'].append(state)
                    dataset['rollout_action'].append(rollout_action)
                    dataset['human_action_prob'].append(final_prob.detach().cpu().numpy())
                    dataset['human_action'].append(human_action)

                dataset['next_state'].append(next_state)
                dataset['reward'].append(reward)
                dataset['done'].append(done)

                state = next_state
                score += reward
                if isinstance(env.action_space, gym.spaces.Box):
                    # If continuous action space
                    if done or info['success']:
                        if info['success']:
                            success = 1
                        break
                elif isinstance(env.action_space, gym.spaces.Discrete):
                    # There is no 'success' info received in the discrete envs
                    # So we just break when done
                    if done:
                        break
            scores.append(score)
            successes.append(success)

    return dataset, np.mean(scores), np.mean(successes)


def main(args, 
         rollout_policy:Union[SACPolicy, policies.ActorCriticPolicy, QNetwork], 
         intervention_policy:Union[SACPolicy, policies.ActorCriticPolicy, QNetwork], 
         mental_model:Union[policies.ActorCriticPolicy, QNetwork]):
    env_name = args.env_name
    n_episodes = args.n_episodes
    
    if env_name+'-goal-observable' in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        assert isinstance(mental_model, policies.ActorCriticPolicy), 'Mental model must be continuous for this environment'
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStackObservation(env, 4)
        env = FlattenObservation(env)
    else:
        assert isinstance(mental_model, QNetwork), 'Mental model must be discrete for this environment'
        env = gym.make(env_name)

    if env_name not in COST_LOOKUP:
        raise ValueError(f'Cost lookup for {env_name} is not available, please add it to COST_LOOKUP')
    cost = COST_LOOKUP[env_name][0]
    cdf_scale = COST_LOOKUP[env_name][1]

    rollout_policy = rollout_policy.load(args.rollout_policy)
    intervention_policy = intervention_policy.load(args.intervention_policy)
    mental_model = mental_model.load(args.mental_model)
    env = Monitor(env)

    # methods = ['Initial Policy', 'Initial Policy + Intervention', 'Mental Model Rollouts']
    methods = ['softmax']
    for method in methods:
        dataset, mean_score, mean_success = collect_synthetic_data(env=env,
                                        n_episodes=n_episodes,
                                        cost=cost,
                                        cdf_scale=cdf_scale,
                                        rollout_policy=rollout_policy,
                                        intervention_policy=intervention_policy,
                                        mental_model=mental_model)
        if method == 'softmax':
            print('\tDataset size: {}'.format(len(dataset['state'])))
            print('\tPercentage of no-intervention: {}'.format(len(np.where(np.array(dataset['intervention'])==0)[0])/len(dataset['intervention'])))
            print('\tNumber of no-intervention: {}'.format(len(np.where(np.array(dataset['intervention'])==0)[0])))
            if args.save_path:
                with open(args.save_path, 'wb') as f:
                    pickle.dump(dataset, f)

        # Mean success will be 0 if it is not set in the environment settings. #
        print(f'\tAverage score: {mean_score}')
        print(f'\tSuccess rate: {mean_success}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='peg-insert-side-v2')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--rollout_policy', type=str, default='./initial_policy', help='Path to the rollout policy')
    parser.add_argument('--intervention_policy', type=str, default='./expert_policy', help='Path to the expert policy')
    parser.add_argument('--mental_model', type=str, default='./gt_mental_model', help='Path to the trained mental model')
    parser.add_argument('--save_path', type=str, default='./intervention_dataset.pkl', help='Path to save the dataset')
    args = parser.parse_args()

    ## Change policy types if you have different pretrained policies. ##
    rollout_policy = SACPolicy.load(args.rollout_policy)
    intervention_policy = SACPolicy.load(args.intervention_policy)
    mental_model = policies.ActorCriticPolicy.load(args.mental_model)

    main(args, rollout_policy, intervention_policy, mental_model)
