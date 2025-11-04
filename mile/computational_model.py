import gymnasium as gym
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributions as D

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.policies import ActorCriticPolicy


# Define a hard-coded cost lookup table for different environments
# Format: 'env_name': [cost, cdf_scale]
COST_LOOKUP = {    
    'button-press-v3': [150, 200.0],
    'peg-insert-side-v3': [75, 175.0],
    'pick-place-v3': [250, 200.0],
    'drawer-open-v3': [60, 75.0],
    'LunarLander-v3': [3, 1.0],
}
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    ## Modified from stable-baselines3 ##
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=-1)
    else:
        tensor = tensor.sum()
    return tensor


def monte_carlo_samples(state: torch.Tensor, 
                        policy: Union[ActorCriticPolicy, SACPolicy], 
                        num_samples: int = 1000) -> torch.Tensor:
    '''Compute the expectation of a function using Monte Carlo sampling
    Input:
        state: state of the environment (batch_size, state_size) or (state_size)
        policy: policy of the robot (preferably SAC policy)
        num_samples: number of samples to use for Monte Carlo sampling
    
    Output:
        samples: samples from the distribution (num_samples, batch_size, action_size) or (num_samples, action_size)
    '''
    if isinstance(policy, SACPolicy):
        # Get q function distribution from actor network
        mu, log_std, _ = policy.actor.get_action_dist_params(state)
        dist = D.Normal(mu, log_std.exp())
    elif isinstance(policy, ActorCriticPolicy):
        # Get q function distribution from policy
        dist = policy.get_distribution(state)
        dist = dist.distribution
    # Randomly sample actions from the distribution
    samples = dist.rsample((num_samples,))
    return samples

def computational_intervention_model(state: torch.Tensor,
                                     mental_model: Union[ActorCriticPolicy, QNetwork],
                                     policy: Union[SACPolicy, ActorCriticPolicy, QNetwork],
                                     cost: int = 0,
                                     cdf_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Compute the intervention criteria for a given continuous policy.
    Input:
        state: state of the environment (batch_size, state_size) or (state_size)
        policy: policy used for intervention
        mental_model: mental model of the robot (What the human thinks the robot will do)
        cost: cost of intervention (0 by default)

    Output:
        total_prob: final probability distribution in the case of intervention (distribution)
        intervention_prob: probability of intervention (batch_size, 2) or (2)
    '''
    if len(state.shape) == 1:
        state = state.unsqueeze(0)

    # Discrete environment
    if isinstance(policy.action_space, gym.spaces.Discrete):
        # Ensure mental model and policy are discrete QNetworks
        assert isinstance(mental_model, QNetwork), "Mental model should be Discrete"
        assert isinstance(policy, QNetwork), "Policy should be Discrete"

        # Obtain expected action probabilities from the trained mental model
        mental_model_expected_action_probs = mental_model(state)
        if len(mental_model_expected_action_probs.shape) == 1:
            mental_model_expected_action_probs = mental_model_expected_action_probs.unsqueeze(0)

        # Normalize the mental model's predicted action probabilities
        mental_model_expected_action_probs = F.softmax(mental_model_expected_action_probs, dim=1)

        # Obtain the policy's expected action probabilities
        policy_expected_action_probs = policy(state)
        if len(policy_expected_action_probs.shape) == 1:
            policy_expected_action_probs = policy_expected_action_probs.unsqueeze(0)

        # Normalize the policy's predicted action probabilities
        policy_expected_action_probs = F.softmax(policy_expected_action_probs, dim=1)
        mean = torch.tensor([0.0], device=state.device)
        std_dev = torch.tensor([1.0], device=state.device)

        # Computes p(v = 1 | a_h = a, s) where v is the intervention variable 
        # and a_h is the nominal action of the human
        inside_cdf = (mental_model_expected_action_probs * torch.log(policy_expected_action_probs)).sum(dim=1, keepdim=True) # computes expectation over a_h
        inside_cdf = torch.log(policy_expected_action_probs) - inside_cdf
        inside_cdf = inside_cdf - cost

        # Compute intervention probability
        intervention_prob = (policy_expected_action_probs * torch.distributions.Normal(mean, std_dev).cdf(inside_cdf)).sum(dim=1)
        
        # Model how the humans would intervene when they do 
        total_prob = torch.ones((policy_expected_action_probs.shape[0],policy_expected_action_probs.shape[1]+1), device=state.device)
        total_prob[:,:-1] = policy_expected_action_probs*intervention_prob.unsqueeze(1)
        total_prob[:,-1] = 1 - torch.sum(total_prob[:,:-1], dim=1)
    
        return total_prob, intervention_prob
    
    # Continuous environment
    elif isinstance(policy.action_space, gym.spaces.Box):
        # Use monte carlo sampling to gather information about 
        # the mental model policy
        mental_model_samples = monte_carlo_samples(state=state, policy=mental_model, num_samples=1000)
        
        # Get policy distribution from current network
        if isinstance(policy, SACPolicy):
            mu, log_std, _ = policy.actor.get_action_dist_params(state)
            policy_dist = D.Normal(mu, log_std.exp())
        elif isinstance(policy, ActorCriticPolicy):
            dist = policy.get_distribution(state)
            policy_dist = dist.distribution
        else:
            raise ValueError("Policy should be either SACPolicy or ActorCriticPolicy")
        print(f"Shape of policy dist mean: {sum_independent_dims(policy_dist.log_prob(mental_model_samples)).shape}")
        
        # Compute expectation under the mental model by summing log probs
        # of sampled actions
        mental_model_expectation = torch.mean(sum_independent_dims(policy_dist.log_prob(mental_model_samples)), dim=0)

        # Run monte carlo sampling on the policy to get action samples
        policy_samples = monte_carlo_samples(state=state, policy=policy, num_samples=1000)
        mean = torch.tensor([0.0], device=state.device)
        std_dev = torch.tensor([cdf_scale], device=state.device)
        value_diff = sum_independent_dims(policy_dist.log_prob(policy_samples))-mental_model_expectation
        inside_cdf = value_diff-cost
        policy_expectation = D.Normal(mean, std_dev).cdf(inside_cdf)
        intervention_prob = torch.mean(policy_expectation, dim=0)
        final_mu = intervention_prob.unsqueeze(1)*mu
        final_log_std = torch.log(intervention_prob.unsqueeze(1)) + log_std
        final_log_std = torch.clamp(final_log_std, LOG_STD_MIN, LOG_STD_MAX)

        # turn intervention_prob into a vector of probabilities
        intervention_prob = intervention_prob.unsqueeze(-1)
        intervention_prob = torch.cat((1-intervention_prob, intervention_prob), dim=-1)

        return final_mu, final_log_std, intervention_prob, policy_dist.mean, torch.log(policy_dist.stddev), value_diff
    

def random_intervention(intervention_rate=0.1):
    # Randomly decide whether to intervene based on a given intervention rate
    intervention = np.random.choice([0, 1], p=[1-intervention_rate, intervention_rate])
    intervention = bool(intervention)
    return intervention
