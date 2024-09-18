import gymnasium as gym
import pickle
import numpy as np
from collections import deque
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributions as D

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common import policies
from stable_baselines3.common.distributions import TanhBijector
from stable_baselines3.common.monitor import Monitor

from imitation.util import util

from mile.utils import Logger, ContInterventionMetrics, DiscInterventionMetrics
from mile.computational_model import computational_intervention_model, sum_independent_dims, COST_LOOKUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rand = np.random.randint(0, 1000)   


def generate_rollout(agent: Union[SACPolicy, policies.ActorCriticPolicy, QNetwork], 
                    env: gym.Env, 
                    env_name: str,
                    num_episodes: int=10, 
                    max_t: int=1000, 
                    scores_window: deque=None, 
                    with_intervention: bool=False, 
                    intervention_policy: Union[SACPolicy, policies.ActorCriticPolicy, QNetwork]=None,
                    mental_model: Union[policies.ActorCriticPolicy, QNetwork]=None,
                    seeded: bool=False,
                    ) -> Tuple[deque, float]:
    
    agent.set_training_mode(False)
    if with_intervention:
        assert intervention_policy is not None, 'Intervention policy is not provided'
        assert mental_model is not None, 'Mental model is not provided'
        intervention_policy.set_training_mode(False)
        mental_model.set_training_mode(False)
        if env_name not in COST_LOOKUP:
            raise ValueError(f'Cost lookup for {env_name} is not available, please add it to COST_LOOKUP')
        cost = COST_LOOKUP[env_name][0]
        cdf_scale = COST_LOOKUP[env_name][1]
    with torch.no_grad():
        success_rate = 0
        for eps in range(num_episodes):
            state, _ = env.reset(seed=eps)
            score = 0
            success = 0
            for t in range(max_t):
                rollout_action, _ = agent.predict(state, deterministic=True)
                if with_intervention:
                    if isinstance(env.action_space, gym.spaces.Box):
                        final_mu, final_log_std, intervention_prob, _, _ = computational_intervention_model(state=torch.from_numpy(state).float().to(device),
                                                                                                            mental_model=mental_model,
                                                                                                            policy=intervention_policy,
                                                                                                            cost=cost,
                                                                                                            cdf_scale=cdf_scale)
                        final_mu = final_mu.squeeze(0)
                        final_log_std = final_log_std.squeeze(0)
                        intervention_prob = intervention_prob.squeeze(0).detach().cpu().numpy()
                        intervene = np.random.choice([0, 1], p=intervention_prob)
                        if intervene:
                            final_action_dist = D.Normal(final_mu, final_log_std.exp())
                            action = final_action_dist.sample()
                            action = action.detach().cpu().numpy()
                        else:
                            if isinstance(agent, SACPolicy):
                                rollout_action = TanhBijector.inverse(torch.from_numpy(rollout_action)).numpy()
                            action = rollout_action
                    elif isinstance(env.action_space, gym.spaces.Discrete):
                        final_prob, intervention_prob = computational_intervention_model(state=torch.from_numpy(state).float().to(device),
                                                                                           mental_model=mental_model,
                                                                                           policy=intervention_policy,
                                                                                           cost=cost,
                                                                                           cdf_scale=cdf_scale)
                        human_action = final_prob.argmax().item()
                        if rollout_action == human_action or human_action == env.action_space.n:
                            action = rollout_action
                        else:
                            action = human_action
                else:
                    if isinstance(agent, SACPolicy):
                        rollout_action = TanhBijector.inverse(torch.from_numpy(rollout_action)).numpy()
                    action = rollout_action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                score += reward
                if done or info['success']==1:
                    if info['success']==1:
                        success = 1
                    break
            success_rate += success
            scores_window.append(score)

        success_rate = success_rate / num_episodes
    return scores_window, success_rate


def mile_disc_loss_fn(pred_probs: torch.Tensor, 
                      gt_labels: torch.Tensor, 
                      reduction='mean'):
    pred_probs = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
    loss = F.nll_loss(torch.log(pred_probs), gt_labels, reduction=reduction)
    return loss


def mile_cont_loss_fn(intervention_prob: torch.Tensor, 
                      mu: torch.Tensor, 
                      log_std: torch.Tensor, 
                      ground_truth_action: torch.Tensor, 
                      ground_truth_intervention: torch.Tensor, 
                      LAMBDA1: float=1.0,
                      LAMBDA2: float=1.0,
                      reduction='mean'):
    '''
    Inputs:
    - intervention_prob: (batch_size, 2)
    - mu: (batch_size, action_size)
    - log_std: (batch_size, action_size)
    - ground_truth_action: (batch_size, action_size)
    - ground_truth_intervention: (batch_size,)
    '''
    discrete_loss = mile_disc_loss_fn(intervention_prob, ground_truth_intervention, reduction=reduction)
    intervention_indices = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:,-1] > 0.0)
    if intervention_indices.sum() == 0:
        continuous_loss = torch.tensor(0.0).to(device)
    else:
        mu = mu[intervention_indices]
        log_std = log_std[intervention_indices]
        ground_truth_action = ground_truth_action[intervention_indices]
        dist = D.Normal(mu, log_std.exp())
        log_prob = sum_independent_dims(dist.log_prob(ground_truth_action))
        continuous_loss = -log_prob.mean()
    loss = LAMBDA1 * continuous_loss + LAMBDA2 * discrete_loss

    return loss, continuous_loss, discrete_loss

class InterventionTrainer:
    def __init__(self,
                 policy: Union[SACPolicy, policies.ActorCriticPolicy, QNetwork],
                 mental_model: Union[policies.ActorCriticPolicy, QNetwork],
                 env: gym.Env,
                 logger: Logger,
                 config: dict,
                 lr: float=1e-3,
                 batch_size: int=64,
                 num_epochs: int=10,
                 lambda1: float=1.0,
                 lambda2: float=1.0,
                 ):
        self.policy = policy
        self.mental_model = mental_model
        self.env = env
        self.config = config
        self.experiment_config = config['experiment']
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(list(policy.parameters()) + list(mental_model.parameters()), lr=lr)
        self.logger = logger

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.loss_fn = mile_disc_loss_fn
        elif isinstance(env.action_space, gym.spaces.Box):  
            self.loss_fn = mile_cont_loss_fn

        self.env_name = config['experiment']['env_name']
        if self.env_name not in COST_LOOKUP:
            raise ValueError(f'Cost lookup for {self.env_name} is not available, please add it to COST_LOOKUP')
        self.intervention_cost = COST_LOOKUP[self.env_name][0]
        self.intervention_scale = COST_LOOKUP[self.env_name][1]

        self.score_window = deque(maxlen=100)
        self.init_policy = deepcopy(self.policy)
        self.score_window, init_success_rate = generate_rollout(self.policy, self.env, env_name=self.env_name, scores_window=self.score_window)
        self.logger.log_rollout(success_rate=init_success_rate, init_success_rate=init_success_rate)

    def _train_one_epoch(self, dataloader):
        self.policy.train()
        self.mental_model.train()
        total_loss = 0
        total_continuous_loss = 0
        total_discrete_loss = 0

        for batch_num, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            state = batch['state'].float().to(device)
            if isinstance(self.env.action_space, gym.spaces.Box):
                intervention_prob = batch['intervention_prob'].float().to(device)
                ground_truth_action = batch['action'].float().to(device)
                ground_truth_intervention = batch['intervention'].long().to(device)


                final_mu, final_log_std, intervention_prob, policy_mu, policy_log_std = computational_intervention_model(state=state, 
                                                                                                                        mental_model=self.mental_model, 
                                                                                                                        policy=self.policy, 
                                                                                                                        cost=self.intervention_cost, 
                                                                                                                        cdf_scale=self.intervention_scale)
                loss, continuous_loss, discrete_loss = self.loss_fn(intervention_prob, 
                                                                        final_mu, 
                                                                        final_log_std, 
                                                                        ground_truth_action, 
                                                                        ground_truth_intervention,
                                                                        LAMBDA1=self.lambda1,
                                                                        LAMBDA2=self.lambda2,)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_continuous_loss += continuous_loss.item()
                total_discrete_loss += discrete_loss.item()

                training_metrics = ContInterventionMetrics(
                                        total_loss=loss.item(),
                                        cont_loss=continuous_loss.item(),
                                        disc_loss=discrete_loss.item(),
                                    )
                self.logger.log_batch(batch_num=batch_num,
                                      batch_size=self.batch_size,
                                      training_metrics=training_metrics,
                                    )

            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                human_action = batch['human_action'].long().to(device)
                final_prob, intervention_prob = computational_intervention_model(state=state,
                                                                                        mental_model=self.mental_model,
                                                                                        policy=self.policy,
                                                                                        cost=self.intervention_cost,
                                                                                        cdf_scale=self.intervention_scale)
                loss = self.loss_fn(final_prob, human_action)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                training_metrics = DiscInterventionMetrics(
                                        total_loss=loss.item(),
                                    )
                self.logger.log_batch(batch_num=batch_num,
                                      batch_size=self.batch_size,
                                      training_metrics=training_metrics,
                                    )

        if total_continuous_loss == 0:
            training_metrics = DiscInterventionMetrics(
                                            total_loss=total_loss,
                                        )
        else:
            training_metrics = ContInterventionMetrics(
                                            total_loss=total_loss,
                                            cont_loss=total_continuous_loss,
                                            disc_loss=total_discrete_loss,
                                        )
        
        return training_metrics
    
    def _validate_one_epoch(self, dataloader):
        self.policy.eval()
        self.mental_model.eval()
        total_loss = 0
        total_continuous_loss = 0
        total_discrete_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                state = batch['state'].float().to(device)
                if isinstance(self.env.action_space, gym.spaces.Box):
                    intervention_prob = batch['intervention_prob'].float().to(device)
                    ground_truth_action = batch['action'].float().to(device)
                    ground_truth_intervention = batch['intervention'].long().to(device)

                    final_mu, final_log_std, intervention_prob, policy_mu, policy_log_std = computational_intervention_model(state=state, 
                                                                                                                            mental_model=self.mental_model, 
                                                                                                                            policy=self.policy, 
                                                                                                                            cost=self.intervention_cost, 
                                                                                                                            cdf_scale=self.intervention_scale)
                    loss, continuous_loss, discrete_loss = self.loss_fn(intervention_prob, 
                                                                        final_mu, 
                                                                        final_log_std, 
                                                                        ground_truth_action, 
                                                                        ground_truth_intervention,
                                                                        LAMBDA1=self.lambda1,
                                                                        LAMBDA2=self.lambda2,)
                    total_loss += loss.item()
                    total_continuous_loss += continuous_loss.item()
                    total_discrete_loss += discrete_loss.item()

                    validation_metrics = ContInterventionMetrics(
                                            total_loss=total_loss,
                                            cont_loss=total_continuous_loss,
                                            disc_loss=total_discrete_loss,
                                        )

                elif isinstance(self.env.action_space, gym.spaces.Discrete):
                    human_action = batch['human_action'].long().to(device)
                    final_prob, intervention_prob = computational_intervention_model(state=state,
                                                                                    mental_model=self.mental_model,
                                                                                    policy=self.policy,
                                                                                    cost=self.intervention_cost,
                                                                                    cdf_scale=self.intervention_scale)
                    loss = self.loss_fn(final_prob, human_action)
                    total_loss += loss.item()

                    validation_metrics = DiscInterventionMetrics(
                                            total_loss=total_loss,
                                        )

        return validation_metrics 
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, round: Optional[int]=None):
        best_success_rate = 0
        for epoch in range(1, self.num_epochs+1):
            validation_metrics = None
            success_rate = None
            init_success_rate = None
            epoch_training_metrics = self._train_one_epoch(train_dataloader)
            if self.experiment_config['validate']['enabled']:
                if epoch % self.experiment_config['validate']['every_n_epochs'] == 0 or epoch == self.num_epochs:
                    validation_metrics = self._validate_one_epoch(val_dataloader)
            if self.experiment_config['rollout']['enabled']:
                if epoch % self.experiment_config['rollout']['every_n_epochs'] == 0 or epoch == self.num_epochs:
                    self.scores_window, success_rate = generate_rollout(self.policy, 
                                                                        self.env, 
                                                                        env_name=self.env_name,
                                                                        num_episodes=self.experiment_config['rollout']['n_episodes'], 
                                                                        scores_window=self.score_window)
                    scores_window = deque(maxlen=100)
                    scores_window, init_success_rate = generate_rollout(self.init_policy, self.env, env_name=self.env_name, num_episodes=self.experiment_config['rollout']['n_episodes'], scores_window=scores_window)
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_policy = deepcopy(self.policy)
                        best_mental_model = deepcopy(self.mental_model)
    
            if self.experiment_config['save']['enabled'] and self.experiment_config['save']['every_n_epochs'] > 0:
                save_every = self.experiment_config['save']['every_n_epochs']
                if epoch % save_every == 0:
                    self.policy.save(self.experiment_config['save']['outdir']+f'/policy_{epoch}')
                    self.mental_model.save(self.experiment_config['save']['outdir']+f'/mental_model_{epoch}')

            epoch_number = round*self.num_epochs + epoch if round is not None else epoch
            self.logger.log_epoch_metrics(epoch_number=epoch_number,
                                          epoch_loss=epoch_training_metrics,
                                          validation_metrics=validation_metrics,
                                          success_rate=success_rate,
                                          init_success_rate=init_success_rate,
                                        )

        if self.experiment_config['save']['enabled']:
            self.policy.save(self.experiment_config['save']['outdir']+'/policy')
            self.mental_model.save(self.experiment_config['save']['outdir']+'/mental_model')
            if self.experiment_config['save']['on_best_rollout_success_rate']:
                best_policy.save(self.experiment_config['save']['outdir']+'/best_policy')
                best_mental_model.save(self.experiment_config['save']['outdir']+'/best_mental_model')
            with open(self.experiment_config['save']['outdir']+'/config.pkl', 'wb') as f:
                pickle.dump(self.config, f)