import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V3_ENVIRONMENTS_GOAL_HIDDEN # type: ignore

import pickle
import numpy as np
import wandb
import datetime
import tempfile
import argparse
import os

import torch
from torch.utils.data import DataLoader

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
import stable_baselines3.common.logger as sb_logger
from stable_baselines3.common.utils import get_schedule_fn

from imitation.util import util
from imitation.util.logger import HierarchicalLogger, _build_output_formats
from imitation.util.networks import RunningNorm

from imitation.policies.base import NormalizeFeaturesExtractor

from mile.utils import prepare_dataset, read_config, DictDataset, Logger, log_to_file
from mile.algorithm import InterventionTrainer
from collect_synthetic_interventions import collect_synthetic_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rand = np.random.randint(0, 1000)   

def offline_training(config):
    env_name = config['experiment']['env_name']
    if env_name+'-goal-observable' in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStackObservation(env, 4)
        env = FlattenObservation(env)
    else:
        env = gym.make(env_name)
    env = Monitor(env)
    env.reset()

    with open(config['experiment']['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)    
    train_set, valid_set = prepare_dataset(dataset)
    train_set = DictDataset(train_set)
    valid_set = DictDataset(valid_set)
    print(f"Dataset size: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=config['train']['batch_size'], shuffle=False)

    if config['experiment']['policy_type'] == 'sac':
        policy = SACPolicy.load(config['experiment']['policy_path'])
    elif config['experiment']['policy_type'] == 'qnetwork':
        policy = QNetwork.load(config['experiment']['policy_path'])
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])

    if config['experiment']['mental_model_type'] == 'bc':
        mental_model = ActorCriticPolicy(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        lr_schedule=get_schedule_fn(1),
                                        net_arch=[256, 256],
                                        features_extractor_class=NormalizeFeaturesExtractor,
                                        features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                        )
    elif config['experiment']['mental_model_type'] == 'qnetwork':
        mental_model = DQNPolicy(observation_space=env.observation_space,
                                action_space=env.action_space,
                                lr_schedule=get_schedule_fn(1),
                                net_arch=[256, 256],
                                features_extractor_class=NormalizeFeaturesExtractor,
                                features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                ).q_net

    policy.to(device)
    mental_model.to(device)
    if config['experiment']['use_warm_start']:
        mental_model.load(config['experiment']['warm_start_path'])

    print(config)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    format_strs = ["stdout"]
    if config['experiment']['logging']['terminal_output_to_txt']:
        format_strs.append("log")
    if config['experiment']['logging']['log_tb']:
        format_strs.append("tensorboard")
    if config['experiment']['logging']['log_wandb']:
        format_strs.append("wandb")
        EXPERIMENT_NAME = config['experiment']['name']+'-'+timestamp
        run = wandb.init(project=config['experiment']['name'], config=config['experiment']['mile_hyperparams'], name=EXPERIMENT_NAME)
    tempdir = util.parse_path(tempfile.gettempdir())
    folder = tempdir / timestamp
    output_formats = _build_output_formats(folder, format_strs)
    custom_logger = sb_logger.Logger(str(folder), list(output_formats))
    hier_format_strs = [f for f in format_strs if (f != "wandb" and f != "tensorboard")]
    hier_logger = HierarchicalLogger(custom_logger, hier_format_strs)

    logger = Logger(hier_logger)
    learning_cost = config['experiment']['mile_hyperparams']['learning_cost']
    trainer = InterventionTrainer(policy=policy, 
                                  mental_model=mental_model, 
                                  env=env,
                                  logger=logger,
                                  config=config,
                                  cost=learning_cost,
                                  **config['train'])
    trainer.train(train_loader, val_loader)

    if config['experiment']['logging']['log_wandb']:
        run.finish()

def iterative_training(config):
    num_rounds = config['experiment']['num_rounds']
    episodes_per_round = config['experiment']['episodes_per_round']

    env_name = config['experiment']['env_name']
    if env_name+'-goal-observable' in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable']()
        env._freeze_rand_vec = False
        env = FrameStackObservation(env, 4)
        env = FlattenObservation(env)
    else:
        env = gym.make(env_name)
    env = Monitor(env)
    env.reset()

    assert config['experiment']['policy_type'] in ['sac', 'qnetwork', 'bc'], 'Invalid policy type, choose from sac, qnetwork, bc'

    if config['experiment']['policy_type'] == 'sac':
        policy = SACPolicy.load(config['experiment']['policy_path'])
        intervention_policy = SACPolicy.load(config['experiment']['intervention_policy_path'])
        intervention_policy.eval()
    elif config['experiment']['policy_type'] == 'qnetwork':
        # Check if the provided policy paths lead to valid QNetwork policies otherwise
        # initialize randomized QNetworks
        if 'policy_path' in config['experiment'] and config['experiment']['policy_path'] is not None:
            try:
                policy = DQNPolicy.load(config['experiment']['policy_path']).q_net
            except Exception as e:
                print(f"Failed to load policy: {e}")
                policy = DQNPolicy(observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   lr_schedule=get_schedule_fn(1),
                                   net_arch=[256, 256],
                                   features_extractor_class=NormalizeFeaturesExtractor,
                                   features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                   ).q_net
                print(f"Initialized a new policy as a randomized QNetwork")
        if 'intervention_policy_path' in config['experiment'] and config['experiment']['intervention_policy_path'] is not None:
            try:
                intervention_policy = DQNPolicy.load(config['experiment']['intervention_policy_path']).q_net
            except Exception as e:
                print(f"Failed to load intervention policy: {e}")
                intervention_policy = DQNPolicy(observation_space=env.observation_space,
                                               action_space=env.action_space,
                                               lr_schedule=get_schedule_fn(1),
                                               net_arch=[256, 256],
                                               features_extractor_class=NormalizeFeaturesExtractor,
                                               features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                               ).q_net
                print(f"Initialized an intervention policy as a randomized QNetwork")
        intervention_policy.eval()
    elif config['experiment']['policy_type'] == 'bc':
        policy = ActorCriticPolicy.load(config['experiment']['policy_path'])
        intervention_policy = ActorCriticPolicy.load(config['experiment']['intervention_policy_path'])
        intervention_policy.eval()

    if config['experiment']['mental_model_type'] == 'bc':
        mental_model = ActorCriticPolicy(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        lr_schedule=get_schedule_fn(1),
                                        net_arch=[256, 256],
                                        features_extractor_class=NormalizeFeaturesExtractor,
                                        features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                        )
        gt_mental_model = ActorCriticPolicy.load(config['experiment']['gt_mental_model_path'])
        gt_mental_model.eval()
    elif config['experiment']['mental_model_type'] == 'qnetwork':
        mental_model = DQNPolicy(observation_space=env.observation_space,
                                action_space=env.action_space,
                                lr_schedule=get_schedule_fn(1),
                                net_arch=[256, 256],
                                features_extractor_class=NormalizeFeaturesExtractor,
                                features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                ).q_net
        # Initialize a randomized ground-truth mental model if the path is not provided or
        # leads to a policy that is not a QNetwork
        if 'gt_mental_model_path' in config['experiment'] and config['experiment']['gt_mental_model_path'] is not None:
            try:
                gt_mental_model = DQNPolicy.load(config['experiment']['gt_mental_model_path']).q_net
                gt_mental_model.eval()
            except Exception as e:
                print(f"Failed to load ground-truth mental model: {e}")
                gt_mental_model = DQNPolicy(observation_space=env.observation_space,
                                            action_space=env.action_space,
                                            lr_schedule=get_schedule_fn(1),
                                            net_arch=[256, 256],
                                            features_extractor_class=NormalizeFeaturesExtractor,
                                            features_extractor_kwargs=dict(normalize_class=RunningNorm),
                                            ).q_net
        gt_mental_model.eval()

    policy.to(device)
    intervention_policy.to(device)
    mental_model.to(device)
    gt_mental_model.to(device)

    if config['experiment']['use_warm_start']:
        mental_model.load(config['experiment']['warm_start_path'])

    print(config)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    learning_cost = config['experiment']['mile_hyperparams']['learning_cost']
    deployment_cost = config['experiment']['mile_hyperparams']['deployment_cost']
    deployment_cdf_scale = config['experiment']['mile_hyperparams']['deployment_cdf_scale']
    learner_cdf_scale = config['experiment']['mile_hyperparams']['learner_cdf_scale']
    lambda1 = config['experiment']['mile_hyperparams']['lambda1']
    lambda2 = config['experiment']['mile_hyperparams']['lambda2']
    EXPERIMENT_NAME = f"cost_l={learning_cost} cost_d={deployment_cost} ({num_rounds} rounds) " +'-'+timestamp
    format_strs = ["stdout"]
    if config['experiment']['logging']['terminal_output_to_txt']:
        format_strs.append("log")
    if config['experiment']['logging']['log_tb']:
        format_strs.append("tensorboard")
    if config['experiment']['logging']['log_wandb']:
        format_strs.append("wandb")
        run = wandb.init(project=config['experiment']['name'], config=config['experiment']['mile_hyperparams'], name=EXPERIMENT_NAME)
    tempdir = util.parse_path(tempfile.gettempdir())
    folder = tempdir / timestamp
    output_formats = _build_output_formats(folder, format_strs)
    custom_logger = sb_logger.Logger(str(folder), list(output_formats))
    hier_format_strs = [f for f in format_strs if (f != "wandb" and f != "tensorboard")]
    hier_logger = HierarchicalLogger(custom_logger, hier_format_strs)

    logger = Logger(hier_logger)
    learning_cost = config['experiment']['mile_hyperparams']['learning_cost']
    trainer = InterventionTrainer(policy=policy, 
                                  mental_model=mental_model, 
                                  env=env,
                                  logger=logger,
                                  config=config,
                                  cost=learning_cost,
                                  **config['train'])

    if config['experiment']['include_offline_dataset']:
        with open(config['experiment']['offline_dataset_path'], 'rb') as f:
            dataset = pickle.load(f)    
    else:
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

    intervention_probs_per_round = []
    inside_cdf_terms_per_round = []
    lowest_intervention_rate = 1.0
    deployment_cost = config['experiment']['mile_hyperparams']['deployment_cost']
    deployment_cdf_scale = config['experiment']['mile_hyperparams']['deployment_cdf_scale']
    for round in range(num_rounds):
        intervention_probs = np.array([])
        log_to_file('Round: {}'.format(round), EXPERIMENT_NAME+'_log.txt')
        print('Collecting intervention data...')
        additional_data, mean_score, mean_success_rate, mean_intervention_rate, inside_cdf_terms = collect_synthetic_data(env=env,
                                                                                n_episodes=episodes_per_round,
                                                                                cost=deployment_cost,
                                                                                cdf_scale=deployment_cdf_scale,
                                                                                rollout_policy=trainer.policy,
                                                                                intervention_policy=intervention_policy,
                                                                                mental_model=gt_mental_model)
        log_to_file('Dataset size: {}'.format(len(additional_data['state'])), EXPERIMENT_NAME+'_log.txt')
        log_to_file(f"Percentage of no-intervention: {additional_data['intervention'].count(0)/len(additional_data['intervention'])}", EXPERIMENT_NAME+'_log.txt')
        log_to_file(f"Success rate: {mean_success_rate}", EXPERIMENT_NAME+'_log.txt')
        log_to_file(f"Number of interventions: {additional_data['intervention'].count(1)}", EXPERIMENT_NAME+'_log.txt')
        intervention_rate = additional_data['intervention'].count(1)/len(additional_data['intervention'])
        intervention_probs = additional_data['intervention_prob']
        intervention_probs_per_round.append(intervention_probs)
        inside_cdf_terms_per_round.append(inside_cdf_terms)

        # Keep track of the lowest intervention rate
        if mean_intervention_rate < lowest_intervention_rate:
            lowest_intervention_rate = mean_intervention_rate

        # Log intervention rate to wandb
        if config['experiment']['logging']['log_wandb']:
            logger.log_success_rate_deployment(mean_success_rate)
            logger.log_intervention_rate(mean_intervention_rate)

        for key in dataset.keys():
            if round == 0:
                dataset[key].extend(additional_data[key])
            else:
                dataset[key] = np.concatenate((dataset[key], additional_data[key]), axis=0)
        train_set, valid_set = prepare_dataset(dataset)
        print("Current dataset size: {}".format(len(train_set['state'])))
        train_set = DictDataset(train_set)
        valid_set = DictDataset(valid_set)
        train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(valid_set, batch_size=config['train']['batch_size'], shuffle=False)
        trainer.train(train_loader, val_loader, round)

    # Log histogram of intervention probabilities per round
    for round in range(num_rounds):
        round_probs = intervention_probs_per_round[round]
        print(f"Round {round} intervention probabilities: {len(round_probs)} samples")
        if config['experiment']['logging']['log_wandb']:
            histogram = logger.log_intervention_probs(round_probs, title=f"Intervention Probabilities for Round {round}")
            run.log({"intervention_probs_" + f"round_{round}": histogram})
            # histogram_inside_cdf = logger.log_intervention_probs(inside_cdf_terms_per_round[round], title=f"Inside CDF Terms for Round {round}")
            # run.log({"inside_cdf_terms_" + f"round_{round}": histogram_inside_cdf})

    # Log best intervention rate and best success rate achieved
    print(f"Learning cost: {learning_cost}")
    print(f"Deployment cost: {deployment_cost}")
    print(f"Best intervention rate achieved: {lowest_intervention_rate}")
    print(f"Best success rate achieved: {trainer.best_success_rate}")

    if config['experiment']['logging']['log_wandb']:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Intervention training')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--learner_cost', type=float, default=None, help='Override learning cost from config file')
    parser.add_argument('--learner_cdf_scale', type=float, default=None, help='Override learner cdf scale from config file')
    parser.add_argument('--deployment_cost', type=float, default=None, help='Override deployment cost from config file')
    parser.add_argument('--lambda1', type=float, default=None, help='Override lambda1 from config file') 
    parser.add_argument('--lambda2', type=float, default=None, help='Override lambda2 from config file')
    parser.add_argument('--use_dagger', type=float, default=None, help='Override use_dagger boolean from config file')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.learner_cost is not None:
        config['experiment']['mile_hyperparams']['learning_cost'] = args.learner_cost
    if args.learner_cdf_scale is not None:
        config['experiment']['mile_hyperparams']['learner_cdf_scale'] = args.learner_cdf_scale
    if args.deployment_cost is not None:
        config['experiment']['mile_hyperparams']['deployment_cost'] = args.deployment_cost
    if config['experiment']['save']['enabled']:
        os.makedirs(config['experiment']['save']['outdir'], exist_ok=True)
    if args.lambda1 is not None:
        config['train']['lambda1'] = args.lambda1
        config['experiment']['mile_hyperparams']['lambda1'] = config['train']['lambda1']
    if args.lambda2 is not None:
        config['train']['lambda2'] = args.lambda2
        config['experiment']['mile_hyperparams']['lambda2'] = config['train']['lambda2']
    if args.use_dagger is not None:
        print(f"Overriding use_dagger to: {args.use_dagger}")
        config['train']['use_dagger'] = args.use_dagger
        config['experiment']['mile_hyperparams']['use_dagger'] = config['train']['use_dagger']
    # Confirm learner cost and deployment cost got modified correctly
    print(f"Using learning cost: {config['experiment']['mile_hyperparams']['learning_cost']}")
    print(f"Using deployment cost: {config['experiment']['mile_hyperparams']['deployment_cost']}")

    if config['experiment']['mode'] == 'offline':
        offline_training(config)
    elif config['experiment']['mode'] == 'iterative':
        iterative_training(config)
    else:
        raise ValueError('Invalid mode')
    