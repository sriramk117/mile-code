# MILE: Model-based Intervention Learning
Codebase to replicate the results for https://liralab.usc.edu/mile/.

## Setting up the environment
- Create a new conda environment: `conda create -n mile python=3.10`
- Install required packages: `pip install requirements.txt`
- Install MILE: `pip install -e .`
## Dataset generation
You can generate a synthetic dataset of interventions using our intervention model if you have a trained agent and mental model.

```
python scripts/collect_synthetic_interventions.py \
--env_name 'peg-insert-side-v2' \
--n_episodes 20 \
--rollout_policy 'path_to_your_rollout_policy' \
--intervention_policy 'path_to_expert_policy' \
--mental_model 'path_to_trained_mental_model' \
--save_path 'path_to_save' 
```

In order to pretrain the agent and the mental model, you can follow [SB3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) and [Imitation](https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html) documents.

## Training MILE
```
python scripts/train_mile.py --config 'config.json'
```

## Evaluating MILE
```
python scripts/eval_mile.py --trained_model 'path_to_your_trained_model' --num_episodes 100
```
