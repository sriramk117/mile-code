# MILE: Model-based Intervention Learning
Codebase to replicate the results for https://liralab.usc.edu/mile/.

## Setting up the environment
- Create a new conda environment: `conda create -n mile python=3.10`
- Install packages: `pip install requirements.txt`

## Dataset generation
You can generate a synthetic dataset of interventions using our intervention model if you have a trained agent and mental model.

```
python collect_synthetic_interventions.py \
--env_name 'peg-insert-side-v2' \
--n_episodes 20 \
--rollout_policy 'path_to_your_rollout_policy' \
--intervention_policy 'path_to_expert_policy' \
--mental_model 'path_to_trained_mental_model' \
--save_path 'path_to_save' 
```

## Training MILE
```
python train_mile.py --config 'config.json'
```

## Evaluating MILE
```
python eval_mile.py --trained_model 'path_to_your_trained_model' --num_episodes 100
```
