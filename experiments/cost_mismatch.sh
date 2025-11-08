deployment_cost=75
learner_costs=(-150 -75 0 75 150)
for learner_cost in learner_costs; do
    python3 scripts/train_mile.py --config experiments/configs/mile_experiment_config.json \
        --learner_cost $learner_cost \
        --deployment_cost $deployment_cost
    done