#!/bin/bash

deployment_cost=(-75 0 75)
learner_costs=(-75 0 75)
python_path=/home/jovyan/workspace/mile-code
for deployment_cost in "${deployment_cost[@]}"; do
    for learner_cost in "${learner_costs[@]}"; do
        for (( i = 0; i < 2; i++ )); do
            echo "Running with learner cost: $learner_cost and deployment cost: $deployment_cost"
            PYTHONPATH=$python_path python3 scripts/train_mile.py --config $python_path/config.json \
                --learner_cost $learner_cost \
                --deployment_cost $deployment_cost
        done
    done
done