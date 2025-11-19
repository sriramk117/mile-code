#!/bin/bash

dagger=1
python_path=/home/jovyan/workspace/mile-code
learner_cost=60
learner_cdf_scale=27

# Run DAgger experiments
for (( i = 0; i < 3; i++ )); do
    echo "Running dagger: $dagger"
    PYTHONPATH=$python_path python3 scripts/train_mile.py --config $python_path/config.json \
        --use_dagger $dagger
done

# Tune intervention probabilities between 0 and 1
for (( i = 0; i < 3; i++ )); do
    echo "Running with learner cost: $learner_cost and cdf scale: $learner_cdf_scale"
    PYTHONPATH=$python_path python3 scripts/train_mile.py --config $python_path/config.json \
        --learner_cost $learner_cost \
        --learner_cdf_scale $learner_cdf_scale
done