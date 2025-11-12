lambda1=(1.5 1 0.5)
lambda2=(1.5 1 0.5)
python_path=/home/jovyan/workspace/mile-code
for l1 in "${lambda1[@]}"; do
    for l2 in "${lambda2[@]}"; do
        echo "Running with lambda1: $l1, lambda2: $l2 and deployment cost: $deployment_cost"
        PYTHONPATH=$python_path python3 scripts/train_mile.py --config $python_path/config.json \
            --lambda1 $l1 \
            --lambda2 $l2 
    done
done
