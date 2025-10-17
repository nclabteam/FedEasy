#!/bin/bash

# CONFIG_FILE="./examples/sample_configs/benchmark.yaml"
CONFIG_FILE="./config.yaml"

# Define the list of algorithms
algorithms=(FedTVD FedAvg)

# Define the list of seed values
seeds=(123 456 8911)

alphas=(0.5)

# Loop through each algorithm
for alpha in "${alphas[@]}"
do
    for algo in "${algorithms[@]}"
    do
        # Loop through each seed value
        for seed in "${seeds[@]}"
        do
            # Run the main.py script with the current algorithm and seed
            echo "Running main.py with algorithm: $algo and seed: $seed dir alpha : $alpha"
            python main.py --config="$CONFIG_FILE" --seed="$seed" --strategy="$algo" --dirichlet_alpha="$alpha"
        done
    done
done
