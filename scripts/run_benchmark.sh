#!/bin/bash

CONFIG_FILE="./configs/benchmark.yaml"

# Define the list of algorithms
algorithms=(FedAvg FedProx FedAvgM FedNova Scaffold FedMedian FedAdam FedOpt FedLaw PowD)

# Define the list of seed values
seeds=(123 456 8911)

alphas=(0.1 0.5 100)

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
