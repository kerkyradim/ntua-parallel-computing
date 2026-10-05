#!/bin/bash
#PBS -o NeuralNetwork.out
#PBS -e NeuralNetwork.err
#PBS -l walltime=01:00:00
#PBS -q serial
#PBS -l nodes=silver1:ppn=40

cd $HOME/ParallelMatMulOnNeuralNetworkAssignment

echo "=== Running Experiment 1 (Small matrices, many epochs) ==="
python3 testNN.py --epochs 1000 --batch_size 128 --mlp0 128 --mlp1 64

echo "=== Running Experiment 2 (Huge matrices, 1 epoch) ==="
python3 testNN.py --epochs 1 --batch_size 8192 --mlp0 8192 --mlp1 4096
