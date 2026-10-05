#!/bin/bash
#PBS -o CUDAMatMul.out
#PBS -e CUDAMatMul.err
#PBS -l walltime=00:30:00
#PBS -q serial
#PBS -l nodes=silver1:ppn=40

cd $HOME/ParallelMatMulOnNeuralNetworkAssignment

python3 testCUDAMatMul.py   
