#!/bin/bash
#PBS -o MultiprocessingMatMul.out
#PBS -e MultiprocessingMatMul.err
#PBS -l walltime=00:30:00
#PBS -q serial
#PBS -l nodes=silver1:ppn=40

cd $HOME/ParallelMatMulOnNeuralNetworkAssignment

python3 testMultiprocessingMatMul.py 
