#!/bin/bash
### Reserve time in slurm queue manager.

#SBATCH -J mini
#SBATCH -n 16
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=16
#SBATCH -o job_exec.out
##SBATCH --gres=gpu:1
#SBATCH --nodelist=mini3
#SBATCH --partition=mini


module purge
#module load cuda-6.5
#module load openmpi-1.8.8
#module load cuda-9.0
module load openmpi-4.0.2

mpirun -np 16 ../../lmp_g++_openmpi-4.0.2_stableSep2021 -in granular_jkr.input -v data ../../box_Mauricio/box_mauricio_ff_015_np1.data > salida_jkr_50ms_015_np1.log
