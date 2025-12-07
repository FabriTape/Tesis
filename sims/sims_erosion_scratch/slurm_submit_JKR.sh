#!/bin/bash
### Reserve time in slurm queue manager.

#SBATCH -J jkr
#SBATCH -n 32
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=32
#SBATCH -o job_exec.out
###SBATCH --gres=gpu:1
#SBATCH --nodelist=toko06
#SBATCH --partition=XL


module purge
module load openmpi-4.0.2
./run_erosion_bed_sims_lammps_move.sh
##mpirun -np 32 ./lmp_g++_openmpi-4.0.2_stableSep2021 -in collision_erosion_bed_np1.input -v input ./box_fill_0.35_SI.data > salida_jkr_erosion.log
