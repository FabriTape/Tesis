#!/bin/bash
### Reserve time in slurm queue manager.

#SBATCH -J jkr
#SBATCH -n 16
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=16
#SBATCH -o ./logs/job_exec_%j.out
##SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu2
#SBATCH --partition=gpu

## Espacio añadido para ejecutar en gpu2
CARPETA_TMP="/dev/shm/fbazzoni_tmp_${SLURM_JOB_ID}"
mkdir -p $CARPETA_TMP
export TMPDIR=$CARPETA_TMP
export OMPI_MCA_orte_tmpdir_base=$CARPETA_TMP
export PMIX_SERVER_TMPDIR=$CARPETA_TMP
export PMIX_MCA_gds_hash_dir=$CARPETA_TMP
##-------- Fin del codigo añadido ------------------
module purge
module load openmpi-4.0.2
./run_erosion_bed_sims_lammps_move_np3.sh
##Limpieza del codigo añadido (ya se dije que habia terminado pero la parte 2 en esta ocasión si es buena)
rm -rf $CARPETA_TMP
##mpirun -np 32 /home/fbazzoni/lmp_g++_openmpi-4.0.2_stableSep2021 -in in.granular2_JKR -v input ./box_fill_0.15_SI.data     > salida_jkr_erosion.log
