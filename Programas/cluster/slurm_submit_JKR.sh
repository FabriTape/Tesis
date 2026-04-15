#!/bin/bash
### Reserve time in slurm queue manager.

#SBATCH -J N1v25ff0.35mov7x7
#SBATCH -n 32
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=32
#SBATCH -o ./logs/job_exec_%j.out
#SBATCH --nodelist=toko07
#SBATCH --partition=XL

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
