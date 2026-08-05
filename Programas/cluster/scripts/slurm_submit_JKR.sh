#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH -o ./logs/job_exec_%j.out

# job-name, --ntasks-per-node, --nodelist y --partition los pasa enqueue_sims.py
# por línea de comandos de sbatch (dependen del config del job). Acá solo queda
# la preparación de entorno (tmp en /dev/shm + módulos OpenMPI).
# $1 (opcional) = clave del chunk (configs/config_<clave>.toml + sims_<clave>.tsv);
# si falta (envío manual), se usa el SLURM_JOB_ID.

CWD="$(pwd)"

## Espacio añadido para ejecutar en gpu2
CARPETA_TMP="/dev/shm/fbazzoni_tmp_${SLURM_JOB_ID}"
mkdir -p "$CARPETA_TMP"
export TMPDIR="$CARPETA_TMP"
export OMPI_MCA_orte_tmpdir_base="$CARPETA_TMP"
export PMIX_SERVER_TMPDIR="$CARPETA_TMP"
export PMIX_MCA_gds_hash_dir="$CARPETA_TMP"
##-------- Fin del codigo añadido ------------------

module purge
module load openmpi-4.0.2
# Ejecutar
./run_erosion_bed_sims_lammps_move_np3.sh "${1:-${SLURM_JOB_ID}}"

# Limpieza del codigo añadido
rm -rf "$CARPETA_TMP"
