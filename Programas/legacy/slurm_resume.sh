#!/usr/bin/env bash
#SBATCH --job-name=ResumeLAMMPS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=32
#SBATCH -o ./logs/job_resume_%j.out
#SBATCH --nodelist=toko04
#SBATCH --partition=XL

# Use a safe name for current working directory
CWD="$(pwd)"

## Espacio de memoria temporal para ejecucion rapida
CARPETA_TMP="/dev/shm/fbazzoni_tmp_resume_${SLURM_JOB_ID}"
mkdir -p "$CARPETA_TMP"
export TMPDIR="$CARPETA_TMP"
export OMPI_MCA_orte_tmpdir_base="$CARPETA_TMP"
export PMIX_SERVER_TMPDIR="$CARPETA_TMP"
export PMIX_MCA_gds_hash_dir="$CARPETA_TMP"

module purge
module load openmpi-4.0.2

# Ejecuta el controlador en Python que irá retomando cada simulación en serie
# usando los cores asignados.
# OJO: este script referencia scripts/resume_controller.py (relativo al cwd), y
# ese controlador lee resume_collision.input y config.toml. No funcionará tal
# cual si se ejecuta desde legacy/, solo conservado como referencia.
python3 scripts/resume_controller.py

# Limpieza
rm -rf "$CARPETA_TMP"
