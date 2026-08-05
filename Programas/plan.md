## User

propone un plan de orden y unificación de [@scripts](file:///home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/) actualmente las sims se ejecutan mandando multiples veces [@encolar_sims.sh](file:///home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/encolar_sims.sh) hasta que el usuario lo desee

## Assistant

<thinking>
Let me understand what the user is asking. They want me to propose a plan for ordering and unifying the simulation scripts in `/home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/`. Currently simulations are run by sending `encolar_sims.sh` multiple times until the user wants to stop.

Let me explore the directory structure and understand all the scripts involved. Let me read all the files in that directory to understand the current workflow.

Let me also check the surrounding context - the Tesis directory structure, and look at what these scripts do.

Let me start by reading all the scripts and config files.
</thinking>

**Tool Call: Programas/cluster/scripts/encolar_sims.sh**
Status: Completed

#!/usr/bin/env bash
set -euo pipefail

# Use a safe name, don't overwrite PATH
CWD="$(pwd)"
if [ ! -f "config.toml" ]; then
  echo "No se halló config.toml en la dir actual"
  exit 1
fi
# Submit job and capture output
out="$(sbatch slurm_submit_JKR.sh)" || { echo "sbatch failed"; exit 1; }

# Extract job id (e.g., "Submitted batch job 93804")
jobid="$(echo "$out" | awk '{print $4}')"
if [ -z "$jobid" ]; then
  echo "Failed to get job id from sbatch output: $out" >&2
  exit 1
fi

# Ensure destination exists and copy using jobid
mkdir -p "${CWD}/configs"
cp "config.toml" "${CWD}/configs/config_${jobid}.toml"

echo "Copied config to ${CWD}/configs/config_${jobid}.toml"


**Tool Call: Programas/cluster/scripts/slurm_submit_JKR.sh**
Status: Completed

#!/usr/bin/env bash
#SBATCH --job-name=270_0_1_0
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=5
#SBATCH -o ./logs/job_exec_%j.out
#SBATCH --nodelist=toko04
#SBATCH --partition=XL

# Use a safe name for current working directory
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
./run_erosion_bed_sims_lammps_move_np3.sh "${SLURM_JOB_ID}"

# Limpieza del codigo añadido
rm -rf "$CARPETA_TMP"

**Tool Call: Programas/cluster/scripts/run_erosion_bed_sims_lammps_move_np3.sh**
Status: Completed

#!/bin/bash
set -euo pipefail

INPUT="collision_erosion_bed.input"
BASE_DIR=$(pwd)
source /home/fbazzoni/.venv_gpu/bin/activate

# Buscar un Python 3.7+ (en toko04 python3.11 apunta a Python 2.7)
PY311=""
for cmd in python3.11 python3 python3.10 python3.12 /usr/bin/python3.11 /usr/bin/python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version_ok=$("$cmd" -c "
import sys
sys.exit(0 if sys.version_info >= (3, 7) else 1)
" 2>/dev/null && echo "yes" || echo "no")
        if [ "$version_ok" = "yes" ]; then
            PY311="$cmd"
            break
        fi
    fi
done
if [ -z "$PY311" ]; then
    echo "ERROR: No se encontró Python 3.7+ en el sistema" >&2
    exit 1
fi
# Cargar configuración centralizada
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
$PY311 ${SCRIPT_DIR}/read_config.py ${SCRIPT_DIR}/configs/config_$1.toml $1
source ${SCRIPT_DIR}/configs/config_vars_$1.sh
rm -f ${SCRIPT_DIR}/configs/config_vars_$1.sh

# Variables derivadas
#NP=$NP  # viene de config_vars.sh
R=$GRAIN_RADIUS_UM

# Calcular desplazamientos XMOVE, YMOVE a partir de MOVES y R
XMOVE=()
YMOVE=()

for mx in "${MOVES[@]}"; do
	for my in "${MOVES[@]}"; do
		XMOVE+=( "$(echo "$mx * $R" | bc)" )
		YMOVE+=( "$(echo "$my * $R" | bc)" )
	done
done

# Configuración MPI
MPIRUN="mpirun --allow-run-as-root -np $CORES ${BASE_DIR}/$EXE"
Fills=${FILLS[@]}
Vels=${VELS[@]}
Angles=${ANGLES[@]}
Moves=${MOVES[@]}
Orientation="90_0_1_0"
OUTDIR="${BASE_DIR}/Buffer/sims_SI_erosion_bed_100um_${Orientation}_ff_${Fills// /_}_vel_${Vels// /_}_move_${Moves// /_}_NP_${NP}_angle_${Angles// /_}_cores_8"
COUNTER=0

mkdir -p "$OUTDIR"
cd "$OUTDIR"
cp "${BASE_DIR}/$0" ./
cp "${SCRIPT_DIR}/configs/config_$1.toml" ./

for fill in "${FILLS[@]}"
do
	DATA="box_fill_${fill}_SI_100um_${Orientation}.data"
	cp "${BASE_DIR}/datas/$DATA" ./

	for vel in "${VELS[@]}"
	do
		for angle in "${ANGLES[@]}"
		do
			len=${#XMOVE[@]}
			for (( move=0; move<len; move++ )); do
				xm=${XMOVE[$move]}
				ym=${YMOVE[$move]}
				zm=0

				simdir="erosion_bed_SI_ff_${fill}_np_${NP}_vel_${vel}_angle_${angle}_move_${xm}_${ym}_${zm}"
				echo "$simdir"
				if mkdir "$simdir" 2>/dev/null; then
					cd "$simdir" || { echo "ERROR: No se pudo entrar a $simdir. Abortando."; exit 1; }

					cp "${BASE_DIR}/$INPUT" .

					nsteps=${TOTAL_STEPS[$COUNTER]:-${TOTAL_STEPS[0]}}

					echo "$DATA xrotate $XROTATE yrotate $YROTATE zrotate $ZROTATE angle $angle"
					echo "xmove $xm ymove $ym zmove $zm"
					echo "velocity $vel nsteps $nsteps ndumps $NDUMPS nthermo $NTHERMO"

					$MPIRUN -in $INPUT -v data "${BASE_DIR}/datas/$DATA" \
							   -v xrotate "$XROTATE" \
							   -v yrotate "$YROTATE" \
							   -v zrotate "$ZROTATE" \
							   -v angle "$angle" \
							   -v xmove "${xm}e-6" \
							   -v ymove "${ym}e-6" \
							   -v zmove "${zm}e-6" \
							   -v velocity "$vel" \
							   -v nsteps "$nsteps" \
							   -v ndumps "$NDUMPS" \
							   -v nthermo "$NTHERMO" \
							   -v timestep "$TIMESTEP" \
							   -v young_modulus "$YOUNG_MODULUS" \
							   -v poisson "$POISSON" \
							   -v surface_energy "$SURFACE_ENERGY" \
							   -v density_proj "$DENSITY" \
							   -v proc_x "$PROC_X" \
							   -v proc_y "$PROC_Y" \
							   -v proc_z "$PROC_Z" \
							   > salida.log
					cd ..
				else
                                        echo "La simulación $simdir ya está siendo calculada por otro nodo. Saltando..."
                                fi
			done
		done
		COUNTER=$((COUNTER + 1))
	done
	COUNTER=0
done

**Tool Call: Programas/cluster/scripts/config_vars.sh**
Status: Completed

# Auto-generado por read_config.py — NO editar a mano
# Fuente: config.toml

# [simulation]
NP=1
NDUMPS=25000
NTHERMO=5000
CORES=32
PROC_X=4
PROC_Y=4
PROC_Z=2
EXE="lmp_g++_openmpi-4.0.2_stableSep2021"
TIMESTEP=5e-11

# [sweep]
FILLS=("0.35")
VELS=("-25")
ANGLES=("0")
MOVES=("-3" "-2" "-1" "0" "1" "2" "3")
TOTAL_STEPS=("2000000")
GRAIN_RADIUS_UM=0.76
XROTATE=1
YROTATE=0
ZROTATE=0

# [material]
YOUNG_MODULUS=70000000000.0
POISSON=0.17
SURFACE_ENERGY=0.025
DENSITY=2000.0

# [defaults]
RADIO_UM=0.76
USAR_CAJA=true
BED_HEIGHT_FF_015=9.99989e-05
BED_HEIGHT_FF_035=9.99999e-05
CAJA_X=0.0001
CAJA_Y=0.0001

# [analisis_onda]
V_OBJETIVO=-50.0
TOLERANCIA_V=0.1
NP_CRITICOS=("1" "3")
MOVE_REFERENCIA=("0.0" "0.0" "0.0")
TOLERANCIA_MOVE=1

# [filtros]
UMBRAL_VZ_MIN=0.0
UMBRAL_ENERGIA_MIN=2e-05
UMBRAL_TECHO_RADIOS=5
TOLERANCIA_V_CM=1e-06

# [jkr]
FACTOR_CUTOFF=2

# [onda]
UMBRAL_V2_MIN=0.0004
UMBRAL_V2_MAX=0.04
PARTICULAS_CRITICAS=3000
UMBRAL_TECHO=5
UMBRAL_PARED=20

# [rutas]
RESULTADOS="./resultados"
IMAGENES="./resultados/Imagenes"

**Tool Call: Programas/cluster/scripts/read_config.py**
Status: Completed

# -*- coding: utf-8 -*-
"""
Generador de variables shell desde config.toml.

Lee el archivo de configuración TOML centralizado y escribe un archivo
shell con todas las variables necesarias para los scripts de simulación
y análisis.

Uso:
    python read_config.py [config.toml]
    source config_vars.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def read_config(config_path: Path) -> dict:
    """Lee el archivo TOML y retorna el diccionario crudo.

    Args:
        config_path: Ruta al archivo config.toml.

    Returns:
        Diccionario con todas las secciones y parámetros.
    """
    with config_path.open("rb") as f:
        return tomllib.load(f)


def generate_shell_vars(raw: dict, output_path: Path) -> None:
    """Genera archivo shell con variables desde el diccionario TOML.

    Recorre todas las secciones del TOML y convierte cada valor a la
    sintaxis de bash correspondiente (arrays, strings, números).

    Args:
        raw: Diccionario crudo de config.toml.
        output_path: Ruta del archivo shell a generar.
    """
    lines: list[str] = [
        "# Auto-generado por read_config.py — NO editar a mano",
        "# Fuente: config.toml",
        "",
    ]

    for section, values in raw.items():
        if not isinstance(values, dict):
            continue

        lines.append(f"# [{section}]")
        for key, value in values.items():
            if key == "processors" and isinstance(value, list) and len(value) == 3:
                lines.append(f"PROC_X={value[0]}")
                lines.append(f"PROC_Y={value[1]}")
                lines.append(f"PROC_Z={value[2]}")
                continue

            var_name = key.upper()
            lines.append(f"{var_name}={_format_value(value)}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def _format_value(value: object) -> str:
    """Convierte un valor TOML a sintaxis de bash.

    Args:
        value: Valor del TOML (int, float, str, list).

    Returns:
        Representación en sintaxis bash.
    """
    if isinstance(value, list):
        elements = " ".join(f'"{v}"' for v in value)
        return f"({elements})"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return f'"{value}"'

    return str(value)


def main() -> None:
    """Punto de entrada CLI."""
    ID = sys.argv[2]
    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"Error: {config_path} no encontrado.", file=sys.stderr)
        sys.exit(1)

    raw = read_config(config_path)
    output_path = config_path.parent / f"config_vars_{ID}.sh"
    generate_shell_vars(raw, output_path)
    print(f"Configuración cargada: {config_path} → {output_path}")


if __name__ == "__main__":
    main()

**Tool Call: Programas/cluster/scripts/config.toml**
Status: Completed

# =============================================================================
# CONFIGURACIÓN CENTRALIZADA — SIMULACIONES Y ANÁLISIS LAMMPS
# =============================================================================
# Fuente única de verdad. Leída por:
#   - read_config.py → shell (config_vars.sh)
#   - CARBOXI (análisis de erosión)
# =============================================================================

# -----------------------------------------------------------------------------
# SIMULACIÓN LAMMPS
# -----------------------------------------------------------------------------
[simulation]
np         = 3          # Número de partículas del proyectil
ndumps     = 25000      # Frecuencia de dump [timesteps]
nthermo    = 5000       # Frecuencia de thermo [timesteps]
cores      = 4          # Núcleos MPI
processors = [2, 2, 1]  # [Px, Py, Pz], producto debe ser = cores
exe        = "lmp_g++_openmpi-4.0.2_stableSep2021"
timestep   = 5e-11      # Paso temporal [s]

# -----------------------------------------------------------------------------
# BARRIDO DE PARÁMETROS
# -----------------------------------------------------------------------------
[sweep]
fills           = [0.15]
vels            = [-50]
angles          = [90]
moves           = [-3, -2, -1, 0, 1, 2, 3]
total_steps     = [2000000]   # Un valor por cada combinación fill×vel×angle
grain_radius_um = 0.76        # Radio del grano [μm] (para cálculo de desplazamiento)
xrotate         = 1
yrotate         = 0
zrotate         = 0

# -----------------------------------------------------------------------------
# MATERIAL (SÍLICE)
# -----------------------------------------------------------------------------
[material]
young_modulus   = 70.0e9     # Módulo de Young [Pa]
poisson         = 0.17       # Coeficiente de Poisson
surface_energy  = 0.025      # Energía superficial JKR [J/m²]
density         = 2000.0     # Densidad proyectil [kg/m³]

# -----------------------------------------------------------------------------
# VALORES POR DEFECTO (CARBOXI)
# -----------------------------------------------------------------------------
[defaults]
radio_um              = 0.76   # Radio de partícula [μm]
usar_caja             = true   # Usar límites fijos de la caja vs bordes dinámicos
bed_height_ff_015     = 9.99989e-5   # Altura máxima cama para ff=0.15 [m]
bed_height_ff_035     = 9.99999e-5   # Altura máxima cama para ff=0.35 [m]
caja_x                = 1e-4   # Tamaño caja en X [m]
caja_y                = 1e-4   # Tamaño caja en Y [m]

# -----------------------------------------------------------------------------
# CRITICIDAD PARA ANÁLISIS DE ONDA AUTOMÁTICO (CARBOXI)
# -----------------------------------------------------------------------------
[analisis_onda]
v_objetivo       = -50.0       # Velocidad exacta de impacto [m/s]
tolerancia_v     = 0.1         # Margen de error para comparación de velocidad
np_criticos      = [1, 3]      # Cantidad de partículas del proyectil críticas
move_referencia  = [0.0, 0.0, 0.0]   # Vector de reposo para comparación
tolerancia_move  = 1      # Tolerancia para considerar move != 0

# -----------------------------------------------------------------------------
# UMBRALES DE FILTRADO PARA EROSIÓN (CARBOXI)
# -----------------------------------------------------------------------------
[filtros]
umbral_vz_min       = 0.0     # Velocidad vertical mínima [m/s]
umbral_energia_min  = 0.00002 # Energía cinética mínima [m²/s²]
umbral_techo_radios = 5       # Margen de seguridad en número de radios (Z)
tolerancia_v_cm     = 1e-6    # Tolerancia |Δv_cm| entre dumps [m/s]. 0 = estricto

# -----------------------------------------------------------------------------
# MODELO JKR (CARBOXI)
# -----------------------------------------------------------------------------
[jkr]
factor_cutoff = 2.05  # Factor multiplicador sobre el radio para cutoff de OVITO

# -----------------------------------------------------------------------------
# DETECCIÓN DE ONDA (CARBOXI)
# -----------------------------------------------------------------------------
[onda]
umbral_v2_min      = 0.0004  # Energía cinética mínima de partícula activa [m²/s²]
umbral_v2_max      = 0.04    # Energía cinética máxima para impacto crítico [m²/s²]
particulas_criticas = 3000   # Cantidad de partículas para onda macroscópica
umbral_techo       = 5       # Radios de tolerancia respecto a la altura de la cama
umbral_pared       = 20      # Radios de proximidad requerida a las paredes laterales

# -----------------------------------------------------------------------------
# RUTAS DE SALIDA (CARBOXI)
# -----------------------------------------------------------------------------
[rutas]
resultados = "./resultados"
imagenes   = "./resultados/Imagenes"

**Tool Call: Programas/cluster/scripts/collision_erosion_bed.input**
Status: Completed

# Init

variable data index 1

variable xrotate index 1
variable yrotate index 1
variable zrotate index 1
variable angle index 1

variable xmove index 1
variable ymove index 1
variable zmove index 1

variable velocity index 1
variable nsteps index 1
variable ndumps index 1
variable nthermo index 1

variable proc_x index 1
variable proc_y index 1
variable proc_z index 1

variable timestep index 1

variable young_modulus index 1
variable poisson index 1
variable surface_energy index 1
variable density_proj index 1

newton off

units       si
dimension   3
boundary    p p f
atom_style  sphere
comm_style tiled
neighbor  200e-9 bin
neigh_modify delay 5 every 1 check yes
comm_modify mode single vel yes cutoff 1.8e-6
processors ${proc_x} ${proc_y} ${proc_z}

# Atom definition
read_data ${data}
#read_data ./box_fill_0.35_SI.data

#TODO: EDITAR TAMAÑO DE CAJA EN DATA y NUMERO DE GRANOS a 2

change_box all z final -1e-5 0.000410
#TODO: GRANOS EN POSICION original con centro de la caja en 0 0 0

#1 1 1.52e-6 2000 0.76e-6 0 0
#2 1 1.52e-6 2000 -0.76e-6 0 0
#3 1 1.52e-6 2000 0 1.32e-6 0

##### crea granos del proyectil
#NP1
create_atoms 2 single 50e-6 50e-6 152e-6

#NP3
#create_atoms 2 single 50.76e-6 50e-6 152e-6
#create_atoms 2 single 49.24e-6 50e-6 152e-6
#create_atoms 2 single 50e-6 51.316e-6 152e-6

set type 2 diameter 1.52e-6
set type 2 density ${density_proj}

# Groups
group bloque type 1
group sphere type 2

#TODO: CUIDADO CON EL CENTRO DE LA ROTACION, SI CAMBIA EL TAMAÑO DE LA CAJA HAY Q CAMBIAR EL CENTRO
#displace_atoms sphere rotate 50e-6 50e-6 152e-6 ${xrotate} ${yrotate} ${zrotate} ${angle}
displace_atoms sphere move ${xmove} ${ymove} ${zmove}

write_data pre-run_with_projectile.data

#Settings
pair_style granular
pair_coeff * * jkr ${young_modulus} 1.0e16 ${poisson} ${surface_energy} tangential mindlin NULL 1 0.5 rolling sds 5e09 9.895e-09 0.1 twisting marshall

timestep ${timestep}

# fixes & compute
fix 1 bloque nve/sphere
fix 2 sphere nve/sphere

compute 1 all ke/atom

velocity bloque create 10.0 83921
velocity sphere set 0.0 0.0 ${velocity} units box

thermo ${nthermo}
thermo_modify lost warn

#dump 1 all custom ${ndumps} output.*.gz id type x y z vx vy vz fz
dump 1 all custom ${ndumps} output_trayectoria.dump id type x y z vx vy vz fz
# ==========================================================
# TRAMPA: Congela partículas sin borrarlas y sin frenar el run
# ==========================================================

# 1. Definir la región a partir de los 250 um (2.5e-4 m)
region r_trampa block INF INF INF INF 2.5e-4 INF

# 2. Grupo dinámico: LAMMPS revisa cada 100 pasos quién entró a la región.
# (Hacerlo cada 10000 era peligroso, alguna podía rebotar y caer antes de ser detectada)
group gr_trampa dynamic all region r_trampa every 1000

# 3. Anular todas las fuerzas para que dejen de sentir gravedad o colisiones
fix freno_fuerza gr_trampa setforce 0.0 0.0 0.0

# 4. Drenar la velocidad instantáneamente
fix freno_vel gr_trampa viscous 1.0e-8
#fix balanceo all balance 1000 1.05 shift xy 10 1.05
# Mantienes la trampa intacta y agregas el balanceo RCB
fix balanceo all balance 1000 1.05 rcb weight time 1.0
restart 50000 res_A.rst res_B.rst
run ${nsteps}
# Guardado final de seguridad
write_data post-run_final.data

## Assistant

<thinking>
Now let me understand the broader context. Let me look at the whole directory structure to understand where things are and what else exists. Let me explore the Tesis directory tree and related analysis scripts (CARBOXI mentioned in config).

The user wants a plan for "orden y unificación" (order and unification) of these scripts. Currently, simulations run by calling `encolar_sims.sh` multiple times (submitting one Slurm job each time) until the user wants to stop. The concern seems to be that there's no automated way to enqueue multiple simulations/jobs with different configs.

Let me explore the directory more to understand the structure, and check memory for prior context on this project.
</thinking>

**Tool Call: ls -la /home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/**
Status: Completed

total 168
drwxrwxr-x  5 tutturuu tutturuu  4096 ago  5 15:24 .
drwxrwxr-x 12 tutturuu tutturuu  4096 ago  5 14:25 ..
drwxrwxr-x  2 tutturuu tutturuu  4096 ago  5 14:59 datas
drwxrwxr-x  6 tutturuu tutturuu  4096 ago  5 12:19 datos
-rw-rw-r--  1 tutturuu tutturuu 75364 ago  3 14:00 erosion_49.py
-rw-rw-r--  1 tutturuu tutturuu 72050 jul 28 21:40 erosion.py
drwxrwxr-x  2 tutturuu tutturuu  4096 ago  5 15:24 scripts


**Tool Call: ls -R /home/tutturuu/Documentos/Universidad/Tesis/Programas/ 2>/dev/null | head -100**
Status: Completed

/home/tutturuu/Documentos/Universidad/Tesis/Programas/:
ANALISIS
CARBOXI
cluster
comparar_filtros
Ideas_de_implementacion
legacy
LENGA
resultados

/home/tutturuu/Documentos/Universidad/Tesis/Programas/ANALISIS:
cluster_calc.py
confidence_intervals.py
constants.py
data_loader.py
erosion_viz.py
__init__.py
kinematics_calc.py
main.py
__pycache__

/home/tutturuu/Documentos/Universidad/Tesis/Programas/ANALISIS/__pycache__:
kinematics_calc.cpython-313.pyc

/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI:
carboxi.egg-info
config_and_discovery.py
config.toml
erosion_analysis.py
__init__.py
main_pipeline.py
__main__.py
merge_all.py
__pycache__
pyproject.toml
storage_and_viz.py
trajectory_converter.py
wave_analysis.py

/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/carboxi.egg-info:
dependency_links.txt
PKG-INFO
requires.txt
SOURCES.txt
top_level.txt

/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/__pycache__:
browser.cpython-311.pyc
browser.cpython-313.pyc
browser.cpython-37.pyc
config_and_discovery.cpython-313.pyc
config.cpython-311.pyc
config.cpython-313.pyc
discovery.cpython-311.pyc
discovery.cpython-313.pyc
erosion_analysis.cpython-313.pyc
__init__.cpython-311.pyc
__init__.cpython-313.pyc
__init__.cpython-37.pyc
__main__.cpython-311.pyc
__main__.cpython-313.pyc
__main__.cpython-37.pyc
main_pipeline.cpython-313.pyc
pipeline.cpython-311.pyc
pipeline.cpython-313.pyc
plot.cpython-311.pyc
plot.cpython-313.pyc
storage_and_viz.cpython-313.pyc
trajectory_converter.cpython-313.pyc
utils.cpython-311.pyc
utils.cpython-313.pyc
wave_analysis.cpython-313.pyc

/home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster:
datas
datos
erosion_49.py
erosion.py
scripts

/home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/datas:
box_fill_0.15_SI_100um_180_1_0_0.data
box_fill_0.15_SI_100um_270_0_1_0.data
box_fill_0.15_SI_100um_270_1_0_0.data
box_fill_0.15_SI_100um_90_0_1_0.data
box_fill_0.15_SI_100um_90_1_0_0.data
box_fill_0.15_SI_100um.data
box_fill_0.15_SI_145um.data
box_fill_0.35_SI_100um.data
box_fill_0.35_SI_145um.data

/home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/datos:
resultados_49_mov
resultados_9_mov
resultados_bug
resultados_emma

/home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/datos/resultados_49_mov:
merge_resultados.csv



**Tool Call: engram_mem_context**
Status: Completed

## Memory from Previous Sessions

### Recent Sessions
- **Tesis** (2026-08-05 15:26:33) [0 observations]
- **Tesis** (2026-04-04 03:05:26) [49 observations]
- **Tesis** (2026-08-03 13:47:40) [0 observations]
- **Tesis** (2026-08-01 19:34:39) [0 observations]
- **Tesis** (2026-07-28 15:05:29) [0 observations]

### Recent Observations
- [session_summary] **Session summary: Tesis**: ## Goal
Documentar, renombrar y archivar scripts de prueba del proyecto de simulación de erosión granular (Tesis), y limpiar/depurar el proyecto `comparar_filtros`.

## Instructions
- El usuario habla español; responder en español.
- Convención en `Programas/legacy/`: prefijo `prueba_*` para scripts...
- [pattern] **Archivados 3 scripts en legacy/ con documentación**: **What**: Se movieron y documentaron 3 scripts de prueba del proyecto de simulación de erosión a legacy/ con nombres descriptivos: cluster/prueba.py -> legacy/prueba_distribucion_cluster_por_velocidad.py; cluster/profundidad_max.py -> legacy/prueba_profundidad_max_proyectil.py; cluster/verificar_sum...
- [session_summary] **Session summary: Tesis**: ## Goal
Added comparative analysis (9 vs 245 reps) functions and removed all seaborn dependency from erosion_49.py, replacing with pure matplotlib.

## Instructions
- Gráficos con matplotlib puro, sin seaborn
- Límites de gráficos no se modifican
- `import erosion as erosion_9` para reutilizar funci...
- [session_summary] **Session summary: Tesis**: ## Goal
Add comparative analysis functions to `erosion_49.py` that compare 9 vs 245 repetitions (Np:1, φ=0.15, v=50 m/s): a LaTeX summary table and a combined P(k) cluster distribution plot.

## Instructions
- Límites de gráficos NO se modifican
- `import erosion as erosion_9` for reusing functions ...
- [session_summary] **Session summary: Tesis**: ## Goal
Refactorizar el ajuste de supervivencia: fit directo de CCDF a datos de P_surv con K_max como parámetro libre, en vez del enfoque indirecto (fit PDF → CDF analítica → supervivencia normalizada)

## Instructions
- Usar `!= 1` para proyectil_erosion (float), no `~`/`not`
- No tocar errores E71...
- [session_summary] **Session summary: Tesis**: ## Goal
Agregar plot de comparación de modelos (PDF alternativos) a erosion_49.py con power_law, astrom y double_exponential, y corregir error de KeyError en métricas

## Discoveries
- `ajuste_modelos` solo fiteaba power_law, no los otros 2 modelos agregados al dict
- `ajustes_raw` y `ajustes_format...
- [session_summary] **Session summary: Tesis**: ## Goal
Corregir NP 3 → NP 1 en datos y código, cambiar ángulo 90 → 0, y crear tabla resumen LaTeX formateada para erosion_49.py

## Instructions
- El dataset tiene `proyectil_erosion` como float (0.0/1.0), no booleano — usar `!= 1` en vez de `~`
- Ruff con errores pre-existentes E712 y F841 que no ...
- [session_summary] **Session summary: Tesis**: ## Goal
Arreglar animaciones rotas: ejes incorrectos (timestep como x-axis), falta animación de penetración de proyectiles

## Instructions
- Animaciones deben coincidir con los plots estáticos en ejes y escalas
- Eliminar animaciones que no tenían contraparte estática (growth curves, cluster growth...
- [session_summary] **Session summary: Tesis**: ## Goal
Crear archivo de animaciones que genere videos de cada imagen evolucionando en el tiempo (timesteps 0-80)

## Accomplished
- ✅ Creado `animaciones.py` en `Programas/cluster/` con 4 tipos de animación
- ✅ **24 heatmaps**: animación por celda del yield (targets - proyectiles) para cada combina...
- [session_summary] **Session summary: Tesis**: ## Goal
Investigar por qué la celda (0,-1) mostraba yield=0 en el heatmap de NP3 V angle=90 v=50 ff=0.15, cuando la simulación cruda sí tenía erosión ahí.

## Discoveries
- En el CSV mergeado, la celda (0,-1) tiene CERO átomos en cualquier timestep para NP3 V v=50 ff=0.15, mientras que todas las dem...
- [session_summary] **Session summary: Tesis**: ## Goal
Unificar CARBOXI + ANALISIS + cluster/ en BPM_toolkit con CLI unificado

## Instructions
- Usar subagentes para implementación paralela
- Mantener SOLID/KISS/YAGNI + docstrings Google
- Ruff check debe pasar

## Discoveries
- polars unificado reemplaza pandas
- CLI con 5 comandos: dry-run, r...
- [bugfix] **Fixed erosion particle filtering bug in cluster analysis**: **What**: Identified and fixed critical bug in erosion particle filtering

**Why**: The pipeline was analyzing non-eroded particles instead of eroded particles, causing incorrect N values and misleading results

**Where**: 
- ANALISIS/cluster_calc.py: line 67 in compute_unique_ids_raw_with_positions...
- [session_summary] **Session summary: Tesis**: ## Goal
Refactorización completa del pipeline de análisis de erosión para simplificar visualizaciones y tablas de intervalos de confianza

## Instructions
- Mantener solo lo esencial en ANALISIS/ (pipeline principal)
- Mover visualizaciones complejas a legacy/pdf_quitadas.py
- Usar formato uncertain...
- [decision] **Refactorización pipeline erosión - separación core/legacy**: **What**: Refactorización completa del pipeline de análisis de erosión para simplificar y separar responsabilidades

**Why**: Usuario solicitó simplificar el pipeline moviendo visualizaciones complejas a un script legacy independiente y simplificar las tablas de intervalos de confianza

**Where**: 
...
- [bugfix] **Simplificadas tablas IC - solo MAD + SE**: **What**: Simplificadas tablas de IC - solo MAD, solo error estándar (no Student/Bootstrap/Binomial/Tukey)

**Why**: Usuario solicitó simplificar las tablas de intervalos de confianza

**Where**: 
- ANALISIS/confidence_intervals.py - eliminados Tukey, Binomial, Bootstrap
- Agregada variable USE_STUD...
- [session_summary] **Session summary: Tesis**: ## Goal
Pipeline de análisis de erosión completo con intervalos de confianza funcionando

## Instructions
- Engram se consulta antes de editar archivos existentes
- Grid de posiciones calculado una vez en main.py
- I_VALUE=80 hardcodeado en main.py (no dinámico)

## Discoveries
- Pipeline ejecuta si...
- [decision] **Pipeline ICs - sesión completa**: ## Goal
Implementar pipeline de intervalos de confianza con cálculos correctos de err%

## Discusiones y Decisiones

### 1. Opción A vs B para datos con ceros
- **Opción A**: Solo posiciones con erosión (sin ceros)
- **Opción B**: Incluir ceros en los datos
- **Decisión**: Volver a Opción A porque O...
- [feature] **Err% consistente con rango**: **What**: Fórmulas consistentes para err% cuando media=0
**Why**: Antes cada método usaba fórmula diferente (IQR, std), ahora todos usan rango como denominador
**Where**: ANALISIS/confidence_intervals.py - compute_student_ci, compute_binomial_ci, compute_bootstrap_ci
**Fórmula**: Si μ=0: err% = σ/ra...
- [feature] **Filtros MAD/Tukey siempre completan a n=10**: **What**: Filtros MAD/Tukey ahora siempre añaden ceros hasta n=10
**Why**: Antes solo añadían ceros si había outliers, ahora siempre completan a 10 datos
**Where**: ANALISIS/confidence_intervals.py: filter_mad(), filter_tukey()
**Cambio**: n_zeros = 10 - len(filtered) en vez de n_outliers
- [bugfix] **Heatmap cuenta partículas válidas**: **What**: Corregido heatmap para contar solo partículas válidas
**Why**: Heatmap contaba TODAS las partículas (type 1 + 2), pero ICs usaban solo erosionadas (type=2, z>1e-4). Ahora heatmap filtra igual: type=1 O (type=2 Y z>1e-4)
**Where**: ANALISIS/kinematics_calc.py:compute_heatmap_data() - línea ...


---
Memory stats: 113 sessions, 243 observations across projects: Tesis, tesis-bazzoni, CARBOXI, tesis-sims, cluster, Universidad, BPM_toolkit, ANALISIS, espectrometro, sims, CARBOXI_MINI

## Assistant

<thinking>
Let me look at more context. Let me look at the CARBOXI config_and_discovery.py to understand how it discovers sims, and understand the whole workflow. Also let me understand what the user means by "orden y unificación" — the sims are run by submitting `encolar_sims.sh` multiple times manually, until the user wants to stop.

Let me explore a few more things:
1. CARBOXI config_and_discovery.py — how results are discovered
2. The results directory structure 
3. Any existing docs on the workflow
4. git status/log for context

Let me look at the broader repo.
</thinking>

**Tool Call: ls -la /home/tutturuu/Documentos/Universidad/Tesis/ && ls -la /home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/configs 2>/dev/null | head -30**
Status: Completed

total 68
drwxrwxr-x  6 tutturuu tutturuu  4096 jun  5 14:20 .
drwxrwxr-x  5 tutturuu tutturuu  4096 abr 21 19:15 ..
drwxrwxr-x  5 tutturuu tutturuu  4096 ago  4 14:39 Documentos
drwxrwxr-x  8 tutturuu tutturuu  4096 ago  5 15:26 .git
-rw-rw-r--  1 tutturuu tutturuu   297 abr 21 17:28 .gitignore
-rw-rw-r--  1 tutturuu tutturuu 35149 feb 26 22:32 LICENSE
drwxrwxr-x 12 tutturuu tutturuu  4096 ago  5 14:25 Programas
-rw-rw-r--  1 tutturuu tutturuu   661 feb 26 22:32 README.md
drwxrwxr-x  3 tutturuu tutturuu  4096 jun  5 14:20 .ruff_cache


**Tool Call: git -C /home/tutturuu/Documentos/Universidad/Tesis status && git -C /home/tutturuu/Documentos/Universidad/Tesis log --oneline -10**
Status: Completed

En la rama desarrollo
Tu rama está adelantada a 'origin/desarrollo' por 2 commits.
  (usa "git push" para publicar tus commits locales)

Cambios no rastreados para el commit:
  (usa "git add/rm <archivo>..." para actualizar a lo que se le va a hacer commit)
  (usa "git restore <archivo>..." para descartar los cambios en el directorio de trabajo)
	borrados:        "Documentos/Documentos Tesis/Formulario_presentaci\303\263n_plan_de_labor.docx"
	borrados:        "Documentos/Documentos Tesis/Nota lugar de ejecuci\303\263n del seminario de tesis.docx"
	borrados:        Programas/AGENTS.md
	modificados:     Programas/ANALISIS/erosion_viz.py
	modificados:     Programas/CARBOXI/__pycache__/__init__.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/__main__.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/config_and_discovery.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/erosion_analysis.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/main_pipeline.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/storage_and_viz.cpython-313.pyc
	modificados:     Programas/CARBOXI/__pycache__/wave_analysis.cpython-313.pyc
	modificados:     Programas/CARBOXI/config_and_discovery.py
	borrados:        Programas/a.html
	borrados:        Programas/app.py
	borrados:        Programas/atlas.py
	borrados:        Programas/atlas_analisis.html
	borrados:        Programas/atlas_interactivo.html
	borrados:        Programas/cluster/collision_erosion_bed.input
	borrados:        Programas/cluster/config.toml
	borrados:        Programas/cluster/read_config.py
	borrados:        Programas/cluster/run_erosion_bed_sims_lammps_move_np3.sh
	borrados:        Programas/cluster/slurm_submit_JKR.sh
	modificados:     Programas/comparar_filtros/__pycache__/filtros.cpython-313.pyc
	modificados:     Programas/comparar_filtros/__pycache__/lammps_reader.cpython-313.pyc
	modificados:     Programas/comparar_filtros/__pycache__/main.cpython-313.pyc
	modificados:     Programas/comparar_filtros/__pycache__/metricas.cpython-313.pyc
	modificados:     Programas/comparar_filtros/__pycache__/plots.cpython-313.pyc
	borrados:        Programas/comparar_filtros/app.py
	modificados:     Programas/comparar_filtros/filtros.py
	modificados:     Programas/comparar_filtros/lammps_reader.py
	modificados:     Programas/comparar_filtros/main.py
	modificados:     Programas/comparar_filtros/metricas.py
	modificados:     Programas/comparar_filtros/plots.py
	modificados:     Programas/comparar_filtros/run_all.py
	borrados:        Programas/diagrama_main.html
	borrados:        Programas/flujo_main.txt
	borrados:        Programas/scripts/aggregate_parquets.py
	borrados:        Programas/scripts/crater_analisis.py
	borrados:        Programas/scripts/crater_volume_plot.png
	borrados:        Programas/scripts/fabric_tensor.py
	borrados:        Programas/scripts/fabric_tensor_v2.py
	borrados:        Programas/scripts/fabric_tensor_v3.py
	borrados:        Programas/scripts/fabric_tensor_v4.py
	borrados:        Programas/scripts/fabric_tensor_v5.py
	borrados:        Programas/scripts/plot_fabric.py
	borrados:        Programas/scripts/plot_fabric_tensor.png

Archivos sin seguimiento:
  (usa "git add <archivo>..." para incluirlo a lo que será confirmado)
	Documentos/Imagenes/
	Documentos/resultados/
	Programas/Ideas_de_implementacion/
	Programas/cluster/erosion.py
	Programas/cluster/erosion_49.py
	Programas/cluster/scripts/
	Programas/comparar_filtros/simulaciones/
	Programas/comparar_filtros/solo_graficos.py
	Programas/legacy/final/
	Programas/legacy/heatmap_ff_0.15_v-10.png
	Programas/legacy/heatmap_ff_0.15_v-25.png
	Programas/legacy/heatmap_ff_0.15_v-5.png
	Programas/legacy/heatmap_ff_0.15_v-50.png
	Programas/legacy/heatmap_ff_0.35_v-10.png
	Programas/legacy/heatmap_ff_0.35_v-25.png
	Programas/legacy/heatmap_ff_0.35_v-5.png
	Programas/legacy/heatmap_ff_0.35_v-50.png
	Programas/legacy/prueba_distribucion_cluster_por_velocidad.py
	Programas/legacy/prueba_plot_heatmap.py
	Programas/legacy/prueba_profundidad_max_proyectil.py
	Programas/legacy/resume_collision.input
	Programas/legacy/resume_controller.py
	Programas/legacy/slurm_resume.sh
	Programas/legacy/verificar_normalizacion_Pk_sin_relleno.py
	Programas/resultados/

sin cambios agregados al commit (usa "git add" y/o "git commit -a")
9a0e9c0 merge
0e0052a descarte
e0ef586 solucion estadistica al filtrado de datos
8627133 debug de analisis
73b0984 Merge branch 'desarrollo' of github.com:FabriTape/Tesis into desarrollo
9461482 debug de analisis
395d0b6 Merge branch 'desarrollo' of github.com:FabriTape/Tesis into desarrollo
c8f101e ediciones de parte del cluster para seguir la version en computadora
8a78ab0 Ediciones en ANALISIS para que dibuje bien los 0 y la estadistica
60e01e4 refactor: LammpsTrajectoryReader con auto-detección de formato


**Tool Call: Programas/CARBOXI/config_and_discovery.py**
Status: Completed

"""
Carga de configuración y descubrimiento de simulaciones LAMMPS.

Este módulo combina:
- Carga de parámetros desde config.toml
- Descubrimiento de carpetas de simulación
- Extracción de metadatos desde nombres de carpeta
- Navegación interactiva de carpetas (opcional)

Física: Los parámetros definen umbrales para detección de onda de choque,
criterios de erosión de la cama, propiedades del material sílice y
configuración del modelo JKR.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

_SIM_PREFIX: str = "erosion_"

_PATRONES: dict[str, str] = {
    "ff": r"ff_([^_]+)",
    "np": r"np_([^_]+)",
    "vel0": r"vel_([^_]+)",
    "angulo": r"angle_([^_]+)",
    "move": r"move_([^_]+)_([^_]+)_([^_]+)",
}


@dataclass(frozen=True)
class WaveConfig:
    """Parámetros para la detección automática de onda de choque.

    Attributes:
        v_objetivo: Velocidad de impacto objetivo [m/s].
        tolerancia_v: Margen de error para comparar velocidades [m/s].
        np_criticos: Cantidades de partículas proyectil que activan el análisis
            automáticamente (sin verificar velocidad).
        move_referencia: Vector de posición de reposo del proyectil [m, m, m].
        tolerancia_move: Tolerancia para considerar desplazamiento no nulo [m].
        umbral_v2_min: Energía cinética mínima para partícula activa [m²/s²].
        umbral_v2_max: Energía cinética máxima para impacto crítico [m²/s²].
        particulas_criticas: Cantidad de partículas para onda macroscópica.
        umbral_techo: Margen vertical en número de radios respecto a la altura.
        umbral_pared: Radios de proximidad requerida a las paredes laterales.
    """

    v_objetivo: float
    tolerancia_v: float
    np_criticos: tuple[int, ...]
    move_referencia: tuple[float, ...]
    tolerancia_move: float
    umbral_v2_min: float
    umbral_v2_max: float
    particulas_criticas: int
    umbral_techo: int
    umbral_pared: int


@dataclass(frozen=True)
class FilterConfig:
    """Umbrales de filtrado para el criterio de erosión de partículas.

    Attributes:
        umbral_vz_min: Velocidad vertical mínima para considerar eyección [m/s].
        umbral_techo_radios: Margen de seguridad vertical en número de radios.
        tolerancia_v_cm: Tolerancia de |Δv_cm| entre dumps consecutivos [m/s].
        filtro_activo: Nombre del filtro de erosión (F3C).
    """

    umbral_vz_min: float
    umbral_techo_radios: float
    tolerancia_v_cm: float
    filtro_activo: str = "F3C"


@dataclass(frozen=True)
class Config:
    """Configuración completa del programa CARBOXI.

    Attributes:
        radio_m: Radio de las partículas en metros [m].
        usar_caja: Si True, usa los límites fijos de la caja.
        bed_height_ff_015: Altura de la cama para ff=0.15 [m].
        bed_height_ff_035: Altura de la cama para ff=0.35 [m].
        caja_x: Dimensión de la caja en X [m].
        caja_y: Dimensión de la caja en Y [m].
        freeze_height: Altura de congelamiento para filtro F3C [m].
        wave: Parámetros de detección de onda de choque.
        filters: Umbrales de filtrado para criterio de erosión.
        factor_cutoff: Factor multiplicador sobre el radio para cutoff de OVITO.
        dir_resultados: Directorio raíz de salida de resultados.
        dir_imagenes: Subdirectorio de salida para gráficos SVG.
    """

    radio_m: float
    usar_caja: bool
    bed_height_ff_015: float
    bed_height_ff_035: float
    caja_x: float
    caja_y: float
    freeze_height: float
    wave: WaveConfig
    filters: FilterConfig
    factor_cutoff: float
    dir_resultados: Path
    dir_imagenes: Path


def load_config(path: Path = Path("config.toml")) -> Config:
    """Carga y parsea el archivo de configuración TOML embebido.

    Args:
        path: Ruta al archivo config.toml (ignorada, se usa el embebido).

    Returns:
        Config: Instancia inmutable con todos los parámetros de análisis.

    Raises:
        FileNotFoundError: Si el config.toml embebido no se encuentra.
        KeyError: Si faltan secciones obligatorias en el TOML.
    """
    resolved = _resolve_config_path(path)

    with resolved.open("rb") as f:
        raw: dict = tomllib.load(f)

    return _build_config(raw)


def _resolve_config_path(path: Path) -> Path:
    """Resuelve la ruta del archivo config.toml usando siempre el embebido."""
    module_config = Path(__file__).parent / "config.toml"
    if module_config.exists():
        return module_config

    raise FileNotFoundError(f"config.toml embebido no encontrado en '{module_config}'.")


def _build_config(raw: dict) -> Config:
    """Construye la instancia Config a partir del diccionario TOML."""
    radio_um: float = raw["defaults"]["radio_um"]

    wave_raw = raw["analisis_onda"]
    onda_raw = raw["onda"]
    wave = WaveConfig(
        v_objetivo=float(wave_raw["v_objetivo"]),
        tolerancia_v=float(wave_raw["tolerancia_v"]),
        np_criticos=tuple(int(n) for n in wave_raw["np_criticos"]),
        move_referencia=tuple(float(v) for v in wave_raw["move_referencia"]),
        tolerancia_move=float(wave_raw["tolerancia_move"]),
        umbral_v2_min=float(onda_raw["umbral_v2_min"]),
        umbral_v2_max=float(onda_raw["umbral_v2_max"]),
        particulas_criticas=int(onda_raw["particulas_criticas"]),
        umbral_techo=int(onda_raw["umbral_techo"]),
        umbral_pared=int(onda_raw["umbral_pared"]),
    )

    filtros_raw = raw["filtros"]
    filters = FilterConfig(
        umbral_vz_min=float(filtros_raw["umbral_vz_min"]),
        umbral_techo_radios=float(filtros_raw["umbral_techo_radios"]),
        tolerancia_v_cm=float(filtros_raw.get("tolerancia_v_cm", 0.0)),
        filtro_activo=str(filtros_raw.get("filtro_activo", "F3C")),
    )

    rutas = raw["rutas"]
    return Config(
        radio_m=radio_um * 1e-6,
        usar_caja=bool(raw["defaults"].get("usar_caja", True)),
        bed_height_ff_015=float(raw["defaults"]["bed_height_ff_015"]),
        bed_height_ff_035=float(raw["defaults"]["bed_height_ff_035"]),
        caja_x=float(raw["defaults"]["caja_x"]),
        caja_y=float(raw["defaults"]["caja_y"]),
        freeze_height=float(raw["defaults"]["freeze_height"]),
        wave=wave,
        filters=filters,
        factor_cutoff=float(raw["jkr"]["factor_cutoff"]),
        dir_resultados=Path(rutas["resultados"]),
        dir_imagenes=Path(rutas["imagenes"]),
    )


_EROSION_PREFIX: str = "erosion_bed_SI_"


def discover_simulations(
    base_dir: Path, prefix: str = _SIM_PREFIX, type: str = "dir"
) -> list[Path]:
    """Encuentra las carpetas de simulación a procesar desde un directorio base.

    Args:
        base_dir: Directorio raíz donde buscar.
        prefix: Prefijo de las carpetas de simulación a buscar.
        type: Tipo de simulación a buscar ("dir" para carpetas, "file" para archivos).

    Returns:
        Lista ordenada de rutas a las carpetas de simulación encontradas.

    Raises:
        SystemExit: Si no se encuentra ninguna simulación válida.
    """
    if base_dir.name.startswith(prefix):
        sims = sorted(
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(_EROSION_PREFIX)
        )
        if sims:
            print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
            return sims
        return [base_dir]
    if type == "dir":
        sims = sorted(
            p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
        )
    else:
        sims = sorted(
            p for p in base_dir.iterdir() if p.is_file() and p.name.startswith(prefix)
        )

    if not sims:
        print(
            f"Error: No se encontraron {'carpetas' if type == 'dir' else 'archivos'} '{prefix}*' en '{base_dir}'.\n"
            "Verifique que la ruta seleccionada sea correcta."
        )
        sys.exit(1)

    print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
    return sims


def extraer_metadatos(sim_id: str) -> dict[str, str]:
    """Extrae parámetros físicos de la simulación desde el identificador de carpeta.

    Args:
        sim_id: Identificador de la simulación (sin prefijo erosion_bed_SI_).

    Returns:
        Diccionario con claves: ff, np, vel0, angulo, posx, posy, posz.
    """
    metadatos: dict[str, str] = {}

    for clave, patron in _PATRONES.items():
        match = re.search(patron, sim_id)
        if match:
            if clave == "move":
                metadatos["posx"] = match.group(1)
                metadatos["posy"] = match.group(2)
                metadatos["posz"] = match.group(3)
            else:
                metadatos[clave] = match.group(1)
        else:
            print(f"Advertencia: parámetro '{clave}' no encontrado en '{sim_id}'.")
            if clave == "move":
                metadatos.update({"posx": "0", "posy": "0", "posz": "0"})
            else:
                metadatos[clave] = "N/A"

    return metadatos


def print_progress(current: int, total: int, label: str = "análisis") -> None:
    """Actualiza en línea una barra de progreso ASCII en la terminal.

    Args:
        current: Índice de la iteración actual (base 0).
        total: Número total de iteraciones esperadas.
        label: Etiqueta descriptiva del proceso en curso.
    """
    if total == 0:
        return

    pct = int(current * 100 / total)
    filled = pct // 5
    bar = "_" * filled + "@ï" + "·" * (18 - filled)

    if current == 0:
        print(f"Iniciando análisis de {label}...")

    print(f"\r{pct:03d}/100% [{bar}] ({label})", end="", flush=True)


def browse_for_simulation() -> list[Path]:
    """Muestra un menú interactivo para navegar el sistema de archivos y
    seleccionar la carpeta de simulación a procesar.

    Returns:
        Lista de rutas absolutas a las carpetas de simulación seleccionadas.
    """
    current = Path(".").resolve()

    while True:
        subdirs = _list_subdirs(current)
        erosion_dirs = [p for p in subdirs if p.name.startswith(_SIM_PREFIX)]
        dump_files = list(current.glob("output.*.gz"))

        _print_browser_header(current, subdirs, erosion_dirs, dump_files)

        try:
            raw = input("Elección: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return []

        if raw == "0":
            return _select_current(current, subdirs)

        if raw == "-1":
            current = current.parent
            _clear()
            continue

        try:
            idx = int(raw)
        except ValueError:
            _clear()
            print("Entrada inválida — ingrese un número de la lista.\n")
            continue

        if 1 <= idx <= len(subdirs):
            current = subdirs[idx - 1]
            _clear()
        else:
            _clear()
            print("Número fuera de rango.\n")


def _list_subdirs(directory: Path) -> list[Path]:
    """Lista subdirectorios visibles (no ocultos) de un directorio."""
    try:
        return sorted(
            p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    except PermissionError:
        print(f"  Sin permiso para leer '{directory}'.")
        return []


def _print_browser_header(
    current: Path,
    subdirs: list[Path],
    erosion_dirs: list[Path],
    dump_files: list[Path],
) -> None:
    """Imprime la cabecera del menú de navegación."""
    print(f"\n--- Ubicación actual: {current} ---")

    if dump_files:
        print(f"  ✓ {len(dump_files)} archivos output.*.gz")
    if erosion_dirs:
        print(f"  ✓ {len(erosion_dirs)} carpeta(s) 'erosion_bed_SI_*'")

    print()
    print("[0]  Seleccionar esta carpeta")
    print("[-1] Subir un nivel")
    print()

    for i, subdir in enumerate(subdirs, start=1):
        tag = " ★" if subdir.name.startswith(_SIM_PREFIX) else ""
        print(f"[{i}] {subdir.name}{tag}")

    print()


def _select_current(current: Path, subdirs: list[Path]) -> list[Path]:
    """Procesa la selección de la carpeta actual por el usuario."""
    erosion_subs = [p for p in subdirs if p.name.startswith(_SIM_PREFIX)]

    if erosion_subs:
        print(
            f"\n→ Se procesarán {len(erosion_subs)} subcarpetas "
            f"'erosion_bed_SI_*' en '{current.name}'."
        )
        return sorted(erosion_subs)

    return [current]


def _clear() -> None:
    """Limpia la terminal de forma multiplataforma."""
    import os

    os.system("cls" if os.name == "nt" else "clear")

## Assistant

<thinking>
Now let me understand the current workflow better. Let me look at the config.toml in CARBOXI, the merger (merge_all.py), and check if there are any existing notes/docs about how the workflow works. Also I notice there's `Programas/cluster/` with old scripts and `Programas/cluster/scripts/` which is the new one. There's also `legacy/slurm_resume.sh` and `legacy/resume_controller.py` and `legacy/resume_collision.input` — suggesting there was an attempt at a "resume" mechanism.

Let me look at the legacy resume stuff and understand the workflow better. Also let me understand the whole simulation workflow: the OUTPUT goes to Buffer/sims_SI_erosion_bed_100um_... Let me look at the results and how they're structured.

Let me also check the Documentos folder for any plan or workflow documentation.
</thinking>

**Tool Call: Programas/legacy/slurm_resume.sh**
Status: Completed

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

**Tool Call: Programas/legacy/resume_controller.py**
Status: Completed

"""
Controlador para retomar simulaciones incompletas de LAMMPS.
Escanea la carpeta 'Buffer/sims_SI...', identifica carpetas de simulación
sin 'post-run_final.data' pero con archivos restart, y las reanuda.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(
        description="Controlador para retomar simulaciones incompletas de LAMMPS."
    )
    parser.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Muestra las acciones que se tomarían sin ejecutar",
        default=False,
    )
    args = parser.parse_args()

    # Asume que se ejecuta desde la raíz del proyecto
    base_dir = Path.cwd()
    config_path = base_dir / "config.toml"
    input_path = base_dir / "resume_collision.input"
    buffer_dir = base_dir / "Buffer"

    # Fallback por si el config está en scripts/ directamente
    if not config_path.exists():
        config_path = base_dir / "config.toml"

    if not config_path.exists():
        print(f"Error: No se encontró {config_path}")
        sys.exit(1)

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Cargar variables del config.toml
    sim_config = config["simulation"]
    mat_config = config["material"]

    cores = sim_config["cores"]
    exe = sim_config["exe"]
    proc_x, proc_y, proc_z = sim_config["processors"]
    y_mod = mat_config["young_modulus"]
    poisson = mat_config["poisson"]
    surf_ener = mat_config["surface_energy"]
    ndumps = sim_config["ndumps"]
    nthermo = sim_config["nthermo"]
    default_nsteps = config["sweep"]["total_steps"][0]

    exe_path = base_dir / exe

    print("Iniciando búsqueda de simulaciones incompletas en Buffer/ ...")
    if not buffer_dir.exists():
        print("No existe el directorio Buffer.")
        return

    # 1. Iterar sobre las carpetas de los lotes (sims_SI_erosion_bed_...)
    for batch_dir in buffer_dir.glob("sims_SI_erosion_*"):
        if not batch_dir.is_dir():
            continue

        # 2. Iterar sobre las carpetas de las simulaciones individuales adentro del lote
        for sim_dir in batch_dir.glob("erosion_bed_SI_*"):
            if not sim_dir.is_dir():
                continue

            if (sim_dir / "post-run_final.data").exists():
                continue  # Esta simulación terminó correctamente

            # Buscar cuál archivo restart es el más reciente
            res_a = sim_dir / "res_A.rst"
            res_b = sim_dir / "res_B.rst"
            restart_file = None

            if res_a.exists() and res_b.exists():
                if os.path.getmtime(res_a) > os.path.getmtime(res_b):
                    restart_file = "res_A.rst"
                else:
                    restart_file = "res_B.rst"
            elif res_a.exists():
                restart_file = "res_A.rst"
            elif res_b.exists():
                restart_file = "res_B.rst"

            if restart_file:
                print(f"\n[+] Retomando: {sim_dir.name}\n    Desde: {restart_file}")

                # Intentar deducir los nsteps originales del salida.log
                local_nsteps = default_nsteps
                salida_log = sim_dir / "salida.log"
                if salida_log.exists():
                    with open(salida_log, "r") as f_log:
                        for line in f_log:
                            if "nsteps" in line and "velocity" in line:
                                parts = line.split()
                                if "nsteps" in parts:
                                    idx = parts.index("nsteps")
                                    local_nsteps = parts[idx + 1]
                                break

                cmd = [
                    "mpirun",
                    "--allow-run-as-root",
                    "-np",
                    str(cores),
                    str(exe_path),
                    "-in",
                    str(input_path),
                    "-v",
                    "restart_file",
                    restart_file,
                    "-v",
                    "young_modulus",
                    str(y_mod),
                    "-v",
                    "poisson",
                    str(poisson),
                    "-v",
                    "surface_energy",
                    str(surf_ener),
                    "-v",
                    "nsteps",
                    str(local_nsteps),
                    "-v",
                    "ndumps",
                    str(ndumps),
                    "-v",
                    "nthermo",
                    str(nthermo),
                    "-v",
                    "proc_x",
                    str(proc_x),
                    "-v",
                    "proc_y",
                    str(proc_y),
                    "-v",
                    "proc_z",
                    str(proc_z),
                ]
                if args.dry:
                    print(f"    -> Ejecutando: {cmd}")
                    print(f"Simulacion a ejecutar: {sim_dir.name}")
                else:
                    # Guardamos el output en resume.log
                    with open(sim_dir / "resume.log", "w") as log_file:
                        subprocess.run(
                            cmd, cwd=sim_dir, stdout=log_file, stderr=subprocess.STDOUT
                        )

                    print(f"    -> Finalizado: {sim_dir.name}")


if __name__ == "__main__":
    main()

**Tool Call: ls -la /home/tutturuu/Documentos/Universidad/Tesis/Programas/resultados/ /home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/ 2>/dev/null | head -60**
Status: Completed

/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/:
total 108
drwxrwxr-x  4 tutturuu tutturuu  4096 abr 21 19:32 .
drwxrwxr-x 12 tutturuu tutturuu  4096 ago  5 14:25 ..
drwxrwxr-x  2 tutturuu tutturuu  4096 abr 21 17:28 carboxi.egg-info
-rw-rw-r--  1 tutturuu tutturuu 12653 may 13 23:15 config_and_discovery.py
-rw-rw-r--  1 tutturuu tutturuu   601 abr 21 17:28 config.toml
-rw-rw-r--  1 tutturuu tutturuu  9991 abr 21 17:28 erosion_analysis.py
-rw-rw-r--  1 tutturuu tutturuu    98 abr 21 17:28 __init__.py
-rw-rw-r--  1 tutturuu tutturuu  7143 abr 21 17:28 main_pipeline.py
-rw-rw-r--  1 tutturuu tutturuu   271 abr 21 17:28 __main__.py
-rw-rw-r--  1 tutturuu tutturuu  3723 abr 21 19:32 merge_all.py
drwxrwxr-x  2 tutturuu tutturuu  4096 may 13 23:15 __pycache__
-rw-rw-r--  1 tutturuu tutturuu   372 abr 21 17:28 pyproject.toml
-rw-rw-r--  1 tutturuu tutturuu 16886 abr 21 17:28 storage_and_viz.py
-rw-rw-r--  1 tutturuu tutturuu  5302 abr 21 17:28 trajectory_converter.py
-rw-rw-r--  1 tutturuu tutturuu  6178 abr 21 17:28 wave_analysis.py

/home/tutturuu/Documentos/Universidad/Tesis/Programas/resultados/:
total 108
drwxrwxr-x  3 tutturuu tutturuu  4096 jun 15 14:15 .
drwxrwxr-x 12 tutturuu tutturuu  4096 ago  5 14:25 ..
-rw-rw-r--  1 tutturuu tutturuu 49941 may 13 23:16 ff_0.15_np_3_vel_-50_angle_0_move_1.52_1.52_0_TOTAL_F3C_2026-05-13_20-15-51.parquet
-rw-rw-r--  1 tutturuu tutturuu 18012 may 13 23:16 ff_0.35_np_1_vel_-10_angle_0_move_1.52_1.52_0_TOTAL_F3C_2026-05-13_20-16-06.parquet
-rw-rw-r--  1 tutturuu tutturuu 20845 may 13 23:17 ff_0.35_np_1_vel_-25_angle_0_move_1.52_1.52_0_TOTAL_F3C_2026-05-13_20-16-53.parquet
drwxrwxr-x  2 tutturuu tutturuu  4096 may 13 23:17 Imagenes


**Tool Call: find /home/tutturuu/Documentos/Universidad/Tesis/Documentos -maxdepth 3 -name "*.md" -o -maxdepth 3 -name "*.txt" 2>/dev/null | head -30**
Status: Completed

(no output)

## Assistant

<thinking>
Let me look at the CARBOXI config.toml, merge_all.py, and understand the whole workflow. Also let me check if there's documentation in Documentos. Let me also look at the git diff of config_and_discovery.py to see recent changes. Let me also understand the "current" workflow more thoroughly by checking the legacy resume_collision.input.

Let me understand the workflow:
1. User runs `encolar_sims.sh` from `scripts/` dir. This calls `sbatch slurm_submit_JKR.sh` which submits one job to Slurm. It copies the config.toml to `configs/config_<jobid>.toml`.
2. On the cluster node, `slurm_submit_JKR.sh` runs `run_erosion_bed_sims_lammps_move_np3.sh <jobid>`, which reads `configs/config_<jobid>.toml` via `read_config.py`, generates `config_vars_<jobid>.sh`, sources it.
3. It sweeps fills × vels × angles × moves, creating simdirs in `Buffer/sims_SI_erosion_bed_100um_...` with `erosion_bed_SI_ff_..._move_...` subdirs. If a simdir already exists (mkdir fails), it skips (that's the "ya está siendo calculada por otro nodo" dedup mechanism).
4. To run more sims (with different config, e.g. different fill fraction), the user manually edits config.toml and re-runs encolar_sims.sh.

The problem: The user must manually re-run `encolar_sims.sh` multiple times, editing config.toml between runs, "hasta que el usuario lo desee" (until the user wants). This is error-prone and manual.

So the plan is to unify/order this: submit multiple jobs automatically, one per config, from a list of experiments — a batch of configurations. Perhaps a matrix of configs, each submitted as its own Slurm job.

Let me look for more clues. Let me check the CARBOXI config.toml, merge_all.py, and any relevant documentation about the workflow (maybe in Documentos). Let me also check `Programas/AGENTS.md` which was deleted but might have hints. Actually, git shows `Programas/AGENTS.md` deleted. Let me check git history for it.

Let me look at:
- CARBOXI/config.toml
- CARBOXI/merge_all.py
- Documentos folder structure
- Whether there's a "Buffer" dir anywhere
- the git diff for config_and_discovery.py

Let me search memory for encolar_sims / cluster workflow.
</thinking>

**Tool Call: Programas/CARBOXI/merge_all.py**
Status: Completed

from pathlib import Path

import polars as pl

from .config_and_discovery import discover_simulations, extraer_metadatos


def merge_all_parquet(path_file: Path) -> pl.DataFrame:
    datos = discover_simulations(path_file,"ff_", "file")
    temp = []
    for dato in datos:
        metadatos = extraer_metadatos(str(dato.name).replace(".parquet", ""))
        df = pl.read_parquet(dato)
        df = df.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(df)
        # En merge_all_parquet, antes del concat:
    casted = []
    for df in temp:
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in [pl.UInt8, pl.UInt16, pl.UInt32]
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def merge_with_no_erosion(path_dir: Path) -> pl.DataFrame:
    """Merge de archivos parquet incluyendo marcadores de 'sin erosión'."""
    datos = list(path_dir.glob("*.parquet"))
    nada = list(path_dir.glob("*NADA*.txt"))

    temp = []
    for dato in datos:
        metadatos = extraer_metadatos(str(dato.name).replace(".parquet", ""))
        df = pl.read_parquet(dato)
        df = df.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(df)

    for marker in nada:
        marker.read_text()  # noqa: F841
        sim_id = marker.stem.replace("_NADA_F3C_", "")
        metadatos = extraer_metadatos(sim_id)
        metadatos["proyectil_erosion"] = "false"
        row = pl.DataFrame({
            "id": [0],
            "type": [0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "vx": [0.0],
            "vy": [0.0],
            "vz": [0.0],
            "fz": [0.0],
            "coordinacion": [0.0],
            "cluster": [0.0],
            "proyectil_erosion": [False],
            "paso": [0],
            "i": [0],
        })
        row = row.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(row)

    if not temp:
        return pl.DataFrame()

    casted = []
    for df in temp:
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in [pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def main() -> None:
    """Punto de entrada del conversor."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Junta todos los .parquet de un directorio en un .csv"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directorio con archivos .parquet",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Directorio de salida (default: mismo que entrada)",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Entrada no válida: {args.input} (debe ser un directorio)")
        sys.exit(1)

    output_dir = args.output or args.input  # fallback a input

#    try:
    merged_df = merge_with_no_erosion(args.input)
    if merged_df.is_empty():
        print("No se encontraron datos para fusionar")
        sys.exit(1)
    output_path = output_dir / f"merge_{args.input.name}.csv"
    merged_df.write_csv(output_path)
    print(f"Total: {len(list(args.input.glob('*.parquet')) + list(args.input.glob('*NADA*.txt')))} archivos")
    print(f"Guardado en: {output_path}")
#    except Exception as e:
#        print(f"Error: {e}")
#        sys.exit(1)


if __name__ == "__main__":
    main()

**Tool Call: Programas/CARBOXI/config.toml**
Status: Completed

[defaults]
radio_um = 0.76
usar_caja = true
bed_height_ff_015 = 1.0e-4
bed_height_ff_035 = 1.0e-4
caja_x = 1.0e-4
caja_y = 1.0e-4
freeze_height = 2.4e-4

[analisis_onda]
v_objetivo = 50.0
tolerancia_v = 5.0
np_criticos = [1, 3]
move_referencia = [0.0, 0.0, 0.0]
tolerancia_move = 1e-6

[onda]
umbral_v2_min = 0.0064
umbral_v2_max = 1.0
particulas_criticas = 2000
umbral_techo = 10
umbral_pared = 10

[filtros]
umbral_vz_min = 0.0
umbral_techo_radios = 0.0
tolerancia_v_cm = 1e-6
filtro_activo = "F3C"

[jkr]
factor_cutoff = 2.02

[rutas]
resultados = "./resultados"
imagenes = "./resultados/Imagenes"

**Tool Call: find /home/tutturuu/Documentos/Universidad/Tesis/Programas -maxdepth 2 -iname "*buffer*" -o -maxdepth 2 -iname "*AGENTS*" 2>/dev/null; echo "---"; ls /home/tutturuu/Documentos/Universidad/Tesis/Documentos/**
Status: Completed

---
Archive
errores_timeline.tex
Imagenes
linea_temporal.tex
resultados


**Tool Call: engram_mem_search**
Status: Completed

No memories found for: "encolar sims slurm cluster workflow"

## Assistant

<thinking>
Let me check the git history of the deleted AGENTS.md for hints on the workflow and conventions, and look at the git diff for config_and_discovery.py. Also let me check the Documentos/resultados for any analysis, and understand more of the workflow. Let me also check `Programas/ANALISIS/main.py` briefly to understand how data is consumed after sims, to understand the full pipeline that the "unification" should cover.

Actually, the user's request is focused: "propone un plan de orden y unificación" — propose a plan for ordering and unifying the simulation submission scripts. Currently sims are run by submitting `encolar_sims.sh` multiple times until user stops.

Let me look at the deleted AGENTS.md and any docs about the workflow. Let me use git to see AGENTS.md content.
</thinking>

**Tool Call: git -C /home/tutturuu/Documentos/Universidad/Tesis log --all --oneline -- Programas/AGENTS.md | head; echo "---"; git -C /home/tutturuu/Documentos/Universidad/Tesis show HEAD~3:Programas/AGENTS.md 2>/dev/null || git -C /home/tutturuu/Documentos/Universidad/Tesis show HEAD:Programas/AGENTS.md 2>/dev/null || echo "no AGENTS in HEAD"**
Status: Completed

bebd6b4 prueba
---
# Guía de principios SOLID, KISS y YAGNI
# Índice
- [Skills](#skills)
- [Criterios de aceptación](#criterios-de-aceptación)
- [Testing](#testing)
- [Estandares de Código y Calidad](#estándares-de-código-y-calidad)
- [MCPs](#mcp)


## Skills
El agente primero tendrá que tener en cuenta la SKILL.md que se encuentra en la carpeta ./solid/references.
Las Skills que se utilizarán en esta reestructuración son las que se encuentran en la carpeta ./solid/references
Se describen bajo que situaciones se utilizarán cada Skill.

| Situación | Skill a usar | Referencia |
| Voy a crear una función o método nuevo | SOLID (sobre todo SRP) + nombres claros y métodos cortos | references/solid-principles.md |
| Estoy decidiendo arquitectura, tecnologías | KISS – mantenerlo simple, no agregar antes de tiempo | references/complexity.md, ver también, references/achitecture.md |
| Una clase o archivo hace demasiadas cosas | Single Responsibility - dividir en clases más pequeñas | references/solid-principles.md, ver también, references/object-design.md |
| Quiero evitar código e complicado innecesario | YAGNI + KISS – no agregar cosas "por si acaso" | references/complexity.md |
| Revisando código escrito | Clean code + code smells | references/clean-code.md, ver también, references/code-smells.md |
## Criterios de aceptación
- El código cumple con el objetivo de analisis de datos y generación de figuras.
- Las funciones están acortadas, simplificadas y implementan librerías adecuadas, siguiendo los principios SOLID, KISS y YAGNI.
- Al ejecutar una revisión de lo programado con las skills no se encuentran problemas de calidad o complejidad innecesarios.
- Los docstrings generados siguen el formato Google y contienen toda la información necesaria. El docstring **debe** contener:
   - Descripción corta de qué hace (en términos del dominio: física, simulación, etc.)
   - Parámetros con tipos y unidades cuando corresponda
   - Retorno con tipo y significado físico
   - Excepciones importantes si las hay
   - Ejemplo mínimo si la lógica no es obvia

## Testing
El testing será llevado a cabo por el equipo de desarrollo.
## Estándares de Código y Calidad 
- PEP 8: Todo el código debe seguir estrictamente la guía de estilo oficial (nomenclatura snake_case, indentación de 4 espacios).
- Linter y Formateador (Ruff): Se utilizará Ruff para garantizar la limpieza del código. Esto incluye:
    - Eliminación automática de imports no usados.
    - Ordenamiento de imports (isort).
    - Corrección de espaciados y estilos según PEP 8.
- Tipado Estricto: Uso de Type Hints en todas las firmas de funciones.
- Documentación: Docstrings obligatorios explicando la lógica física de cada módulo.

## MCPs
### 1. Uso de engram (memoria persistente)

**Cuándo guardar:**

- Después de terminar la refactorización completa de un archivo entero
- Después de tomar una decisión de arquitectura importante que afecte a varios módulos  
- Despues de terminar el reconocimiento, análisis de un archivo entero y al hacer un resumen de como se relacionan los componentes.
**Qué guardar (formato mínimo y obligatorio):**

Título de la entrada:  
`[Módulo/Decisión] - [fecha corta] - Resumen`

Contenido (máximo 400–500 palabras, conciso):

- Nombre del archivo o tema de la decisión
- Resumen en 4-6 oraciones de los cambios más relevantes
- Lista corta de nombres de clases / funciones clave creadas o renombradas
- Decisiones de diseño importantes (ej: "se eligió clase Config en vez de dict global", "se eliminó back-navigation por SRP")
- Razones breves por las que se rechazó otra alternativa

**Cuándo consultar engram (obligatorio antes de cualquier edición de código):**

- Antes de empezar a trabajar en un archivo que ya existe
- Antes de proponer cualquier cambio estructural que pueda contradecir una decisión anterior
- Antes de crear una función o clase nueva que pueda duplicar algo existente

Instrucción literal que el agente debe seguir:  
“Primero busco en engram si existe entrada para este archivo o tema. Si existe, la leo completa antes de escribir cualquier línea.”

## Flujo de trabajo obligatorio

Cada vez que se edita, crea o revisa código, se deben seguir estos pasos en orden:

1. **Consultar engram** — buscar entradas previas sobre el archivo/tema antes de escribir nada
2. **Leer skill/reference** correspondiente según la situación (ver tabla Skills)
3. **Implementar** siguiendo SOLID/KISS/YAGNI
4. **Ejecutar Ruff** (`ruff check`) para verificar calidad del código
5. **Commitear** con mensaje descriptivo (qué, por qué, archivos afectados)
6. **Guardar en enram** checkpoint si fue refactorización completa o decisión de arquitectura


**Tool Call: git -C /home/tutturuu/Documentos/Universidad/Tesis diff HEAD -- Programas/CARBOXI/config_and_discovery.py | head -80**
Status: Completed

diff --git a/Programas/CARBOXI/config_and_discovery.py b/Programas/CARBOXI/config_and_discovery.py
index 76f0cca..4874044 100644
--- a/Programas/CARBOXI/config_and_discovery.py
+++ b/Programas/CARBOXI/config_and_discovery.py
@@ -20,7 +20,7 @@ import tomllib
 from dataclasses import dataclass
 from pathlib import Path
 
-_SIM_PREFIX: str = "sims_SI_"
+_SIM_PREFIX: str = "erosion_"
 
 _PATRONES: dict[str, str] = {
     "ff": r"ff_([^_]+)",


**Tool Call: ls -la /home/tutturuu/Documentos/Universidad/Tesis/Documentos/resultados/ /home/tutturuu/Documentos/Universidad/Tesis/Programas/ANALISIS/ 2>/dev/null**
Status: Completed

/home/tutturuu/Documentos/Universidad/Tesis/Documentos/resultados/:
total 44
drwxrwxr-x 5 tutturuu tutturuu  4096 abr 23 15:41 .
drwxrwxr-x 5 tutturuu tutturuu  4096 ago  4 14:39 ..
drwxrwxr-x 2 tutturuu tutturuu 12288 abr 23 18:13 0r
drwxrwxr-x 2 tutturuu tutturuu 12288 abr 23 18:13 -5r
drwxrwxr-x 2 tutturuu tutturuu 12288 abr 23 18:21 5r

/home/tutturuu/Documentos/Universidad/Tesis/Programas/ANALISIS/:
total 68
drwxrwxr-x  3 tutturuu tutturuu 4096 abr 21 19:45 .
drwxrwxr-x 12 tutturuu tutturuu 4096 ago  5 14:25 ..
-rw-rw-r--  1 tutturuu tutturuu 6664 abr 21 19:09 cluster_calc.py
-rw-rw-r--  1 tutturuu tutturuu 4982 abr 21 19:29 confidence_intervals.py
-rw-rw-r--  1 tutturuu tutturuu  880 abr 21 19:45 constants.py
-rw-rw-r--  1 tutturuu tutturuu 1068 abr 21 17:28 data_loader.py
-rw-rw-r--  1 tutturuu tutturuu 8443 may  8 15:33 erosion_viz.py
-rw-rw-r--  1 tutturuu tutturuu   66 abr 21 17:28 .gitignore
-rw-rw-r--  1 tutturuu tutturuu  464 abr 21 17:28 __init__.py
-rw-rw-r--  1 tutturuu tutturuu 4272 abr 21 19:47 kinematics_calc.py
-rw-rw-r--  1 tutturuu tutturuu 1873 abr 21 17:28 main.py
drwxrwxr-x  2 tutturuu tutturuu 4096 abr 21 19:30 __pycache__


## Assistant

<thinking>
Now I have a good understanding of the whole picture. Let me summarize what I've learned:

**Current simulation workflow** (on cluster, at `Programas/cluster/scripts/`):

1. User edits `config.toml` (single config file) — sweep params: fills, vels, angles, moves, plus simulation/material settings.
2. User runs `./encolar_sims.sh` → submits ONE Slurm job via `sbatch slurm_submit_JKR.sh`. It copies the config.toml to `configs/config_<jobid>.toml`.
3. On the node (toko04, partition XL, 5 tasks but config says cores=4... inconsistency!), `slurm_submit_JKR.sh`:
   - creates a /dev/shm temp dir, exports MPI tmp vars
   - module purge + load openmpi-4.0.2
   - calls `./run_erosion_bed_sims_lammps_move_np3.sh "${SLURM_JOB_ID}"`
4. `run_erosion_bed_sims_lammps_move_np3.sh <jobid>`:
   - sources python venv
   - finds Python 3.7+
   - runs `read_config.py configs/config_<jobid>.toml <jobid>` → generates `config_vars_<jobid>.sh`
   - sources it, deletes it
   - computes XMOVE/YMOVE from moves × R
   - loops fill → vel → angle → move, creates `erosion_bed_SI_...` simdir under `Buffer/sims_SI_erosion_bed_100um_...`
   - dedup via `mkdir` (if exists, skip — "ya siendo calculada por otro nodo")
   - runs `mpirun -np $CORES ... lmp ... -in collision_erosion_bed.input` with all -v vars → salida.log

**Problems / pain points:**
1. **No batch**: one config = one job. To run many experiments (multiple fills, angles, velocities, NP), the user must manually edit config.toml and re-run encolar_sims.sh for each — many times. This is exactly what the user described: "actualmente las sims se ejecutan mandando multiples veces encolar_sims.sh hasta que el usuario lo desee" (submitting multiple times until the user wants).

2. **Inconsistencies:**
   - `slurm_submit_JKR.sh` has `#SBATCH --ntasks-per-node=5` but config.toml says `cores = 4`, and config_vars.sh (older) said CORES=32 with PROC 4/4/2. And OUTDIR has `_cores_8` hardcoded. So there's a mismatch between sbatch ntasks, cores, processors, and the hardcoded `_cores_8` in OUTDIR.
   - `slurm_submit_JKR.sh` job name is hardcoded `270_0_1_0`.
   - `run_erosion_bed_sims_lammps_move_np3.sh` has `Orientation="90_0_1_0"` hardcoded but the .input has comments showing angles 90 and 270 variants and data files named `box_fill_0.15_SI_100um_270_0_1_0.data` etc. The datas dir has orientation-suffixed data files (90_0_1_0, 270_0_1_0, 180_1_0_0, 270_1_0_0, 90_1_0_0, 100um, 145um). But the runner only uses `box_fill_${fill}_SI_100um_${Orientation}.data` with a hardcoded Orientation.
   - The .input has hardcoded NP=1 projectiles (create_atoms NP1 commented in/out), but config has `np = 3`.
   - `read_config.py` generates `config_vars_<id>.sh` with variable names from TOML keys uppercased. But runner uses `MOVES`, `VELS`, `FILLS`, `ANGLES` etc. The TOML keys: `fills`, `vels`, `angles`, `moves`, `total_steps`, `grain_radius_um`. OK so that maps: FILLS, VELS, ANGLES, MOVES, TOTAL_STEPS, GRAIN_RADIUS_UM. Good. And `xrotate/yrotate/zrotate` map to XROTATE etc. OK.
   - Note: the `np` key in TOML maps to `NP` in shell — used as a variable in OUTDIR, but the actual NP (number of projectile particles) is hardcoded in the .input file. So `NP` config has no effect on the actual simulation! That's a latent bug.

3. **Dedup/resume**: dedup by mkdir existence. If a job crashes mid-way, the simdir exists → will be skipped on re-run (no auto-resume). There was a `resume_controller.py` in legacy that scanned for missing `post-run_final.data` and resumed from restart files. Not integrated.

4. **Config copy/registration**: config is copied to `configs/config_<jobid>.toml` after sbatch submission, keyed by jobid. Fine but scattered.

5. **Naming / organization**: scripts dir contains a mix: input file, config, launcher, submit script, python helper, compiled lammps binary (`lmp_g++_openmpi-4.0.2_stableSep2021`!). The compiled binary lives in the scripts dir — odd but maybe intentional to keep it with the launcher.

6. **Buffer output naming**: OUTDIR hardcodes `cores_8` and the orientation. The sim dir naming: `erosion_bed_SI_ff_${fill}_np_${NP}_vel_${vel}_angle_${angle}_move_${xm}_${ym}_${zm}`. CARBOXI's discover matches prefix `erosion_bed_SI_` and parses ff/np/vel/angle/move. Good consistent naming.

Now the user asks: "propone un plan de orden y unificación" — propose a plan for order and unification. The context: currently sims run by sending encolar_sims.sh multiple times until the user wants to stop.

I should check the current config.toml state in scripts/ (it has fills=[0.15], vels=[-50], angles=[90], np=3) vs config_vars.sh (older, fills=[0.35], vels=[-25], angles=[0], moves, NP=1, CORES=32, EXE...). Note there's also a discrepancy: the scripts/config.toml has np=3, cores=4, processors=[2,2,1], but the slurm script requests ntasks-per-node=5. And OUTDIR name says cores_8. So there are 3 different core counts: sbatch 5, config cores 4, OUTDIR 8.

Wait, also note the config.toml in scripts/ differs from the config_vars.sh (the old generated one from a previous config). So the current config in scripts/config.toml was modified (fills 0.15, vels -50, angles 90, np 3) but config_vars.sh is stale (0.35, -25, 0, np 1, cores 32). This means there's config drift.

Also `run_erosion_bed_sims_lammps_move_np3.sh` sets OUTDIR with `Orientation="90_0_1_0"` but data files use names like `box_fill_0.15_SI_100um_90_0_1_0.data` — note the data files in datas/ have orientations 90_0_1_0, 270_0_1_0, 180_1_0_0, etc. So orientation is somewhat configurable but hardcoded in the script.

Also the `#SBATCH --job-name=270_0_1_0` in slurm_submit_JKR.sh suggests they were running orientation 270_0_1_0 at some point, but now the runner hardcodes 90_0_1_0. Inconsistency.

So the plan should cover:

**Goal**: Unify the simulation workflow so that:
1. Multiple configurations/experiments can be submitted in a single command (a "batch"), each as its own Slurm job (or a single job that processes a queue).
2. Consistent, single source of truth for config (config.toml, maybe extended with an `experiments` list or a batch file).
3. Fix inconsistencies (cores, orientation, job name, NP not actually applied).
4. Track submitted/run/complete sims (manifest) so re-running only submits what's missing (idempotency) — "hasta que el usuario lo desee" becomes "it auto-detects what's missing".
5. Possibly integrate resume of incomplete sims.
6. Better organization of scripts directory.

I should present options and ask for user preference on key design decisions. The user is in plan mode and wants a plan. I should ask clarifying questions about scope.

Let me think about what a well-formed plan looks like.

Design options for batch submission:

**Option A: Multi-job batch (one sbatch per config)**
- Extend config.toml with an array of experiment configs (each defining fills/vels/angles/np/orientation, etc.), or a separate `experiments.toml` / `sweep.json` listing configs.
- `encolar_sims.sh` (or `submit_all.sh`) loops over experiments, for each: writes a temp config, submits sbatch, copies config as config_<jobid>.toml.
- Each job runs the full sweep of that experiment.
- Pros: independent jobs, parallelism at batch level, naturally uses multiple nodes (but nodelist=toko04 restricts to one node!). Cons: same node contention; nodelist hardcoded.

**Option B: Single job, queue inside**
- One sbatch job that processes a queue of configs serially (or a config list), like the resume_controller pattern (python controller iterating configs).
- Pros: fewer submissions, no risk of flooding scheduler; can share preprocessing. Cons: serial within one job unless you spawn parallel sub-processes with the allocated tasks.

**Option C: Hybrid / manifest-based**
- A Python controller (`enqueue_sims.py`) that: reads a batch manifest (list of experiments), computes all simdir names, checks which already exist (or are in a done list), and only submits jobs for missing experiments. Keeps a log/manifest of submitted jobs and their configs. This makes repeated invocation idempotent — you can run it as many times as you want and it only enqueues what's missing. This directly addresses "se ejecutan mandando multiples veces encolar_sims.sh hasta que el usuario lo desee."

Given the current design (dedup via mkdir on simdirs), the natural unification is a manifest-based batch controller. The "hasta que el usuario lo desee" becomes: run submit script once per batch; it computes the full cartesian product, submits N jobs (or one job), and subsequent runs skip already-submitted/existing sims.

Now, about "orden" (order): organize the scripts folder, standardize config naming (config per job id), document the workflow.

Let me also consider: there might be an even simpler approach the user wants — just a unified script that, given config.toml, submits ALL jobs for ALL experiments defined in a list, and can be re-run safely. And a way to know when everything is done (squeue / manifest with status).

Let me look at a couple more things to ground the plan:
- The `Programas/cluster/erosion_49.py` and `erosion.py` are analysis scripts (post-processing of merge results). Not part of the sim submission. Fine.
- `Programas/comparar_filtros/` — filters comparison, separate.
- `Programas/Ideas_de_implementacion/` — let me peek, might contain relevant docs about this exact plan.

Let me check Ideas_de_implementacion for anything about encolar/batch and check the legacy folder fully.
</thinking>

**Tool Call: ls -R /home/tutturuu/Documentos/Universidad/Tesis/Programas/Ideas_de_implementacion/ /home/tutturuu/Documentos/Universidad/Tesis/Programas/legacy/ 2>/dev/null | head -60**
Status: Completed

/home/tutturuu/Documentos/Universidad/Tesis/Programas/Ideas_de_implementacion/:
aggregate_parquets.py
analisis_ocupacion.py
bayesian_sandbox.py
censored.py
crater_analisis.py
crater_volume_plot.png
fabric_tensor.py
fabric_tensor_v2.py
fabric_tensor_v3.py
fabric_tensor_v4.py
fabric_tensor_v5.py
plot_fabric.py
plot_fabric_tensor.png

/home/tutturuu/Documentos/Universidad/Tesis/Programas/legacy/:
analisis_Tesis.ipynb
analyze.py
app.py
CARBOXIMINI
CARBOXI_mini.md
Energy.py
final
heatmap_ff_0.15_v-10.png
heatmap_ff_0.15_v-25.png
heatmap_ff_0.15_v-50.png
heatmap_ff_0.15_v-5.png
heatmap_ff_0.35_v-10.png
heatmap_ff_0.35_v-25.png
heatmap_ff_0.35_v-50.png
heatmap_ff_0.35_v-5.png
__main__.py
pdf_quitadas.py
prueba_distribucion_cluster_por_velocidad.py
prueba_plot_heatmap.py
prueba_profundidad_max_proyectil.py
__pycache__
resultados
resume_collision.input
resume_controller.py
slurm_resume.sh
solid
verificar_normalizacion_Pk_sin_relleno.py

/home/tutturuu/Documentos/Universidad/Tesis/Programas/legacy/CARBOXIMINI:
analysis
config.py
config.toml
__init__.py
io
__main__.py
pipeline.py
plot.py
utils.py

/home/tutturuu/Documentos/Universidad/Tesis/Programas/legacy/CARBOXIMINI/analysis:
__init__.py
topology.py

/home/tutturuu/Documentos/Universidad/Tesis/Programas/legacy/CARBOXIMINI/io:


**Tool Call: grep -rln "encolar\|enqueue\|batch\|lote" /home/tutturuu/Documentos/Universidad/Tesis/Programas/Ideas_de_implementacion/ /home/tutturuu/Documentos/Universidad/Tesis/Documentos/Archive/ 2>/dev/null | head**
Status: Completed

(no output)

**Tool Call: ls -la /home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/logs/ 2>/dev/null; ls /home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/scripts/configs/ 2>/dev/null | tail -20**
Status: Completed

(no output)

## Assistant

<thinking>
No logs or configs dirs exist yet in scripts/ (they'd be created on the cluster). Good.

I have enough context. Let me now formulate a plan. The user's message is truncated: "propone un plan de orden y unificación de" — "propose a plan for order and unification of..." and then showed the encolar_sims.sh script with the description that sims currently run by sending encolar_sims.sh multiple times until user wants.

So the plan should focus on:
1. **Unificar el flujo de envío**: que un solo comando encole todos los experimentos pendientes de una vez (batch), sin necesidad de correr encolar_sims.sh a mano N veces.
2. **Orden**: estructura/roles claros de cada script, consistencia de parámetros (cores, orientation, np, job name), y registro/manifesto para idempotencia.

Let me structure the plan with:
- Diagnóstico actual (flujo + problemas concretos encontrados)
- Diseño propuesto (opciones + recomendación)
- Cambios por archivo
- Preguntas para el usuario

I need to present tradeoffs and ask clarifying questions. Let me be thorough but concise. Key open questions for the user:

1. ¿Un job por experimento (paralelo, cada config = un sbatch) o un solo job con cola interna? Con nodelist=toko04 hardcodeado, la paralelización multi-nodo no aplica de todos modos. Actually with a single node, multiple concurrent jobs on the same node compete for cores. Hmm. Actually toko04 with partition XL — if the node has many cores (earlier config_vars had CORES=32, and slurm_resume had ntasks 32), maybe multiple jobs can coexist on the node (partition XL). But running several jobs each requesting the full node would oversubscribe. Let me note this.

2. ¿La fuente de verdad de los experimentos: extender config.toml con una lista de experimentos, o un manifiesto separado (batch.toml / experiments.toml)?

3. ¿Incluir la resurrección/resume de sims incompletas en este trabajo, o solo el envío en lote?

4. ¿Los correos/output? Not relevant.

5. ¿Mantener el patrón dedup por carpeta (mkdir), o reemplazar por manifiesto con estados?

Let me think about the recommended architecture more concretely.

**Recomendación: controlador en Python + manifestos, submit multi-job.**

Concretely:

```
scripts/
  config.toml            # config base (material, sim defaults, umbrales...)
  experiments.toml       # NUEVO: lista de experimentos (cada uno: fills, vels, angles, moves, np, orientation, nsteps, cores, procesadores, notas)
  enqueue_sims.py        # NUEVO: lee experiments.toml, calcula job id/experimento, chequea manifiesto y Buffer, genera config_<id>.toml, llama sbatch
  slurm_submit_JKR.sh    # modificado: job name/cores desde config, sin hardcodeo
  run_erosion_bed_sims_lammps_move_np3.sh  # modificado: orientación/np/cores desde config
  read_config.py         # sin cambios (o renombrado a config_sh.py)
  collision_erosion_bed.input  # plantilla; NP y orientación parametrizados
  configs/               # generado: config_<jobid>.toml
  jobs/ o queue/         # generado: manifiesto de jobs (jobid, experimento, estado, fecha)
```

Actually wait — let me reconsider. The sims compute output in `Buffer/sims_SI_erosion_bed_100um_...` which is relative to CWD where the job runs (the scripts/ dir on the cluster). The `configs/` dir stores per-job configs. And there's a `datas/` dir at `Programas/cluster/datas/`. But run script references `"${BASE_DIR}/datas/$DATA"` and `"${BASE_DIR}/$EXE"` where BASE_DIR=$(pwd). But on the cluster, do datas and the binary live in scripts/? The binary `lmp_g++_openmpi-4.0.2_stableSep2021` IS in scripts/. datas is at `Programas/cluster/datas/`, NOT in scripts/. So there's a path assumption that the working directory on the cluster contains datas/ and Buffer/ and logs/. Actually the cluster may have its own layout mirroring this (the repo is synced to the cluster machine). The exact cluster layout is unknown to me. I should note this assumption and ask.

Actually, given slurm_submit references `./logs/job_exec_%j.out`, and run script uses `${BASE_DIR}/datas/`, the cluster working dir must contain `logs/`, `datas/`, `Buffer/`, and the `scripts/` (with configs). It seems the whole `Programas/cluster/` tree is synced to the cluster, and jobs are launched from `Programas/cluster/scripts/`. OK.

Let me also reconsider the "un solo job procesa todo" option more carefully given the mkdir-dedup design: currently each submission runs the full cartesian sweep of the config's fills×vels×angles×moves, skipping existing dirs. So even the current flow is "resumable-ish" at the simdir level across jobs. The pain is only the manual config editing + repeated submission for each new experiment.

So the MINIMAL unification that addresses the user's pain:
- A batch runner that takes a list of experiments and submits one job per experiment automatically (all pending at once).
- Idempotent: keeps a manifest (submitted jobs → experiment → config hash), so re-running doesn't resubmit completed/queued experiments.

That's clean, KISS, and matches existing design.

Now the "orden" part (ordering): 
- Separate concerns: `config.toml` (shared defaults) vs `experiments.toml` (what to run) vs generated `configs/` (per-job frozen config) vs generated `queue/` manifest.
- Name the scripts consistently.
- Remove hardcoded values: job-name, ntasks, orientation, cores_8 in OUTDIR, NP in .input (parametrize create_atoms via -v or a small rewrite).

Let me now check: is NP actually parametrizable? The .input has hardcoded create_atoms for NP1 (commented NP3). To parametrize NP, need conditional create_atoms in LAMMPS input. LAMMPS supports `if "${np} == 1" then "create_atoms ..." elif ...`. That's doable but adds complexity. Alternatively use a pre-processing step that writes the .input per job (like read_config.py generates shell vars). Actually a cleaner approach: since config per job is already materialized as config_<jobid>.toml, the runner could generate a per-job .input file (sed/templating) that includes the right create_atoms block. But LAMMPS `if` with create_atoms is actually fine in input script. Let me note it as an option: parametrize NP via LAMMPS if/elif in the input file.

Also the orientation: data files are prebuilt per orientation (box_fill_0.15_SI_100um_90_0_1_0.data etc.). So orientation must be a config key that selects the data file AND the OUTDIR name. Currently hardcoded "90_0_1_0". This should become an experiment parameter.

Let me also note the `cores`/`ntasks` mismatch and `_cores_8` hardcode as things to unify into the config (one place: config.toml [simulation].cores + processors, and slurm submits `--ntasks-per-node` and `-np` from the same value).

Now, let me draft the plan. Since I'm in plan mode, I present the plan and ask clarifying questions. Let me keep it structured and in Spanish (user speaks Spanish).

Let me organize:

## Diagnóstico
Flujo actual + lista de problemas concretos (con referencias a archivos/líneas).

## Objetivos
1. Un solo comando encole todos los experimentos pendientes.
2. Idempotencia (re-correr no duplica).
3. Coherencia de parámetros (cores, np, orientation, job name).
4. Trazabilidad (manifiesto de jobs).

## Diseño propuesto
- experiments.toml (lista de experimentos)
- enqueue_sims.py (controlador)
- Modificaciones a slurm_submit_JKR.sh, run_erosion..., collision_erosion_bed.input
- Manifiesto queue/

## Preguntas para decidir
1. ¿Job por experimento vs un solo job con cola interna?
2. ¿experiments.toml como fuente, o extender config.toml con [[experiments]]?
3. ¿Incluir auto-resume de sims incompletas?
4. ¿Parametrizar NP por input de LAMMPS o por generación del .input?
5. ¿Alcance: solo envío, o también el post-proceso (CARBOXI)?

Let me also present the two submission models with tradeoffs since that's the central decision.

Actually, let me reconsider whether one-job-per-experiment is right given nodelist=toko04. If all jobs pin to toko04 and each requests, say, 4-8 cores with ntasks-per-node=5, then multiple jobs can run concurrently if the node is big (64 cores?). But oversubscription risk. The current design already relies on mkdir-dedup to let different jobs on the same node share the sweep — meaning they intentionally run concurrent jobs on one node, each taking a slice. So one-job-per-experiment fits the existing concurrency model.

I should present this and let the user decide. Let me write the plan concisely but completely. I'll present it in Spanish.

I'll include a concrete file/script proposal. Let me write it now.

One more thing: let me verify there is no AGENTS.md / docs that prescribe the batch workflow. There's the solid/ folder in legacy, deleted AGENTS.md. Not relevant to submission.

Also, I should mention where each job writes config. Currently `configs/config_<jobid>.toml`. Keep that, plus add a `queue/` or `jobs/` manifest dir with a row per experiment: id, experiment label, config hash, jobid, submit time, state (submitted/completed), and a done marker when post-run_final.data appears for all sims of that experiment. The controller on re-run reads the manifest + Buffer to decide what to submit.

Idempotency detection without scheduler access: the runner checks, for each experiment, whether all its simdirs exist under Buffer (i.e., already processed). If the batch was previously submitted, re-running only submits experiments whose sims are not all complete. That's pure filesystem-based, matches the mkdir-dedup philosophy, and works offline. Plus optional `squeue` check for queued/running jobs (the controller could also parse `squeue -u $USER` and skip experiments with a running jobid). Nice.

Let me draft the final plan. I'll aim for something actionable and ask 4-5 clarifying questions at the end. Since plan mode expects me to possibly ask clarifying questions, I'll present the plan then the questions.

I should be reasonably concise. Let me write it.
</thinking>

He relevado todo el flujo. Te dejo el diagnóstico y el plan propuesto. Antes de finalizar, hay 2-3 decisiones de diseño que quiero confirmar con vos.

## Diagnóstico del flujo actual

**Cadena de ejecución actual** (cada corrida manual):

```
editar config.toml
  → ./encolar_sims.sh            (sbatch slurm_submit_JKR.sh; copia config.toml → configs/config_<jobid>.toml)
    → slurm_submit_JKR.sh        (job → toko04; tmp en /dev/shm; module openmpi)
      → run_erosion_bed_sims_lammps_move_np3.sh <jobid>
          (read_config.py → config_vars_<jobid>.sh → source)
          (sweep fill × vel × angle × move; mkdir-dedup; mpirun lammps)
```

**Problemas encontrados:**

1. **Un config = un job.** Cada `encolar_sims.sh` encola UNA corrida completa (todo el producto cartesiano de su `config.toml`). Para probar varias configuraciones hay que editar el TOML y re-enviar N veces a mano ("hasta que el usuario lo desee"). No hay noción de "lote" ni de "pendientes".
2. **Números de núcleos inconsistentes**: `slurm_submit_JKR.sh:4` pide `--ntasks-per-node=5`, `config.toml:16` dice `cores = 4`, `run_erosion_bed...sh:54` hardcodea `_cores_8` en el OUTDIR. Tres fuentes distintas.
3. **Valores hardcodeados**: job-name `270_0_1_0` (`slurm_submit_JKR.sh:2`), `Orientation="90_0_1_0"` (`run_erosion...:53`), mientras los `.data` en `datas/` vienen para varias orientaciones.
4. **`np` del TOML no se usa**: `config.toml:13` dice `np=3` pero el proyectil se crea en `collision_erosion_bed.input:57` (bloque NP1 activo, NP3 comentado). Cambiar `np` no cambia la simulación.
5. **El dedup es por existencia de carpeta** (`run_erosion...:79`): si un job muere a mitad de un simdir, al re-correr se saltea (no hay auto-resume). El `resume_controller.py` de `legacy/` resuelve esto pero no está integrado.
6. **Nada registra qué se envío**: no hay manifiesto de jobs → no se puede saber qué experimentos quedaron pendientes ni cuáles terminaron.

## Plan propuesto

**Idea central**: un controlador (`enqueue_sims.py`) que lea una lista de experimentos y encole automáticamente **todos los pendientes** en un solo comando, de forma idempotente (re-correr = no duplica).

**Arquitectura de archivos** (en `Programas/cluster/scripts/`):

```
config.toml          # config BASE (material, sim, umbrales) — sin cambios estructurales
experiments.toml     # NUEVO: [[experiment]] — qué correr (fills/vels/angles/moves/np/orientation/nsteps/cores)
enqueue_sims.py      # NUEVO: controlador de envío (reemplaza el uso manual repetido de encolar_sims.sh)
slurm_submit_JKR.sh  # MOD: job-name, ntasks y partición derivados de la config del experimento
run_erosion_bed_sims_lammps_move_np3.sh  # MOD: orientation, cores, np desde config; sin hardcodeos
collision_erosion_bed.input              # MOD: proyectil parametrizado por np
configs/             # generado: config_<jobid>.toml (ya existe)
queue/               # NUEVO: manifiesto jobs.tsv (experimento, hash, jobid, fecha, estado)
```

**Comportamiento de `enqueue_sims.py`:**
1. Lee `experiments.toml` (lista de experimentos).
2. Para cada experimento calcula qué simdirs produciría (`erosion_bed_SI_...`) y chequea contra `Buffer/` + `queue/` + `squeue -u $USER`.
3. Encola **solo lo que falta** (un `sbatch` por experimento pendiente) y registra el job en `queue/jobs.tsv` con hash de su config.
4. Se puede correr las veces que se quiera: es un no-op cuando no queda nada pendiente (esto reemplaza el "mandar hasta que quiera").

**Consistencia a resolver:**
- Una sola fuente de núcleos: `cores`/`processors` del experimento → `--ntasks-per-node` + `-np` + OUTDIR.
- `orientation` como parámetro de experimento (elige el `.data` y el OUTDIR).
- `np` parametrizado en el `.input` con `if`/`elif` de LAMMPS (sin generar archivos por caso).

## Decisiones que necesito confirmar

1. **Modelo de envío**: ¿un `sbatch` por experimento (paralelo, como hoy pero automático) — o un solo job que procesa la cola internamente en serie? Con `nodelist=toko04` fijo, lo recomendable es job-por-experimento.
2. **Fuente de verdad de los experimentos**: ¿`experiments.toml` separado, o `[[experiment]]` dentro del `config.toml` existente?
3. **Alcance**: ¿incluimos el auto-resume de sims incompletas (integrar el `resume_controller.py` de legacy) o lo dejamos para otra iteración?
4. **`np` parametrizado**: ¿ok con `if/elif` de LAMMPS en el `.input`, o preferís que el runner genere un `.input` por job?

¿Cómo querés responder?

## User

1- Lo que pasa es que por usuaro por normativa solo se puede usar hasta 2 nodos, que en mi caso solo uso toko0x XXL los 64 nucleos que depende de la partición que se elija para la simulación, sin embargo ocurren 3 cosas:
- No siempre estan libres los nodos y por ahi hay libre 32 núcleos compartiendo con otro usuario
- No siempre se terminan todas las sims, a veces por ejemplo se envian 32 tareas de 4 nucleos en 2 nodos, sin embargo estas pueden llegar a alcanzar el tiempo máximo de 3 días y se cortan subditamente las tareas, luego las que estaban encoladas no continuan la simulacion que estaba parada sino que la saltean por lo que hay que borrar las que no se trminaron y mandar una segunda pasada.
- La distribución de núcleos cambia con ff pues para 0.15 se adopta un 2x2x1 y para 0.35 un 2x4x1
Los otros puntos los discutimos luego

## Assistant

<thinking>
The user clarified important constraints:

1. **2 nodes max per user** by norm. He uses toko0x XXL with 64 cores each. Depends on partition.
   - Nodes aren't always free — sometimes only 32 cores free sharing with another user.
   - Sims don't always finish: e.g. 32 tasks of 4 cores in 2 nodes can hit the 3-day max time and get cut suddenly. Queued tasks then skip the interrupted sim (because the simdir exists via mkdir-dedup) → must delete unfinished ones and do a second pass.
   - Core distribution changes with ff: 0.15 → 2x2x1 (4 cores), 0.35 → 2x4x1 (8 cores).

So the real pain points that must drive the design:

- **Dynamic core allocation**: the number of available cores varies (partition-dependent, shared nodes). So jobs must be able to adapt to how many cores are free.
- **Time limit kills sims**: max 3 days; sims get killed. Because of mkdir-dedup, queued/next sims skip the interrupted ones → need a second pass after deleting unfinished simdirs. So we need a proper resume mechanism (auto-resume from restart files) AND a way to detect incomplete sims (no post-run_final.data) and resume them rather than skip.
- **ff determines processor grid**: ff=0.15 → 2x2x1; ff=0.35 → 2x4x1. This must be derived per-experiment, not hardcoded.

So the design must incorporate:
1. **Resume-integrated runner**: instead of mkdir-dedup-skip, if a simdir exists but lacks `post-run_final.data`, resume from the latest restart file (res_A.rst/res_B.rst) — like the legacy resume_controller.py but integrated into the normal run loop.
2. **Adaptive cores**: the enqueue controller should know how many cores are free per node (squeue/availability) and scale the number of concurrent jobs accordingly; each job should use a fixed number of cores per sim (4 or 8) but the grid `processors` derived from ff.
3. **Completion-aware manifest**: track which experiments are done (all simdirs have post-run_final.data). Re-running enqueue submits only what's missing.
4. **ff → processors mapping** in config (a dict): e.g. `[processors_by_ff] 0.15 = [2,2,1], 0.35 = [2,4,1]`.

The core question about model of submission: given that node availability is variable and jobs can die at 3 days, a job-per-experiment (or job-per-(experiment-slice)) approach is needed, but more importantly each job should be **self-healing**: within its time window it runs sims; if it dies, the next pass resumes.

Actually, the current architecture already handles "resume" partially via dedup: a second manual pass (after deleting incomplete dirs) re-runs them. The user wants this automated: the runner should NOT skip incomplete sims; it should either resume them (from restart) or re-run them fresh (delete + rerun). 

Given the 3-day wall limit, a good pattern: one Slurm job per experiment; within the job, the runner processes sims sequentially (or a few in parallel given cores). If the job times out, the incomplete simdirs remain; on the next enqueue pass, the controller detects them and resubmits. The runner resumes from restart files if available, otherwise restarts.

Actually let me reconsider: 32 tasks of 4 cores in 2 nodes = 8 sims running concurrently per... wait 32 tasks × 4 cores = 128 cores across 2 nodes = 64 cores/node = XXL. So each experiment job with 32 sims, each sim 4 cores, running 8 concurrently. The 3-day limit cuts the whole job, killing currently-running sims mid-flight; completed simdirs remain, incomplete ones are skipped by the next job due to mkdir-dedup.

So the fix: in the runner loop, before skipping an existing simdir, check for `post-run_final.data`. If missing → resume or re-run. This directly solves pain #2 without needing to delete manually.

Also important: with mkdir-dedup across concurrent jobs, two jobs might both try to run the same simdir → mkdir-dedup prevents that. If we switch to "resume incomplete", we must keep the mkdir-claim mechanism but make the check before mkdir smarter: if dir exists and has post-run_final.data → skip; if dir exists and incomplete → attempt to resume (only if not already claimed by another job?). Concurrency with resume is tricky (two jobs could resume the same sim). Options: keep it simple — if dir exists and incomplete, and no other job is currently running (check via a lock file or `flock` on the dir, or check file locks). Or simpler: the enqueue controller handles resume in a dedicated serial pass, while fresh submission only creates new dirs. Actually the cleanest: keep mkdir-claim for fresh sims. For incomplete sims, add a per-sim lock (e.g. `mkdir .incomplete.lock` or flock) so only one job resumes it.

I think for the plan, the important design elements are:

**A. Controlador de envío (`enqueue_sims.py`)**
- Read experiments.toml.
- For each experiment, determine processor grid from ff (mapping in config).
- Compute all simdirs; classify: pending (not created), complete (has post-run_final.data), incomplete (dir exists, no final file).
- Enqueue missing work: one job per experiment with pending+incomplete sims, OR a smart allocation given free cores.

**B. Runner con auto-resume**
- Replace `mkdir || skip` logic with: if simdir exists + post-run_final.data → skip; if exists + incomplete → resume from newest res_A/res_B restart; if not exists → run fresh.
- Resume input: reuse resume_collision.input from legacy or make main input handle restart via `if` variable.
- Keep locking to avoid double-running across concurrent jobs.

**C. Consistencia**
- cores/processors from config, processors grid derived from ff.
- orientation as experiment param.
- np parametrized.
- Remove hardcoded job name, cores_8.

**D. Manifiesto**
- queue/jobs.tsv records jobid ↔ experiment ↔ state; state updated when all sims of the experiment are complete.

Now, the "adaptive cores / node availability" part: the user says sometimes only 32 cores free on a node. The controller should query available cores and pick a number of tasks that fits. But this is complex (squeue parsing, partition specifics). I should propose: `enqueue_sims.py` accepts `--max-nodes`/`--min-free-cores` and checks `squeue`/`scontrol` to determine available free cores, then computes how many jobs to submit so as not to oversubscribe. Or, simpler: let the user pass target total cores and the controller splits experiments into jobs that fit. Actually simplest robust approach: one job per experiment, each job requests cores = experiment's `tasks` (e.g. 8×sims parallel) and `--ntasks-per-node` accordingly; Slurm will queue the job until resources free. Slurm's scheduler already handles waiting. The oversubscription issue: two of the user's jobs could both get scheduled on the same node, oversubscribing. Using `scontrol` reservations or just monitoring `squeue` and limiting concurrent jobs is more work.

I should present this as a design decision and ask. But the user said "Los otros puntos los discutimos luego" — they only want to address point 1 (the submission model / constraints) now, and discuss the other points (source of truth, scope of resume, np) later.

So the user's response focuses on **decision #1: submission model**, giving context that changes the design. They want the plan to reflect these realities:
- 2 nodes max, 64 cores each (XXL), shared → variable free cores
- 3-day time limit → sims die → must not skip incomplete sims
- ff-dependent processor grid

So now I should update the plan to incorporate these constraints, and confirm the resulting design choices. The key question is: given this, how should enqueue + runner work?

Let me think about the best design given these constraints:

**Wall-clock reality**: A job has max 3 days. An experiment (e.g. ff sweep with 7 moves) with, say, 32 sims at ~X hours each won't fit in 3 days if run serially; so the current approach runs tasks in parallel within the job (8 sims × 4 cores = 32 cores on a node). Even so, some sims may not finish.

**Two-level parallelism**:
- Level 1: sims within a job run concurrently (mpirun per sim, split cores).
- Level 2: multiple jobs per experiment? Since an experiment might exceed 3 days total, you can't put the whole experiment in one 3-day job. So the controller must **chunk** experiments: submit multiple jobs for the same experiment, each job takes a subset of sims (a chunk) that fits in 3 days and fits in available cores. Each chunk's sims are assigned statically (split the list), and each job processes its chunk, skipping complete and resuming incomplete.

Hmm, this is getting complex. Let me simplify:

The `run_erosion...sh` currently iterates the whole cartesian product and dedups by mkdir. The natural enhancement: the runner, for its assigned chunk, should:
1. Try to claim a simdir (mkdir or lock).
2. If already complete → skip.
3. If incomplete → resume.
4. If fresh → run.

And the controller decides chunking: given an experiment with N sims, task size T (sims per job), cores per sim C (grid from ff), free cores F: submit ceil(N/T) jobs... but each job needs a time budget. Time budget depends on nsteps and sim duration. The user knows how many sims fit in 3 days with a given concurrency.

Actually, maybe simpler and closer to current behavior: The user already knows "32 tareas de 4 núcleos en 2 nodos" — i.e., they submit jobs with a number of tasks (concurrent sims). The problem is just: (a) skipped incomplete sims, (b) manual repetition.

So perhaps the minimal, robust design is:

**`enqueue_sims.py`** takes a `--tasks` and `--nodes` budget (e.g. 32 tasks × 4 cores on 2 nodes), and for each experiment:
- Splits the sim list into chunks of size `tasks` (or a time-based estimate).
- Submits a job per chunk with `--ntasks-per-node` = tasks_per_node and `-np` per sim = C.
- Each job runs its chunk's sims: fresh if new, resume if incomplete, skip if complete.
- Registers in manifest.

**On re-run**: it recomputes what's incomplete/complete and only submits the missing chunks. This is idempotent and self-healing. No manual deletion.

This directly solves pain #2 (the second pass becomes automatic).

For pain about core distribution changing per ff: the runner/controller computes processors grid from ff: `ff=0.15 → [2,2,1]=4 cores`, `ff=0.35 → [2,4,1]=8 cores`. The chunking uses cores-per-sim from ff. Also note: current config.toml has cores=4 and processors=[2,2,1], and config_vars had 32 cores. The user now says ff determines grid. So `processors` becomes per-ff mapping.

Hmm wait — actually with the chunking approach, since sims within a chunk run concurrently and each uses C cores, tasks per node = free cores / C. For ff=0.15 (C=4): 16 sims/node on 64 cores. For ff=0.35 (C=8): 8 sims/node.

I need to confirm with the user:
1. Does a single experiment's sims (e.g., one ff with 7 moves) always fit within the jobs they send, and the issue is only cross-experiment? Or do they also need chunking within an experiment (i.e., one experiment may take more than 3 days)?
2. Time per sim: how many nsteps and roughly how long? (to size chunks) — or do they prefer to keep specifying "tasks" manually and just get resume + idempotency?

Actually, let me re-read the user's message: "a veces por ejemplo se envian 32 tareas de 4 nucleos en 2 nodos, sin embargo estas pueden llegar a alcanzar el tiempo máximo de 3 días y se cortan subditamente las tareas, luego las que estaban encoladas no continuan la simulacion que estaba parada sino que la saltean por lo que hay que borrar las que no se trminaron y mandar una segunda pasada."

"las que estaban encoladas no continuan la simulacion que estaba parada sino que la saltean" — the queued (subsequent) tasks don't continue the interrupted sim but skip it. Because mkdir-dedup. "hay que borrar las que no se terminaron y mandar una segunda pasada" — must delete unfinished and send a second pass.

So yes: the core fix is the runner must detect incomplete (no post-run_final.data) and resume them (from restart) instead of skipping. And the controller should automate the "second pass."

Given the complexity, I'll propose the design with these components and ask targeted questions. Since the user said "Los otros puntos los discutimos luego", I should keep the plan focused on the submission/execution model (decision #1) and the adaptations derived from their constraints (resume, ff-grid, adaptive cores), and not push too hard on experiments.toml vs config.toml (decision #2), resume scope (now clearly IN scope — it's the core of the fix), and np parametrization (leave for later).

Let me now write a focused response:
1. Reflect the constraints back (confirm understanding).
2. Present the revised execution model design.
3. Ask 2-3 specific questions that block the design.

Key questions:
- Q1: Chunking model. Is each experiment's full sim list meant to be split across multiple jobs (because 3-day limit), or does one experiment normally fit in one job and the "second pass" is just re-running the same job later? This determines whether the controller needs time-based chunking or simple task-based chunking.
- Q2: Resume method preference: auto-resume from restart files (res_A/res_B) inside the normal runner, or fresh re-run of the interrupted sim? (Restart preserves progress; fresh is simpler but wastes time.) I lean auto-resume.
- Q3: Concurrency/locking: with resume added, how to prevent two concurrent jobs from claiming the same incomplete sim? (flock / lockfile per sim, or rely on the manifest assigning each sim to exactly one job.)

Actually Q3 can be solved by static assignment via manifest: controller assigns sims to jobs at submit time (recorded in queue/jobs.tsv). Then each sim belongs to exactly one job → no double-run. That's cleaner than flock. But static assignment means if a job dies and you resubmit "its" sims, the manifest must allow reassignment of incomplete sims. Controller: on re-run, for incomplete sims, reassign to a new job (if no running job claims them). Track running vs dead jobs — dead jobs detected by absence in squeue. This is getting sophisticated but is the right design.

Simpler alternative: dynamic claim via `mkdir` for fresh + `flock` for resume, no static assignment; any job can pick up any pending/incomplete sim. This is closer to current design (mkdir-dedup) and very robust to job death. I think dynamic claim is actually the best fit here: it's self-healing, no manifest-assignment bookkeeping, works across arbitrary job death. The only risk is two concurrent jobs resuming the same incomplete sim simultaneously; `flock` on a lockfile inside the simdir solves it cheaply.

Let me propose: dynamic claim (mkdir for new; flock for resume). Controller just ensures enough jobs are running for each experiment and resubmits when chunks are incomplete.

Given the 3-day limit and variable free cores, I'll propose the controller computes chunks from a user-provided budget (`--tasks N --cores-per-sim auto-from-ff --time`) OR keep it simpler: `--chunks-per-experiment K` (split each experiment's sims into K jobs), plus `--max-concurrent-jobs`. Hmm.

Let me step back and keep the plan digestible. Present the revised model at a high level, ask the blocking questions, and defer the rest.

Structure of my reply:
1. Confirm understanding of constraints (bullets).
2. Revised model: 
   - Runner: claim → complete? skip : (incomplete? resume-from-restart : fresh run). flock to avoid double-claim. This kills the "second pass + delete" pain.
   - Processor grid per ff (config map ff→[px,py,pz]) driving -np, ntasks, OUTDIR cores.
   - Controller: reads experiments, splits into chunks sized to fit time/cores, submits, registers manifest, idempotent re-run.
   - Adaptive cores: submit respecting free cores (squeue/scontrol), or user passes budget.
3. Questions.

Let me write it. Keep concise. Spanish.

Actually — one more consideration on "solo se puede usar hasta 2 nodos": so total budget = 2 nodes × 64 = 128 cores (if XXL free), or less if shared. Jobs must not oversubscribe beyond free cores. The controller could query `scontrol show node` to get free CPUs, or simpler: user passes `--tasks-per-node` and the controller ensures total requested ≤ available. I'll ask how they want to specify the budget.

Let me finalize.
</thinking>

Entendido, esto cambia el diseño de forma importante. Confirmo cómo veo las restricciones:

- **Límite de 2 nodos** (XXL, 64 cores c/u), pero los cores **no siempre están libres** (nodos compartidos) → la capacidad disponible varía por corrida.
- **Tope de 3 días de wall-time**: si el job se corta, los simdirs a medio correr quedan y la siguiente pasada los **saltea** por el dedup `mkdir` (run_erosion...sh:79) → hay que borrarlos a mano y re-enviar. Ese es el dolor central.
- **El grid de procesadores depende de ff**: 0.15 → `2x2x1` (4 cores), 0.35 → `2x4x1` (8 cores). Hoy está desacoplado del `-np` y hasta hay un `_cores_8` hardcodeado en el OUTDIR.

Esto redefine la solución. El diseño pasa a tener 3 pilares:

## Modelo revisado

**1. Runner con auto-resume (elimina el "segunda pasada manual")**

En vez de `mkdir || saltear`, la lógica de claim por simdir pasa a ser:

```
si post-run_final.data existe        → completo: skip
si dir existe y hay res_A/res_B.rst  → incompleto: retomar desde el restart más reciente
si no existe                          → fresco: correr de cero
```

Con **`flock` sobre un lockfile dentro del simdir** para que dos jobs concurrentes no retomen el mismo sim incompleto a la vez (hoy lo evita el `mkdir`, pero el resume lo necesita). El `resume_collision.input` + lógica de `resume_controller.py` de `legacy/` se integran acá, no como script aparte.

**2. Grid de procesadores derivado de ff**

Una tabla en config (`ff → [px,py,pz] → cores`) que alimenta a la vez `-np`, `--ntasks-per-node`, `processors` del `.input` y el sufijo del OUTDIR. Un solo lugar de verdad, adiós al `_cores_8`.

**3. Controlador que encole "lo que falta", no "todo otra vez"**

`enqueue_sims.py` recibe el presupuesto de recursos (p. ej. `--nodes 2 --tasks 32` o cores libres detectados de `scontrol/squeue`), divide cada experimento en **chunks** que entren en el tope de 3 días, y solo envía jobs para sims pendientes/incompletos. Re-correr es idempotente y autocurativo: los sims que el job anterior dejó a medias los retoma el siguiente.

## Preguntas que bloquean el diseño

1. **Chunking**: ¿un experimento (p. ej. un ff con 7 moves) entra holgado en un job de 3 días, o suele necesitar varios jobs? Necesito saber si el split debe ser por tiempo estimado o solo "re-enviar lo que quedó" (que es más simple).
2. **Resume por restart vs re-correr fresco**: ¿preferís retomar desde `res_A/res_B.rst` (conserva progreso, más complejo) o está bien re-ejecutar el sim incompleto desde cero? Con 2M de steps, asumo que el restart vale la pena.
3. **Presupuesto de cores**: ¿cómo querés especificar la capacidad disponible al correr el controlador — manual (`--tasks 32`) o auto-detectada de `scontrol`? El auto-detecta complica la lógica y depende de cómo reporta free-cores el cluster; manual es KISS.
4. ¿Los sims de un mismo job corren todos en paralelo (varias tareas MPI de 4/8 cores) como hoy, correcto? Eso define el `--ntasks-per-node` = cores_libres / cores_por_sim.

Los otros puntos (experiments.toml vs config.toml, np parametrizado) quedan para después como dijiste.

## User

y 0.15 por move (que pueden ser de 9 a 25) tarda entre 3 y 8 hs y 0.35 tarda el doble

## Assistant

<thinking>
Let me process this new information:

- ff=0.15: each move takes 3-8 hours (moves can be 9 to 25 per experiment)
- ff=0.35: takes double (6-16 hours)

So for ff=0.15 with 9 moves at ~5h average = 45h, fits in 3 days (72h). With 25 moves at 8h = 200h → needs multiple jobs/chunks.

For ff=0.35: 9 moves × ~10-16h avg = 90-144h → doesn't fit in 72h. Definitely needs chunks.

So the chunking MUST be time-based / job-based, not just "resend what's left." Because an experiment can need multiple jobs within 3-day windows, running in parallel chunks.

Let me think about the numbers concretely to design the chunking.

Per sim (one move):
- ff=0.15: 3-8 h, say ~5-6h avg. Cores per sim = 4 (2x2x1).
- ff=0.35: 6-16 h, say ~10-12h avg. Cores per sim = 8 (2x4x1).

Node capacity: 64 cores/node, up to 2 nodes = 128 cores. But free cores vary.

If fully free (64 cores/node, 2 nodes = 128 cores):
- ff=0.15 (4 cores/sim): 16 sims/node = 32 sims concurrently. If each sim ~5h, in 72h you can run 32 × (72/5) ≈ 32 × 14 = 448 sim-hours... wait let me compute: with 32 concurrent slots, each slot can run 72/5 ≈ 14 sims sequentially → 32×14 = 448 sims per job-window? That seems too many. Hmm, but moves per experiment = 9-25. So actually the issue is different.

Wait, let me reconsider. The user said "a veces por ejemplo se envian 32 tareas de 4 nucleos en 2 nodos, sin embargo estas pueden llegar a alcanzar el tiempo máximo de 3 días y se cortan". So they send 32 tasks (each a sim, 4 cores) on 2 nodes. Each task = one sim running with mpirun -np 4. 32 sims run concurrently. If each sim takes 5h, all 32 finish in 5h. So why would it hit 3 days?

Unless... within a single Slurm job, the runner (run_erosion...) runs the sims in a loop, and the mpirun calls run sequentially? Let me re-check the runner. 

Looking at run_erosion_bed_sims_lammps_move_np3.sh: it loops fill → vel → angle → move, and calls `$MPIRUN -np $CORES ...` sequentially (one after another). So within ONE job, sims run SEQUENTIALLY, one at a time, each using CORES=4 (or 8). So a single job with 25 moves at 8h each = 200h >> 72h → job dies at 3 days.

But the user said "32 tareas de 4 nucleos en 2 nodos" — meaning they submit 32 JOBS? No wait — "se envian 32 tareas de 4 nucleos en 2 nodos" — they send 32 tasks of 4 cores on 2 nodes. Hmm, could mean one job with 32 concurrent tasks? But the runner is sequential per job. 

Actually re-reading: "se envian 32 tareas de 4 nucleos en 2 nodos, sin embargo estas pueden llegar a alcanzar el tiempo máximo de 3 días y se cortan subditamente las tareas". I think this means: they submit jobs such that the total work = 32 sims × 4 cores, distributed over 2 nodes. Whether that's 32 separate sbatch submissions (one per sim) or one job per move... The mkdir-dedup suggests multiple jobs can run concurrently and skip each other's sims. 

Actually the current architecture: encolar_sims.sh submits ONE job per config; the runner processes the whole cartesian product sequentially. If a config has, say, fill=0.15 with 25 moves = 25 sims × 4h = 100h > 72h → dies. So the "32 tareas" might be 32 sbatch submissions across several configs, OR the user submits multiple times to get parallel coverage.

Hmm, but wait — with the mkdir-dedup, if you submit the SAME config twice, both jobs run the same sims list, and each claims sims via mkdir, effectively load-balancing. So the user might submit N copies of encolar_sims.sh with the same config to parallelize across the sweep! That's a clever use of the dedup. So "32 tareas" = 32 job submissions (copies) → they race to claim sims; total = 32 concurrent sims × 4 cores = 128 cores = 2 nodes.

OK so the real architecture question: the current design parallelizes by submitting multiple copies of the same job; each copy claims sims via mkdir-dedup; incomplete sims (dir exists) get skipped by later copies → second pass needed after manual deletion.

Given time per sim (3-8h ff=0.15, double for 0.35) and 72h limit:
- A sim can take up to 16h (ff=0.35 max). Fine within 72h for one sim.
- The issue is when a single copy (sequential loop) tries to run too many sims in its 72h window. So the controller must **split the sim list into chunks that fit in 72h**, OR the controller submits "worker" jobs that each claim a bounded number of sims.

Better design given constraints: **worker-pool model**:
- Controller computes the full list of sims (all experiments' cartesian products), each sim with its estimated time (by ff) and cores (by ff).
- Given a resource budget (nodes/cores free), controller launches `W` worker jobs, each with a task limit. But chunks still need to fit in 72h.

Alternative cleaner model: **controller splits each experiment into chunks of K sims** where K = floor(72h / est_time_per_sim) considering sequential execution per job, and submits one job per chunk. Plus multiple concurrent chunks per experiment.

But if we instead make each job capable of running sims in PARALLEL (multiple mpirun at once), then a single job with --ntasks-per-node=64 (or free cores) could run 16 sims concurrently (4 cores each), and in 72h process 16 × 14 ≈ 224 sims. That's overkill per experiment but fine. Actually parallel-within-job is better: fewer jobs, better utilization, simpler manifest.

Hmm, but parallel mpirun within one job: LAMMPS jobs each spawn their own MPI_COMM_WORLD; multiple mpiruns in the same job is fine (each gets its own process group). Using OpenMPI with --allow-run-as-root. The runner would use GNU parallel or a simple xargs -P to run N sims concurrently. That's a bigger change to the runner.

Let me reconsider what to propose. The user's time info mainly informs: **time-based chunking is required** (a single 72h job can't always fit an experiment). So my Q1 answer: chunking must split the experiment across multiple jobs based on time budget.

Given that, and the "second pass" pain, the design:

**enqueue_sims.py**:
1. Input: experiments list + budget (`--nodes 2 --cores-free-per-node 64` or `--tasks-per-node 16` for ff=0.15), and optionally `--max-duration 72h`.
2. Compute all sims per experiment with est. time (config: per-ff est hours) and cores (per-ff grid).
3. For each experiment, greedily pack sims into "work packages" such that each package ≤ max-duration (sequential sims in a job) — but allow parallelism within the job to use free cores: actually package = (sims) to be processed by one job, running P-at-a-time where P = free_cores/cores_per_sim.

Hmm, let me keep it simpler for the plan and present it at the right level:

The runner becomes **parallel within a job**:
- `run_erosion...sh` takes a chunk list (file of sims), runs them with `xargs -P $CONC` (or GNU parallel), where each worker: claims sim (mkdir/flock), completes, resumes, or skips.
- Chunk size chosen so that chunk_time = ceil(sims/P) × est_time ≤ 72h.

Actually simpler: don't chunk by hand. The controller submits jobs with a **time budget**: each job declares how many sims it will process; runner processes them in parallel P-at-a-time. Controller sizes each job = P × floor(72h / est_time).

This is getting complex for a plan. Let me present the numbers to show chunk sizing and propose the worker model, then ask 2 focused questions.

Let me compute the numbers for the plan:

**Timing data:**
- ff=0.15: est 3-8h/sim, avg ~5.5h, cores=4
- ff=0.35: est 6-16h/sim, avg ~11h, cores=8

**Capacity within 72h window (2 nodes, 128 cores, if fully free):**
- ff=0.15: 32 concurrent sims; per slot 72/5.5 ≈ 13 sims → ~416 sims/job-window. Way more than 25 moves. So with a parallel runner, one experiment easily fits. Even a single job on ONE node (64 cores = 16 concurrent) does 16×13 ≈ 208 sims. Overkill.
- ff=0.35: 16 concurrent (8 cores each); per slot 72/11 ≈ 6.5 → ~104 sims/job-window. Also fine.

Wait — so with a PARALLEL runner, an experiment (9-25 sims) fits easily in 3 days even at the slowest (25 sims × 16h = 400 sim-hours; with 8 concurrent on one node → 50h). So the ONLY reason they hit the 3-day limit is that the current runner is SEQUENTIAL (one mpirun at a time). A 25-move ff=0.35 experiment sequential = 25×11h ≈ 275h > 72h → dies.

That's a key insight! The sequential runner is the root cause of the 3-day failures. Two ways to fix:
(a) Parallel runner within a job (xargs -P) → one job per experiment suffices.
(b) Split into chunks across jobs (as today's multi-submission approach).

Option (a) is cleaner and fewer jobs. But parallel LAMMPS runs in one job: each mpirun needs its cores; total concurrent = floor(allocated_tasks / cores_per_sim). Allocated tasks per node = free cores. The job requests `--ntasks-per-node=64` (or free). Then the runner launches P concurrent mpiruns each `-np 4` (for ff=0.15). This works and is simpler than chunking across jobs. And the mkdir/flock dedup handles resubmission.

But there's a subtlety: mpirun -np 4 × 16 concurrent = 64 ranks total. OpenMPI can run multiple concurrent mpiruns; with `--allow-run-as-root`. Fine.

However, mixing different ff (different cores/sim) in one job complicates P. If experiments are organized by ff (one experiment = one ff typically? not necessarily — fills is a sweep param; an experiment could have multiple fills!). If an experiment has both 0.15 and 0.35, cores-per-sim differs → P varies. Solution: split the sim list into groups by ff within the job, or just per-experiment-ff chunks. Simpler: controller splits each experiment into **sub-batches by cores/sim** (ff), so each job has uniform cores-per-sim.

So the design:
- **One job per (experiment, ff) sub-batch** (or per experiment if single ff).
- Job requests tasks = min(free_cores, ...) per node.
- Runner runs its sim list in parallel with `xargs -P`, each sim uses cores_by_ff; claim via mkdir, resume incomplete via flock, skip complete.
- est_time_by_ff used by controller to decide if a sub-batch fits 72h (should, given numbers), and to warn.

Hmm wait, but is parallel-mpirun-in-one-job reliable on this cluster? I should ask/flag it. Actually, the current setup already runs mpirun sequentially within a job; running multiple concurrently is standard but I should flag as a risk to verify.

Also, the "shared node with only 32 free cores" case: controller should only request what's free. If the controller asks for 64 but only 32 free, Slurm queues the job until free. The user's norm: 2 nodes max, but only if free. So maybe the controller shouldn't over-request; it should query free cores and request that. Or accept queueing. Let me ask.

Actually, I realize the user probably knows best whether to parallelize within job vs keep multiple jobs. Let me present the timing math (which is the key new insight) and offer the two options:

**Option A: Runner paralelo dentro del job** (xargs -P). One job per experiment (or per ff-sub-batch). Fewer submissions, no manual chunking, self-healing resume. Requires multiple concurrent mpiruns per job (needs validation on cluster).

**Option B: Chunks por job** (estilo actual pero automático). Controller splits sims into chunks sized to fit 72h sequentially (or small parallelism), one job per chunk. Closer to current behavior; more jobs to manage but each job is simpler (sequential as today).

Given the numbers, Option A means no experiment should ever hit 3 days again. Option B is a smaller change to the runner but keeps the multi-job mess.

I think the plan should recommend Option A with B as fallback.

Now, let me also reconsider: does the current job run mpirun with CORES from config? Yes `$MPIRUN="mpirun --allow-run-as-root -np $CORES ..."`. CORES=4 in current config. For ff=0.35 CORES=8 (2x4x1). The config_vars.sh had CORES=32 (old). The point is cores per sim depends on ff.

Also the data files: `box_fill_0.35_SI_100um_90_0_1_0.data` — the datas/ dir has 0.15 and 0.35 files. So an experiment with both fills is possible.

Let me now also nail down the wall-time math to include in the plan:

ff=0.15, 4 cores/sim:
- 9 moves × 5.5h ≈ 50h sequential → fits 72h marginally; 25 moves ≈ 137h → NO.
- With P=16 (one node, 64 cores): 25 sims / 16 concurrent → 2 waves × 5.5h ≈ 11h. Trivial.
- Even P=4 on 16 cores: 25/4 × 5.5 ≈ 35h. Fits.

ff=0.35, 8 cores/sim:
- 9 moves × 11h ≈ 99h sequential → NO.
- With P=8 (one node): 9/8 → 2 waves × 11h ≈ 22h. 25 moves → 4 waves × 11 ≈ 44h. Fits in 72h.

So with a parallel runner, everything fits in ≤2 days even on ONE node with full cores. On shared nodes (32 free cores), ff=0.15 P=8 → 25 sims → 4 waves × 5.5 ≈ 22h. Fine. ff=0.35 P=4 → 25 sims → 7 waves × 11 = 77h → marginal, might need 2 jobs. OK.

So Option A works in almost all cases; worst case marginal needs re-run (and the idempotent resume handles it).

Great. Now let me present this analysis and the revised plan, and ask focused questions:

Q1: Parallel-mpirun-in-job: do they want to validate it on the cluster (Option A) or keep sequential chunks (Option B)?
Q2: Per-ff estimates: confirm est. times and cores mapping (0.15→4 cores 2x2x1; 0.35→8 cores 2x4x1) to hardcode in config.
Q3: Free-cores detection: manual budget vs scontrol auto-detect.
Q4: Since the runner becomes parallel, does the enqueue still need per-job chunking? Probably not → one job per experiment (or per ff-group).

Also the resume choice from before (Q2 restart vs fresh): with parallel runner and restart files every 50000 steps, resume is natural. Still pending decision from earlier round but they said discuss later. I can fold it: "asumo resume desde restart como parte del fix, lo confirmamos al implementar."

Let me keep response focused. Present:
1. Nueva info → implicación: el runner secuencial es la causa raíz de los cortes de 3 días. Mostrar la matemática.
2. Plan revisado con opción A (paralelo en job) vs B (chunks).
3. Preguntas.

I'm in plan mode; no edits. Just analysis + questions. Keep concise but with the numbers. Let me write.
</thinking>

Esa data de tiempos revela algo clave. Hago las cuentas:

**El runner actual es secuencial** (`run_erosion...sh` llama un `mpirun` tras otro en el loop fill→vel→angle→move). Con eso, un job de 72h solo alcanza para:

| ff | time/sim | cores/sim | 9 moves | 25 moves |
|---|---|---|---|---|
| 0.15 | 3–8 h (~5.5) | 4 | ~50 h ✓ | ~137 h ✗ |
| 0.35 | 6–16 h (~11) | 8 | ~99 h ✗ | ~275 h ✗ |

O sea: **el corte a los 3 días no es que "falten cores", es que un job intenta correr 25 sims en serie.** Con el mismo presupuesto pero **corriendo los sims en paralelo dentro del job**, todo entra holgado:

| caso | concurrencia P | 25 moves ff=0.15 | 25 moves ff=0.35 |
|---|---|---|---|
| 1 nodo completo (64c) | P=16 / P=8 | ~2 oleadas ≈ 11 h | ~4 oleadas ≈ 44 h |
| 1 nodo compartido (32c) | P=8 / P=4 | ~4 oleadas ≈ 22 h | ~7 oleadas ≈ 77 h (marginal) |

Esto define dos caminos de diseño:

## Opción A — Runner paralelo dentro del job (recomendada)

El job pide los cores libres (`--ntasks-per-node = cores_libres`) y el runner lanza los sims de su lista con `xargs -P P` (P = cores_libres / cores_por_sim). Cada worker hace el claim: `mkdir` si es nuevo, `flock` + resume si está incompleto, skip si terminó.

- **Un solo job por experimento** (o por sub-batch ff, ya que 0.15→4c y 0.35→8c no comparten P). Se acabó el "32 tareas en 2 nodos" y el dividir a mano.
- El tope de 3 días deja de ser un problema salvo el caso marginal de nodo muy compartido, y ahí el re-envío idempotente + resume lo termina.
- **Riesgo a validar**: múltiples `mpirun` concurrentes dentro de un mismo job de OpenMPI en toko0x (debería andar, pero hay que probarlo en el cluster).

## Opción B — Chunks por job (estilo actual, automatizado)

El controlador parte cada experimento en N chunks que entren en 72h **en serie** (lo de hoy) y manda un job por chunk, con el manifest para no re-enviar. Menos cambio en el runner, pero mantiene el ruido de muchos jobs y el riesgo de cortes a mitad de chunk.

## Preguntas para cerrar el diseño

1. ¿Validamos la **Opción A** en el cluster (probamos 2–3 `mpirun` paralelos en un job) o preferís la B?
2. ¿Confirmás los **cores por ff** como fuente única: 0.15 → `[2,2,1]`=4 y 0.35 → `[2,4,1]`=8, y que van en config (no hardcodeados)? ¿Hay más ff que se vayan a usar?
3. Presupuesto de cores: ¿**manual** (`--tasks` al enqueue) o querés que el controlador detecte cores libres de `scontrol`? (manual = KISS, auto = robusto al nodo compartido)
4. Si un mismo experimento tiene 0.15 **y** 0.35 (distinto cores/sim), ¿lo divido en 2 jobs (uno por ff) — asumo que sí?

Con esto confirmado, cierro el plan de implementación completo (archivos, comandos y orden de pasos).

## User

1 si opcion A
2 si mejor hacer ese mapeo
3 por el momento manual
4 bueno dividilo por ff
continuemos con las preguntas anteriores

## Assistant

<thinking>
The user confirmed:
1. Option A (parallel runner within job)
2. Yes, do the ff → cores mapping as single source
3. Manual budget for now
4. Split by ff into separate jobs

And says "continuemos con las preguntas anteriores" — continue with the previous questions. The previous questions pending were:

From round 1, "Los otros puntos los discutimos luego":
- Q2: Source of truth for experiments: separate `experiments.toml` vs `[[experiment]]` inside config.toml
- Q3: Scope: include auto-resume of incomplete sims (integrate legacy resume_controller.py) or leave for another iteration
- Q4: np parametrized: LAMMPS if/elif in .input vs generated per-job .input

And from round 2:
- Q2: Resume by restart vs fresh re-run (I assumed restart is worth it given 2M steps)

So "las preguntas anteriores" = the pending ones. Let me re-ask/continue with those:

1. **Fuente de verdad de experimentos**: ¿experiments.toml separado o [[experiment]] en config.toml?
2. **Resume por restart vs re-ejecutar fresco** (Q2 from round 2).
3. **np parametrizado**: ¿if/elif de LAMMPS o generar .input por job? (Q4 round 1)
4. **Alcance del resume**: integrar resume_controller de legacy ahora (parte del diseño) — though given the runner now includes resume via flock, this is basically settled as part of Option A. But I should confirm the restart-based resume is in scope.

Actually, let me reconsider what to ask. The user wants me to continue with the previous questions. The previous open questions were:

Round 1 (deferred): 
- #2 source of truth (experiments.toml vs config.toml)
- #3 resume scope
- #4 np parametrization

Round 2:
- #2 resume by restart vs fresh

Since Option A already includes resume-from-restart (flock + resume), #3 from round 1 and #2 from round 2 are related — resume is now IN scope by design. So really the remaining open questions:

1. **Source of truth for experiments**: experiments.toml vs [[experiment]] in config.toml
2. **np parametrization approach**: LAMMPS if/elif vs generated .input per job
3. **Resume restart vs fresh**: confirm restart-based resume (given 2M steps, restart recommended)

Let me also think about whether there are any other open design questions I should raise now that the core model is set:

- The `collision_erosion_bed.input` has orientation-dependent data files. Since we now split by ff and parametrize orientation... The datas exist: `box_fill_0.15_SI_100um_90_0_1_0.data`, `box_fill_0.15_SI_100um_270_0_1_0.data`, etc. So orientation is a per-experiment parameter that selects the data file. This ties into the experiments config. Good to include in the experiments schema.
- Time estimates per ff: needed for the controller to decide "fits in 72h" and warn/split. Should go in config (est_horas per ff) or just be a warning with manual override. I'll propose a simple `est_horas_por_ff` table used only to warn, since chunking across jobs isn't needed with Option A (one job per experiment-ff), except the marginal case. Actually with Option A, per (experiment, ff) = one job. If it doesn't fit 72h, the controller can split into 2 jobs by halving the sim list. But splitting a single ff with moves... moves are the sims. Could split by move range. Keep it simple: warn + optional `--chunks` override. I'll mention.
- Locking: flock file per simdir. Also the resume input file: need a `resume` mode in the LAMMPS input (read_restart vs read_data). Currently legacy has `resume_collision.input`. The runner would need two input templates or one input with a variable that switches. LAMMPS: can't conditionally `read_restart`/`read_data` with `if` easily because `if` can't contain `read_data`. Actually you CAN with `if` + `read_restart`? LAMMPS `if` command only allows specific commands... Actually LAMMPS allows many commands in if bodies. `read_restart` and `read_data` are allowed inside `if`? I believe `read_data`/`read_restart` ARE allowed in `if` bodies since they're normal input script commands — yes, LAMMPS allows them (they're not among the disallowed ones like `quit`, `label`... actually `label` is disallowed). Let me not overcommit; I'll propose a single `.input` with a `restart_file` variable and `if` block: if restart_file != "none" then read_restart else read_data. The legacy resume_collision.input presumably does something like this. Let me check it to ground the plan.

Let me read legacy/resume_collision.input quickly to inform the plan. Also, I recall the runner creates simdir and copies input; the resume would need the restart file path — restart files are written every 50000 steps as res_A.rst/res_B.rst inside the simdir. Good.

Let me also double check: does LAMMPS input support `if "${var}" == ... then "read_restart ..."`? I'm fairly confident yes (conditional execution in input scripts supports most commands except jump/quit/etc.). I'll flag it as an implementation detail to verify.

Also, for the parallel runner: need to confirm how the job requests cores. With Option A: job requests `--ntasks-per-node=64` (or `-N 1 -n 64`), and the runner does `xargs -P 16` with each `mpirun -np 4`. Important OpenMPI consideration: each mpirun creates its own MPI universe; running 16 mpiruns concurrently each with 4 ranks on 64 slots — should work with OpenMPI (each mpirun uses its own resource). Fine.

One more: the `/dev/shm` temp dir setup in slurm_submit — with parallel mpiruns, the shared TMPDIR might be fine. OK.

Let me also reconsider the "experiments.toml vs [[experiment]]" question and give a clear recommendation with reasoning, so the user can just confirm. My recommendation: a separate `experiments.toml` file. Rationale:
- config.toml is the "base" (material, sim defaults, thresholds) — also read by CARBOXI (analysis). Mixing a list of experiments (a batch that changes every run) into the analysis config creates drift: CARBOXI reads config.toml for thresholds. Actually CARBOXI reads ITS OWN config.toml (in CARBOXI/), not the cluster one. The cluster config.toml is separate. Hmm. So sharing isn't an issue. But conceptually, config.toml = "how to simulate", experiments.toml = "what to simulate". A batch is transient; the base config is stable. Separate file keeps git diffs clean when changing what to run. Also `[[experiment]]` arrays in TOML are fine too. I'll recommend separate experiments.toml but present the tradeoff.

Also note: read_config.py generates shell vars from config.toml only. If we add experiments, the runner needs per-experiment vars merged over base config. Approach: for each experiment, controller writes a merged `configs/config_<jobid>.toml` = base config.toml + experiment overrides (fills/vels/angles/moves/np/orientation/nsteps/cores...). read_config.py already reads whatever TOML path you give → perfect. So the merged per-job TOML keeps the existing read_config.py pipeline working unchanged. 

For the ff→grid mapping: put in base config.toml as a table `[grid_por_ff]` with keys "0.15"=[2,2,1], "0.35"=[2,4,1]. read_config.py would generate weird shell vars for dict values (it can't represent dicts → it does `for key, value in values.items()` and writes VAR=... for each; a dict value would be formatted via _format_value which doesn't handle dict → returns str(dict) which is invalid bash). So the ff→grid table can't live in config.toml as-is without extending read_config.py. Options: (a) extend read_config.py to emit e.g. `GRID_FF_0_15=("2" "2" "1")` per key, or (b) keep ff→grid in experiments or in the controller (Python) only. Since the runner (bash) needs cores per sim and grid, and the controller (Python) also needs it for --ntasks/processors, maybe simplest: the controller computes grid from ff and writes processors/cores INTO the merged per-job TOML ([simulation] cores + processors). Then read_config.py emits CORES/PROC_X/Y/Z as today, and the runner needs no changes for grid! That's elegant: the per-ff mapping lives in the controller (Python), and the merged TOML materializes the resolved grid. But then the mapping should be configurable, not hardcoded in the controller → put `grid_por_ff` in experiments.toml or a small table in config.toml that read_config.py knows how to handle.

Simplest for plan: keep `[grid_por_ff]` in config.toml, extend read_config.py to serialize dict values as multiple shell vars (e.g., for each key emit `GRID_FF_0_15=([2 2 1])`). Then both controller and runner use it. But runner doesn't need it if controller resolves per-job. I'll propose: mapping lives in config.toml; controller resolves ff→grid→(cores, processors) and writes them into the merged per-job TOML under [simulation]; read_config.py unchanged. Runner keeps using CORES/PROC_X/PROC_Y/PROC_Z. Minimal changes.

Wait, but the runner's OUTDIR has `_cores_8` hardcoded — we replace with computed from CORES. And Orientation param. And np. These get added as experiment overrides in the merged TOML → read_config.py emits ORIENTATION, NP etc. Runner reads them.

OK. Also the `np` and `orientation` keys: np exists in [simulation] already; orientation doesn't. Add `orientation = "90_0_1_0"` to [simulation] or per experiment. read_config.py uppercases keys → ORIENTATION.

Now the resume decision: I'll recommend restart-based resume (uses res_A/res_B.rst, written every 50000 steps). Confirm with user. The legacy resume_collision.input exists — let me read it to base the plan.

Let me read legacy/resume_collision.input before answering, to ground the np/resume discussion.

Also, let me reconsider the "continuemos con las preguntas anteriores" — the user wants me to continue with the previously pending questions. So I should present the remaining questions, but given I now have solid recommendations, present them with recommendations so the user can confirm quickly. Also present the consolidated plan since most decisions are locked.

Let me read the resume input file and also verify the datas naming to ground orientation param.
</thinking>

**Tool Call: Programas/legacy/resume_collision.input**
Status: Completed

# ==========================================================
# Input script independiente para RETOMAR simulaciones
# ==========================================================
# Variables pasadas desde el Python:
# restart_file, young_modulus, poisson, surface_energy, nsteps, ndumps, nthermo, proc_x, proc_y, proc_z

processors ${proc_x} ${proc_y} ${proc_z}

# Leer el estado exacto donde se cortó
read_restart ${restart_file}

# Re-declarar el par granular (obligatorio tras un read_restart en simulaciones granulares)
pair_style granular
pair_coeff * * jkr ${young_modulus} 1.0e16 ${poisson} ${surface_energy} tangential mindlin NULL 1 0.5 rolling sds 5e09 9.895e-09 0.1 twisting marshall

# Re-declarar los fixes integradores (los grupos 'bloque' y 'sphere' se heredan del restart)
fix 1 bloque nve/sphere
fix 2 sphere nve/sphere

compute 1 all ke/atom

thermo ${nthermo}
thermo_modify lost warn

# Crear un archivo de dump distinto para la continuación (evita corrupciones si hubo corte abrupto)
dump 1 all custom ${ndumps} output_trayectoria_resume.dump id type x y z vx vy vz fz

# ==========================================================
# Re-declarar TRAMPA: Las regiones dinámicas NO se guardan en el restart
# ==========================================================
region r_trampa block INF INF INF INF 2.5e-4 INF
group gr_trampa dynamic all region r_trampa every 1000
fix freno_fuerza gr_trampa setforce 0.0 0.0 0.0
fix freno_vel gr_trampa viscous 1.0

fix balanceo all balance 1000 1.05 shift xy 10 1.05

# Volver a activar los restart iterativos
restart 50000 res_A.rst res_B.rst

# 'upto' corre los pasos restantes exactos. Ejemplo: Si iba por el 1.500.000 y nsteps es 2.000.000, corre 500.000.
run ${nsteps} upto

write_data post-run_final.data

**Tool Call: ls /home/tutturuu/Documentos/Universidad/Tesis/Programas/cluster/datas/**
Status: Completed

box_fill_0.15_SI_100um_180_1_0_0.data
box_fill_0.15_SI_100um_270_0_1_0.data
box_fill_0.15_SI_100um_270_1_0_0.data
box_fill_0.15_SI_100um_90_0_1_0.data
box_fill_0.15_SI_100um_90_1_0_0.data
box_fill_0.15_SI_100um.data
box_fill_0.15_SI_145um.data
box_fill_0.35_SI_100um.data
box_fill_0.35_SI_145um.data


## Assistant

<thinking>
Now I have everything. The `resume_collision.input` from legacy is a complete, working resume input (uses read_restart, re-declares pair_style/fixes/trap, uses `run ${nsteps} upto`, writes post-run_final.data). This is exactly what the runner needs for resume mode.

Key detail from legacy resume: it writes `output_trayectoria_resume.dump` (a different dump name) and uses `run ${nsteps} upto` (nsteps passed = remaining steps presumably... actually the controller passed local_nsteps = original nsteps from salida.log; with `upto` it runs remaining). Also note the legacy resume had `fix freno_vel ... viscous 1.0` while main input uses `1.0e-8`. Minor drift. And balanceo `shift xy` vs `rcb weight time 1.0`. So the resume input is slightly out of sync with the current main input — the plan should unify these (single source input that handles both fresh and resume modes, OR keep two templates but keep them in sync). I'll propose unifying into ONE input file with an `if restart_file` branch if LAMMPS supports read_data/read_restart in if-bodies. Actually, I recall LAMMPS `if` command DOES support arbitrary commands in the then-body as long as they're complete commands. read_data is allowed. But there's a known gotcha: `if` bodies are parsed for variables; `read_data ${data}` inside if works. Let me not be too assertive; I'll propose it as: single input with a variable `restart_file = "none"` and an if/else branch; fallback is to keep two templates and have the runner pick. That's an implementation detail to verify on the cluster (can test quickly).

Now let me consolidate the remaining questions and present the full plan. The pending questions were:

Round 1 deferred: #2 source of truth (experiments.toml vs [[experiment]]), #3 resume scope, #4 np parametrization.
Round 2: #2 resume by restart vs fresh.

Since Option A now includes resume (flock + resume from restart) as a core component, resume scope is settled: IN scope, restart-based. I should confirm that (restart-based, using the legacy resume_collision.input approach, unified).

Remaining to ask:
1. Source of truth: experiments.toml (separate) vs [[experiment]] in config.toml. Recommend separate.
2. np parametrization: single input with LAMMPS `if`/`elif` (create_atoms blocks for NP1/NP3) vs controller generates per-job .input (sed). Recommend the if/elif in input (fewer files, keeps read_config pipeline) — but note create_atoms inside if: LAMMPS allows `if` with `create_atoms` in the body. Actually wait — is there a constraint? `create_atoms` in an `if` body... I believe it works. But there's a subtlety: `create_atoms` needs atoms not yet created — in fresh mode, read_data creates type 1, then create_atoms adds type 2. Fine. Recommend if/elif, verify on cluster.
3. Confirm restart-based resume + that they want the trap/balanceo/freno_vel details unified (I'll just note it).
4. Also confirm: one job per (experiment, ff). And est-time table for warning (est_horas_por_ff) — optional; manual. I'll include as config with defaults 5.5/11 but only used to warn/validate against 72h, and allow `--chunks` override to split. Actually the user said manual budget for cores; for time they gave estimates. I'll include an optional validation that warns if est > 72h and suggests `--split`. Keep it as a warning.

Also I should confirm the OUTDIR / results naming: keep existing pattern but replace `_cores_8` with actual cores, and include orientation. And the Buffer path. This is design detail I can just state in the plan.

Also one more question that came up implicitly: whether to keep the multi-node/2-node flexibility. With Option A, one job per (experiment, ff) on one node is enough per the math. So we mostly use 1 node per job. That's fine and within the 2-node norm.

Let me also present the consolidated file structure and implementation steps so the user sees the whole plan, then ask the final 3 questions.

Let me structure the final plan response:

## Decisiones confirmadas (lock)
- Opción A runner paralelo; presupuesto manual; split por ff; ff→grid mapeo; resume integrado (restart).

## Preguntas finales (las pendientes)
1. experiments.toml separado (recomiendo) vs [[experiment]] en config.toml
2. np por if/elif en el input (recomiendo) vs generar input por job
3. Confirmar resume restart-based con el resume_collision.input de legacy como base (unificado con el input principal)

## Plan de implementación (resumen)
File-by-file changes + steps.

Then ask them to confirm the 3 questions. Since we're in plan mode, present plan and wait for go.

Let me write it concisely. I'll present the questions first (since that's what they asked: "continuemos con las preguntas anteriores"), with recommendations, then the consolidated plan skeleton.

Actually the user said "continuemos con las preguntas anteriores" = continue with the previous questions. So the primary deliverable now is to present those remaining questions. I'll present them with clear recommendations and reasoning, plus briefly the locked decisions, and the implementation plan outline. Keep it tight.

Let me be careful about response length but comprehensive. I'll write in Spanish.
</thinking>

Perfecto, quedan cerradas: **Opción A (runner paralelo), mapeo ff→grid en config, presupuesto manual, split por ff**, y el resume queda **integrado** (es parte del fix del punto 2, ya no es opcional). Con eso retomo las preguntas pendientes:

## Preguntas pendientes (con recomendación)

**P1 — Fuente de verdad de los experimentos.**
- (a) `experiments.toml` separado — recomendado: separa "cómo simular" (config.toml, estable) de "qué simular" (experiments.toml, cambia en cada lote). Git diffs limpios, un archivo por lote.
- (b) `[[experiment]]` dentro de config.toml — un solo archivo, pero mezcla lo transitorio con lo estable.

En ambos casos, el controlador genera un **config.toml mergeado por job** (`configs/config_<jobid>.toml` = base + overrides del experimento), así `read_config.py` sigue funcionando sin tocarlo.

**P2 — Parametrizar `np` (hoy hardcodeado en `collision_erosion_bed.input:57`).**
- (a) `if/elif` de LAMMPS dentro del input (bloques create_atoms NP1/NP3) — recomendado: un solo input, nada que generar.
- (b) El controlador genera un `.input` por job con el bloque correspondiente.
- (a) es más limpio; hay que validar en el cluster que `create_atoms` ande dentro de un `if` (si no, caemos a (b)).

**P3 — Resume desde restart (confirmar).**
Base: el `resume_collision.input` de `legacy/` ya está probado (usa `read_restart` + re-declara pair_style/fixes/trampa + `run ${nsteps} upto`). Propongo **unificar** el input principal y el de resume en uno solo con `if restart_file` (fallback: dos templates sincronizados). Nota: el resume de legacy está desincronizado del input actual (`freno_vel 1.0` vs `1.0e-8`, `balanceo shift xy` vs `rcb weight time 1.0`) — la unificación los deja en un solo lugar.

## Plan de implementación (una vez confirmado lo de arriba)

**Archivos (en `Programas/cluster/scripts/`):**

```
config.toml              # base + [grid_por_ff] {0.15→[2,2,1], 0.35→[2,4,1]} + est_horas_por_ff
experiments.toml         # NUEVO (o [[experiment]]): fills/vels/angles/moves/np/orientation/nsteps
enqueue_sims.py          # NUEVO: mergea base+exp, calcula cores por ff, splits por ff, manda 1 sbatch por (exp,ff), registra en queue/jobs.tsv, idempotente
slurm_submit_JKR.sh      # MOD: --ntasks-per-node desde el config del job; job-name desde experimento
run_erosion_bed_sims_lammps_move_np3.sh  # MOD: xargs -P; claim mkdir/flock; resume si incompleto; grid/orientation/np desde config; OUTDIR sin _cores_8
collision_erosion_bed.input  # MOD: np y resume parametrizados
resume_collision.input   # se unifica con el principal (o se elimina)
queue/jobs.tsv           # NUEVO: manifest (jobid, experimento, ff, estado, fecha)
configs/                 # ya existe: config_<jobid>.toml por job
```

**Lógica del runner (núcleo del fix):**

```
worker(simdir):
  si post-run_final.data existe      → skip (completo)
  elif dir existe y hay res_*.rst    → flock + resume desde restart
  else                                → mkdir (claim) + correr fresco
job: xargs -P P lista_de_sims   (P = ntasks / cores_por_sim)
```

**Verificación previa en cluster (paso 0):** probar N `mpirun` concurrentes en un job + `create_atoms`/`read_restart` dentro de `if`.

**Orden de ejecución:** (0) pruebas en cluster → (1) config+grid → (2) runner paralelo+resume → (3) input parametrizado → (4) enqueue + manifest → (5) `ruff` si hay python y prueba en seco (`--dry-run`).

¿Confirmás P1, P2 y P3 para cerrar el plan?

## User

1 - Prefiero centralizar y tener como un formulario donde el susuario rellene con las combinaciones que desea hacer (pueden ser muchas)

## Assistant

<thinking>
The user answered P1: they prefer to centralize and have a "form" where the user fills in the combinations they want to do (can be many). So they chose option (b)
</thinking>

## User

1 - Prefiero centralizar y tener como un formulario donde el susuario rellene con las combinaciones que desea hacer (pueden ser muchas)
2 - Si metele ese if
3 - Ese script creo que no funciona, habria que plantear otra forma

## Assistant

<thinking>
The user answered:

1. P1 (source of truth): Prefers to centralize — have it as a "form" where the user fills in the combinations they want to do (can be many). So they want a central config that acts as a form/table for experiments. This is essentially option (b) but more "form-like" — a central file where you fill rows of combinations. Could be experiments.toml with a list, or a table. They want centralization and a form-style interface. Since they say "centralizar y tener como un formulario donde el usuario rellene con las combinaciones que desea hacer (pueden ser muchas)" — a form where you fill the combinations, many possible.

So design: a single central file (experiments.toml or a section in config.toml) with a clear table/list of experiments. "Pueden ser muchas" — many experiments. A TOML list of experiment rows is a form. Or a CSV? They said "formulario" — form. A TOML file with a list of experiments is good. Let me propose: central `config.toml` keeps base + a `[[experiment]]` table (array of tables). Or a separate `experiments.toml`. They said "centralizar" — centralize. Hmm, they said in the earlier question option (b) was "un solo archivo". They answered "Prefiero centralizar y tener como un formulario" — so one central file with the form. I'll interpret: `experiments.toml` as THE central "form" (single place where you fill combinations), separate from base config. Actually "centralizar" could mean everything in one place. Let me propose the form be a single file `experiments.toml` (a clear "formulario" with sections/comments to fill), and config.toml remains the base defaults. But they said "centralizar"... 

Let me re-read: "1 - Prefiero centralizar y tener como un formulario donde el susuario rellene con las combinaciones que desea hacer (pueden ser muchas)". I think they mean: instead of separate experiments.toml, centralize it into a single form file where you fill in the combos. The emphasis is on "formulario" — a fillable template. So one file = the form. Given they previously had the choice (a) experiments.toml separado vs (b) [[experiment]] en config.toml, and they say "centralizar", I'd say: a single `experiments.toml` (or a section) acting as the form. But wait — "centralizar" as opposed to "separado"? The options were separate vs embedded. They prefer centralizing — which suggests ONE file containing everything, i.e., the base config AND the experiments together? Or centralize experiments into one file (the form)?

I think the cleanest interpretation that satisfies "formulario que el usuario rellena": a single `config.toml` that contains base settings + the experiment list, acting as the form. Or maybe they want the experiments to be the central thing. Let me ask a quick clarifying question OR just propose the design and confirm. Given they want to move forward, I'll propose: a single central file `experiments.toml` (the "formulario") where each experiment is a row with all params, and base settings (material, grid por ff, umbrales) stay in `config.toml`. This keeps CARBOXI's config.toml reading intact (CARBOXI reads its own config, not cluster's). Actually cluster config.toml is used by read_config.py for the runner. If experiments live in config.toml, read_config.py would iterate sections and might choke on [[experiment]] array-of-tables (it does `for section, values in raw.items()` and `if not isinstance(values, dict): continue` — an array of tables is a list of dicts, so it would skip it. OK, safe). But the controller generates merged per-job configs anyway, so the base config.toml with experiments wouldn't be fed directly to read_config.py. Fine.

Hmm, but do they want everything in ONE file (base + experiments)? "centralizar" strongly suggests one file. Let me propose ONE central file, e.g. `experimentos.toml`... no wait. If they centralize, they might not want two files. Let me just propose: **one central `config.toml`** that has the form: a `[[experiment]]` array of tables, each row = fills/vels/angles/moves/np/orientation/nsteps (optional overrides of the base). The controller merges each experiment over base defaults. This is option (b) — one file. And it acts as the form. I'll go with that, and mention that if they prefer a separate form file it's trivial to change.

Actually, re-reading their answer 1: they said "Prefiero centralizar y tener como un formulario donde el susuario rellene con las combinaciones que desea hacer (pueden ser muchas)". This is a clear preference for (b)-like: one central file that's a form. I'll go with `[[experiment]]` in `config.toml`.

2. P2: "Si metele ese if" — yes, use the LAMMPS if/elif approach for np.

3. P3: "Ese script creo que no funciona, habria que plantear otra forma" — the resume script (resume_collision.input / the resume approach) they think doesn't work; need another approach.

Hmm, important. The user says the resume approach doesn't work and we need another way. So restart-based resume is unreliable. Need to think about alternative.

Why might resume not work? Possible reasons:
- The `read_restart` after kill: restart files res_A.rst/res_B.rst might be corrupted if killed mid-write.
- The trap (dynamic region), balance fix, etc. aren't saved in restart and re-declaring may differ.
- The `run ${nsteps} upto` semantics.
- dump files: the resume writes output_trayectoria_resume.dump, but the analysis pipeline (CARBOXI/trajectory_converter) might expect a single dump per simdir; two dumps (main + resume) complicates the trajectory merging. CARBOXI reads `output.*.gz` or `output_trayectoria.dump`? The main input writes `output_trayectoria.dump`. The resume writes `output_trayectoria_resume.dump`. So post-processing would need to concatenate/merge dumps across the cut — complicated and error-prone. This is likely why "no funciona" — the trajectory dump continuity breaks.

Alternative approach for interrupted sims: **re-run from scratch (fresh) instead of resume**. I.e., the "resume" is: delete the incomplete simdir and re-run it fresh. Simple, robust, but wastes the compute already done (3-16h per sim). Given 2M steps and long runtimes, wasteful but reliable.

Or: **resume via continuing the same dump**: write dumps with `*` restart and re-open... Actually a cleaner resume that keeps trajectory continuity: use `restart` files (binary) BUT make the dump append-compatible. LAMMPS dump can't append to existing file easily. The standard approach: when resuming, use the SAME dump filename but that would create a second file overwriting. Alternatively, make dumps per-chunk (e.g., output files named by range) so the analysis merges them. Complex.

Let me reconsider: maybe the restart resume doesn't work because of the granular pair style re-declaration, or because the `write_data pre-run_with_projectile.data` in main input conflicts, or because after `read_restart` the `comm_style`, `atom_style` etc. must match. Or the killed job leaves restart files from the LAST write (every 50000 steps = 50000×5e-11s = 2.5µs of sim per restart... each 50000 steps takes ~8-14 min). Losing 8-14 min of progress on resume is fine, but if resume is broken, re-running fresh loses everything.

Given the user says "creo que no funciona, habría que plantear otra forma" — they want an alternative. Options:

**Option 1: Fresh re-run of incomplete sims (simplest, robust).**
The runner, on finding an incomplete simdir, deletes it and re-runs it fresh. Since the parallel runner makes everything much faster and one job per experiment-ff fits in 72h, the probability of incomplete sims drops dramatically. And when it does happen (marginal shared-node case), re-running fresh is acceptable. This is KISS and robust. Downside: waste on the rare incomplete case. But with 3-day limit rarely hit now, this is fine.

**Option 2: Chunked dump + append resume (complex, keep progress).**
Make each job write time-chunked dump files (e.g. dump files named `output_chunk_<start>_<end>.dump`), and on resume append new chunks; post-processing concatenates. But the analysis pipeline (CARBOXI trajectory_converter) reads a single trajectory per simdir. Concatenation of LAMMPS dumps across restarts is possible (each dump is a self-contained block with header), but LAMMPS dump header has "TIMESTEP ..." blocks; concatenating is doable with a script. Still, this changes post-processing and is fragile.

**Option 3: Resume but verify integrity; fall back to fresh if restart corrupt.**
Implement resume, but wrap in a validation: if restart file missing/corrupt or resume run fails, delete and re-run fresh. Keep dump naming: main dump `output_trayectoria.dump`; on resume, instead of a separate `_resume` dump, write to `output_trayectoria_<jobid>_resume.dump` and have CARBOXI merge, OR make resume write a NEW dump and add a post-step that concatenates into a single file. Complex.

Given the user's preference for simplicity (KISS) throughout, and that Option A parallel runner makes 3-day truncation rare, I think **Option 1 (fresh re-run)** is the pragmatic choice, possibly with an integrity check: if `post-run_final.data` missing and the restart is newer than X, still just delete+rerun.

But wait — losing hours of compute on the marginal case (nodo compartido, ff=0.35 with 25 moves ≈ 77h → will get cut) means MANY sims re-run fresh = lots of wasted compute. Hmm. In that marginal case, the controller could split the experiment into 2 jobs (via `--split` or automatic by est time) so each fits 72h, avoiding the cut entirely. So: controller warns/splits when est_time > 72h → cuts never happen in practice → fresh re-run for the rare failure is acceptable.

Actually, let me reconsider whether resume "no funciona" due to the dump issue specifically. Let me think about what the trajectory analysis needs. CARBOXI trajectory_converter converts dump → parquet per simdir. It probably globs `output*` files. Let me check quickly? I already have CARBOXI files. trajectory_converter.py — let me check how it reads dumps. Actually I should check to inform the resume design. But I'm in plan mode, read-only — I can read files. Let me check trajectory_converter.py and storage_and_viz for how dumps are consumed. That informs whether resume-with-separate-dump could ever work.

Actually, let me just check trajectory_converter.py quickly.

Also I should think about what "plantear otra forma" means concretely — propose alternatives and let them choose. The key insight: with Option A parallel runner + one job per (exp, ff) + est-time-based split to fit 72h, truncation becomes rare, so **fresh re-run of incomplete sims** is a perfectly reasonable "otra forma" (simpler than restart resume). If they still want to save compute on the frequent case... but they said resume doesn't work.

Let me present:
- Alternative 1 (recomendada): **re-ejecutar fresco** las incompletas (delete + rerun), apoyado por el split automático por tiempo para que los cortes de 3 días no ocurran (o sean raros). Robust, simple, cero riesgo de datos corruptos.
- Alternative 2: **resume con integridad + merge de dumps**: retomar desde restart, escribir dump de continuación en el MISMO archivo formato de bloques (o un segundo dump), y que CARBOXI concatene. Más complejo, y toca el post-proceso.

I should verify the dump consumption to say something accurate. Let me read trajectory_converter.py.

Also, note the user's answer 1 said "pueden ser muchas" — many combinations. So the form must handle many experiments cleanly. A `[[experiment]]` array of tables is fine for many rows. I'll mention a `--dry-run` to preview.

Let me read trajectory_converter.py and check how dumps are read (to ground the resume discussion).
</thinking>

**Tool Call: Programas/CARBOXI/trajectory_converter.py**
Status: Completed

"""
Conversor de resultados CARBOXI a formato compatible con OVITO.

Convierte archivos .parquet con trayectorias de partículas a formatos
que OVITO puede leer y visualizr:
- LAMMPS dump (.dump)
- XYZ (.xyz)

Uso:
    python3 -m CARBOXI.trajectory_converter <archivo.parquet> [--format dump|xyz]
    python3 -m CARBOXI.trajectory_converter <directorio> [--format dump|xyz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def read_parquet(path: Path) -> pl.DataFrame:
    """Lee archivo parquet de resultados CARBOXI."""
    return pl.read_parquet(path)


def get_timesteps(df: pl.DataFrame) -> list[int]:
    """Obtiene lista de timesteps únicos."""
    return sorted(df["i"].unique().to_list())


def export_to_dump(df: pl.DataFrame, timestep: int, output_path: Path) -> None:
    """Exporta un timestep a formato LAMMPS dump."""
    df_step = df.filter(pl.col("i") == timestep)
    
    if df_step.is_empty():
        return
    
    n_atoms = len(df_step)
    
    lines = [
        "ITEM: TIMESTEP",
        str(timestep),
        "ITEM: NUMBER OF ATOMS",
        str(n_atoms),
        "ITEM: ATOMS id type x y z vx vy vz fx fy fz cluster coordinacion",
    ]
    
    for row in df_step.iter_rows():
        id, type_, x, y, z, vx, vy, vz, fz, coord, cluster = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]
        coord_val = coord if coord is not None else 0
        cluster_val = cluster if cluster is not None else 0
        lines.append(f"{id} {type_} {x:.6e} {y:.6e} {z:.6e} {vx:.6e} {vy:.6e} {vz:.6e} {0.0:.6e} {0.0:.6e} {fz:.6e} {cluster_val} {coord_val}")
    
    output_path.write_text("\n".join(lines))


def export_to_xyz(df: pl.DataFrame, timestep: int, output_path: Path) -> None:
    """Exporta un timestep a formato XYZ."""
    df_step = df.filter(pl.col("i") == timestep)
    
    if df_step.is_empty():
        return
    
    n_atoms = len(df_step)
    
    lines = [
        str(n_atoms),
        f"Timestep {timestep}",
    ]
    
    for row in df_step.iter_rows():
        _, type_, x, y, z, coord, cluster = row[0], row[1], row[2], row[3], row[4], row[9], row[10]
        element = "Si" if type_ == 1 else "C"  # Si = sílice (cama), C = proyectil
        coord_val = coord if coord is not None else 0
        cluster_val = cluster if cluster is not None else 0
        lines.append(f"{element} {x:.6e} {y:.6e} {z:.6e} {cluster_val} {coord_val}")
    
    output_path.write_text("\n".join(lines))


def convert_file(
    parquet_path: Path,
    output_dir: Path | None = None,
    format: str = "dump",
) -> list[Path]:
    """Convierte un archivo parquet a trajectory.
    
    Args:
        parquet_path: Ruta al archivo .parquet
        output_dir: Directorio de salida. Si None, usa mismo que parquet.
        format: Formato de salida ('dump' o 'xyz')
    
    Returns:
        Lista de archivos generados.
    """
    df = read_parquet(parquet_path)
    timesteps = get_timesteps(df)
    
    if output_dir is None:
        output_dir = parquet_path.parent
    output_dir = output_dir / f"{parquet_path.stem}_{format}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "dump":
        export_fn = export_to_dump
        ext = "dump"
    else:
        export_fn = export_to_xyz
        ext = "xyz"
    
    generated = []
    for t in timesteps:
        output_path = output_dir / f"traj_{t:04d}.{ext}"
        export_fn(df, t, output_path)
        generated.append(output_path)
    
    return generated


def convert_directory(
    dir_path: Path,
    output_dir: Path | None = None,
    format: str = "dump",
) -> list[Path]:
    """Convierte todos los archivos .parquet en un directorio."""
    parquet_files = list(dir_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No se encontraron archivos .parquet en {dir_path}")
        return []
    
    all_files = []
    for pf in parquet_files:
        print(f"Convirtiendo {pf.name}...")
        files = convert_file(pf, output_dir, format)
        all_files.extend(files)
        print(f"  → {len(files)} archivos generados")
    
    return all_files


def main() -> None:
    """Punto de entrada del conversor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convierte resultados CARBOXI a formato OVITO"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Archivo .parquet o directorio con archivos .parquet",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["dump", "xyz"],
        default="dump",
        help="Formato de salida (default: dump)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Directorio de salida (default: mismo que entrada)",
    )
    
    args = parser.parse_args()
    
    if args.input.is_file():
        files = convert_file(args.input, args.output, args.format)
        print(f"Generados {len(files)} archivos")
    elif args.input.is_dir():
        files = convert_directory(args.input, args.output, args.format)
        print(f"Total: {len(files)} archivos")
    else:
        print(f"Entrada no válida: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()

## Assistant

<thinking>
The trajectory_converter is CARBOXI→OVITO (parquet→dump/xyz), not the raw dump consumer. The raw consumer is probably erosion_analysis.py in CARBOXI, which reads the LAMMPS dump (output_trayectoria.dump) and filters eroded particles → parquet. Let me check erosion_analysis.py / how it discovers dump files per simdir. Let me grep for "dump" and "output" in CARBOXI to see how trajectories are consumed.

Actually, I don't strictly need the full post-processing chain for the plan, but it's relevant to the resume discussion: if resume writes a second dump file (output_trayectoria_resume.dump), the analysis must merge both. Let me check how erosion_analysis reads dumps.
</thinking>

**Tool Call: grep -rn "output\|dump\|glob" /home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/erosion_analysis.py | head -40**
Status: Completed

56:    prev_dump: pl.DataFrame | None = None,
68:    if prev_dump is None or df.is_empty():
71:    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
117:    prev_dump: pl.DataFrame | None,
134:        prev_dump=prev_dump,
175:    prev_dump: pl.DataFrame | None = None
189:            prev_dump=prev_dump,
194:        prev_dump = df_raw.select(["id", "vx", "vy", "vz"])


## Assistant

<thinking>
Let me check how the raw dumps are read — probably in erosion_analysis.py there's a function that reads the dump per simdir. Let me grep more broadly in CARBOXI for "trajectory" / "output" / reading the raw dump, and check main_pipeline.py to see what it processes per simdir.
</thinking>

**Tool Call: grep -rn "trajectory\|read_dump\|\.dump\|output" /home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/main_pipeline.py /home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py | head -30**
Status: Completed

/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/main_pipeline.py:78:            f"  Sin datos de trayectoria (.dump o .gz) en '{sim_dir.name}'. "
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/main_pipeline.py:123:            output_dir=config.dir_resultados,
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:33:    Soporta tanto archivos individuales (.dump) como secuencias legacy (.gz).
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:46:            pattern: Ruta a archivo único ('output.dump') o comodín ('output.*.gz').
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:95:    single_dump = sim_dir / "output_trayectoria.dump"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:103:    gz_files = list(sim_dir.glob("output.*.gz"))
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:105:        wildcard = str(sim_dir / "output.*.gz")
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:112:def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:144:        if f.is_file() and f.name.startswith("output") and f.name.endswith(".gz"):
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:150:                    if f.is_file() and f.name.startswith("output") and f.name.endswith(".gz"):
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:165:    match = re.search(r"output\.(\d+)\.gz", filename)
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:173:    output_dir: Path,
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:179:    output_dir.mkdir(parents=True, exist_ok=True)
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:185:    path = output_dir / f"{sim_id}_TOTAL_{filter_name}{suffix}_{timestamp}.parquet"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:191:    sim_id: str, timestamp: str, output_dir: Path, filter_name: str = "F3C"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:194:    output_dir.mkdir(parents=True, exist_ok=True)
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:195:    path = output_dir / f"{sim_id}_NADA_{filter_name}_{timestamp}.txt"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:210:    output_dir: Path,
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:214:    output_dir.mkdir(parents=True, exist_ok=True)
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:215:    path = output_dir / f"{sim_id}_ONDA_{filter_name}_{step_index}_{timestamp}.parquet"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:267:    output_path = dir_imagenes / f"plot_{sim_id}_{filter_name}_{timestamp}.svg"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:268:    plt.savefig(output_path, dpi=250, bbox_inches="tight")
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:271:    print(f"  Gráfico guardado en: {output_path}")
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:484:    output_path = dir_imagenes / f"plot_{sim_id}_{filter_name}_{timestamp}.svg"
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:485:    plt.savefig(output_path, dpi=250, bbox_inches="tight")
/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI/storage_and_viz.py:488:    print(f"  Gráfico guardado en: {output_path}")


## Assistant

<thinking>
CARBOXI reads per simdir: `output_trayectoria.dump` (single) OR `output.*.gz` sequence. The resume dump would be `output_trayectoria_resume.dump` — NOT matched by the single-file path (which looks for exactly `output_trayectoria.dump`) and not matched by `output.*.gz`. So a resumed sim's continuation data would be silently ignored by CARBOXI. That's a concrete reason the resume approach "no funciona" end-to-end (plus restart corruption risk).

So the alternative "otra forma" needs to account for this. Options:

**Alt 1 — Re-ejecutar fresco (recomendada):**
- Runner: si el simdir existe pero sin `post-run_final.data` → `rm -rf` + correr fresco.
- Apoyado por: split automático por est. tiempo para que los jobs entren en 72h → los cortes se vuelven raros (solo el caso marginal nodo-compartido + ff=0.35 + muchos moves).
- Robust: no depende de restart binarios ni de merge de dumps; CARBOXI consume un solo `output_trayectoria.dump` por simdir, que queda limpio.
- Costo: si un sim se corta, se pierde lo avanzado (3-16h). Pero con el split y el runner paralelo, la probabilidad baja mucho.

**Alt 2 — Resume por bloques de dump + merge (guardar progreso):**
- El dump se escribe en bloques por rango (`output_<start>_<end>.dump`), el resume genera el bloque siguiente, y CARBOXI concatena. Toca el pipeline de análisis (storage_and_viz lee un solo archivo o `output.*.gz`).
- Más complejo, toca post-proceso, riesgo de corrupción de restart. No lo recomiendo para esta iteración.

Actually there's a middle option:

**Alt 3 — Dump continuo append:**
LAMMPS doesn't support append mode for dumps natively in a simple way... There's `dump_modify` ... no append. Skipping.

So recommend Alt 1. Given they said the resume script "no funciona" and to propose otra forma, Alt 1 (fresh re-run) is the natural KISS answer, and I'll reinforce it with the auto-split so cuts rarely happen.

Now let me also finalize the central form design for P1:

"centralizar y tener como un formulario donde el usuario rellene con las combinaciones que desea hacer (pueden ser muchas)".

So ONE central file that acts as the form. I'll propose `config.toml` as the single central file containing:
- base: [simulation], [material], [sweep]→ actually the sweep is per experiment now.
- [grid_por_ff]: ff→[px,py,pz]
- [[experiment]]: array of tables, each a row with fills, vels, angles, moves, np, orientation, nsteps, (nombre/label opcional)

The user fills [[experiment]] rows. "Pueden ser muchas" — fine, TOML arrays of tables.

But wait — there's a subtlety with read_config.py: it iterates sections and skips non-dict values. `[[experiment]]` = list of dicts → skipped. Good, no break. But the controller generates merged per-job TOMLs from base + each experiment, so read_config.py gets a clean TOML without the experiments section. Good.

Alternatively, the "form" could be the experiments section and the user edits one file. I'll go with one file `config.toml` that has both base and `[[experiment]]`. Controller extracts experiments, merges with base, resolves grid by ff, submits.

Hmm, one concern: CARBOXI also has its own config.toml in CARBOXI/ (the analysis one). The cluster's config.toml in scripts/ is separate. So no conflict.

Now, the est-time-based split: with Alt 1 (fresh re-run), I want the controller to split an experiment into multiple jobs when est_total > 72h so cuts don't happen. est_total = n_sims × est_horas_por_ff (sequential within job) — wait no, with parallel runner P, wall time ≈ ceil(n_sims/P) × est_horas_por_sim. The controller knows P = floor(tasks_manual / cores_por_ff). It can warn if est_wall > 72h and suggest/split automatically. But "P" depends on free cores which is manual. So: user passes `--tasks` (total concurrent sims budget, e.g. 16). Controller: for each (exp,ff): P_exp = min(--tasks, ...). est_wall = ceil(n_sims/P_exp) × est_horas_por_ff. If est_wall > 72h → warn + either split the sim list into 2 jobs (chunks of moves) or let user raise --tasks. Since split adds complexity, maybe just warn and let the runner's idempotent re-run handle it. But the whole point is avoiding the cut. I'll include the automatic split as a config option `max_horas = 72` with default, splitting sims into chunks that fit. Actually splitting sim list of a single (exp,ff) by moves is easy (slice the sim list in half). Two jobs for the same (exp,ff), each takes a slice. That's fine with mkdir/flock claims (slices are disjoint anyway). Keep it simple: controller splits the sim list into the minimum number of chunks such that each chunk's est_wall ≤ 72h, submits one job per chunk. Each chunk job runs its slice in parallel. This is clean.

But careful: with disjoint slices, the mkdir/flock cross-job dedup isn't even needed for same-exp chunks (disjoint). It's still needed if the user re-runs enqueue (idempotency): chunks recomputed, but completed sims are skipped by runner. Fine.

OK. Let me also decide the manifest: queue/jobs.tsv rows: chunk_id, experiment_index, ff, sim_list (or count), jobid, submitted_at, state. State updated by... the runner? The controller on re-run checks squeue + Buffer completeness. Keep manifest as an audit log; status derived from filesystem + squeue. Good, no need for runner to write state.

Now, I need to present the final consolidated plan and the P3 resolution (fresh re-run), confirm with user, and then we exit plan mode. But the user still needs to confirm P3's "otra forma". Let me propose the alternatives for P3 and recommend fresh re-run, ask to confirm.

Also I should note the concrete reason the resume fails end-to-end (CARBOXI only reads `output_trayectoria.dump` or `output.*.gz`, so the `_resume.dump` would be ignored), which supports "no funciona" and justifies fresh re-run.

Let me now write the final response:
1. P1 locked: single central form in config.toml ([[experiment]]).
2. P2 locked: if/elif in input.
3. P3: why resume fails + alternatives, recommend fresh re-run, ask confirm.
4. Updated full plan summary (files, runner logic, controller logic, verification, order).

Keep concise but complete. This is plan mode, ending with a request to confirm P3 (and maybe the auto-split by time). Let me present.

Actually, let me reconsider whether to ask more or present the final plan for approval. The user has answered all questions. P3 needs their pick. I'll present the plan with the recommended P3 solution and ask for green light, offering the alternative if they want to keep progress.

Let me write.
</thinking>

Revisé cómo consume los dumps el análisis: `storage_and_viz.py:95` lee **exactamente** `output_trayectoria.dump` (o `output.*.gz`). El resume de legacy escribe `output_trayectoria_resume.dump` → **ese archivo sería ignorado por CARBOXI**. Además el restart binario (`res_A/res_B.rst`) puede quedar corrupto si el job se corta en mitad de la escritura. Eso explica el "no funciona" y confirma que hay que replantear.

Con eso, cierro las tres:

## P1 — Formulario central único ✓
Un solo archivo: **`config.toml`** con base + una tabla `[[experiment]]` que es el formulario a rellenar (muchas filas posibles):

```toml
[sweep]   # base
grid_por_ff = { "0.15" = [2,2,1], "0.35" = [2,4,1] }
est_horas_por_ff = { "0.15" = 5.5, "0.35" = 11.0 }
max_horas_job = 72

[[experiment]]   # ← el usuario agrega filas
fills = [0.15]
vels = [-50]
angles = [90]
moves = [-3, -2, -1, 0, 1, 2, 3]
np = 1
orientation = "90_0_1_0"
nsteps = [2000000]   # opcional (default = base)

[[experiment]]
fills = [0.35]
... (otra fila)
```

El controlador genera por job un TOML mergeado (base + experimento, con `cores`/`processors` resueltos por ff) → `read_config.py` no cambia.

## P2 — np por `if/elif` en el input ✓
`collision_erosion_bed.input` pasa a tener `variable np index 1` + bloque `if "${np}==1" then "create_atoms..." elif "${np}==3" ...`. (Validar en cluster que `create_atoms` ande dentro de `if`.)

## P3 — Reemplazo del resume: re-ejecutar fresco (recomendado)

- El runner, si el simdir existe **sin** `post-run_final.data` → `rm -rf` del simdir y lo corre de cero.
- No depende de restart binarios ni de merge de dumps; cada simdir termina con un solo `output_trayectoria.dump` coherente para CARBOXI.
- El costo (perder 3–16 h de un sim cortado) se mitiga con el **split automático por tiempo**: el controlador parte cada experimento en chunks que estimen ≤72 h (usa `est_horas_por_ff` y la concurrencia `P`), así los cortes dejan de ocurrir salvo casos raros, y el re-envío idempotente los recupera.

Alternativa B (más cara): resume por restart + dump por bloques concatenados — toca el post-proceso de CARBOXI y agrega riesgo de corrupción. No la recomiendo para esta iteración.

---

## Plan consolidado

**Archivos en `Programas/cluster/scripts/`:**

| Archivo | Cambio |
|---|---|
| `config.toml` | base + `[[experiment]]` (formulario) + `grid_por_ff` + `est_horas_por_ff` |
| `enqueue_sims.py` | NUEVO: lee experimentos, resuelve grid por ff, calcula chunks ≤72h, mergea config por job, 1 `sbatch` por chunk, registra `queue/jobs.tsv`, idempotente, `--dry-run` |
| `slurm_submit_JKR.sh` | `--ntasks-per-node`/job-name desde el config del job (sin hardcodeos) |
| `run_erosion_bed_sims_lammps_move_np3.sh` | `xargs -P`; claim `mkdir`/`flock`; incompleto → re-run fresco; grid/orientation/np desde config; OUTDIR sin `_cores_8` |
| `collision_erosion_bed.input` | `np` por `if/elif`; resume eliminado (o solo resta el input principal) |
| `queue/jobs.tsv` | NUEVO: manifiesto (chunk, experimento, ff, jobid, fecha) |

**Runner (núcleo del fix):**
```
worker(sim):
  si post-run_final.data existe  → skip
  si dir existe sin final        → rm -rf + correr fresco
  si no existe                    → mkdir (claim) + correr
job: xargs -P P lista_de_sims    (P = ntasks / cores_por_ff)
```

**Orden:** (0) validar en cluster: `mpirun` paralelos en un job + `create_atoms` en `if` → (1) `config.toml`+grid → (2) runner paralelo → (3) input `np` → (4) `enqueue_sims.py`+manifiesto → (5) `--dry-run` + `ruff`.

¿Le doy luz verde con **P3 = re-ejecutar fresco** (y la split automática por ≤72 h), o preferís la alternativa B del resume?

## User

pueden haber varias rotaciones en el .toml, por otro lado el reejectuar me gusta

## Assistant

<thinking>
The user confirmed:
1. P3 = re-ejecutar fresco ✓ (they like it)
2. New point: "pueden haber varias rotaciones en el .toml" — there can be multiple orientations/rotations in the .toml. So orientation is a per-experiment list, not a single string. Also "rotaciones" may refer to the orientation as a sweep parameter.

This changes the design: orientation can be a list in each experiment row, e.g. `orientations = ["90_0_1_0", "270_0_1_0"]`. Each orientation selects a different `.data` file (the datas dir has files per orientation: `box_fill_0.15_SI_100um_90_0_1_0.data`, `270_0_1_0.data`, `180_1_0_0.data`, `90_1_0_0.data`, `270_1_0_0.data`). So the cartesian product expands by orientation.

Wait — but the `.input` file uses rotation via `displace_atoms sphere rotate ... ${xrotate} ${yrotate} ${zrotate} ${angle}` (commented out currently) and the data files are pre-rotated variants. Actually looking at the datas: `box_fill_0.15_SI_100um_90_0_1_0.data` — the suffix `90_0_1_0` is orientation `angle_axis`? The OUTDIR uses `Orientation="90_0_1_0"`. And the .input has `xrotate/yrotate/zrotate/angle` variables used for displace rotation (commented). Hmm.

So orientation in the naming = the data file rotation. Multiple orientations = multiple data files per ff. The experiment row should allow a list of orientations, each generating sims with that orientation's data file and OUTDIR variant.

Also note: the current OUTDIR is `sims_SI_erosion_bed_100um_${Orientation}_ff_...`. So orientation is part of the batch dir name. Each orientation produces a separate batch dir. The simdir names `erosion_bed_SI_ff_..._np_..._vel_..._angle_..._move_...` do NOT include orientation — CARBOXI parses ff/np/vel/angle/move from simdir name but not orientation. If two orientations produce sims with identical ff/np/vel/angle/move, the simdirs would collide if they were in the same directory. But they live under different batch dirs (sims_SI_erosion_bed_100um_90_0_1_0_ff_... vs _270_0_1_0_...). So orientation distinguishes batch dirs. Good — CARBOXI's discover_simulations finds `erosion_bed_SI_*` subdirs within a base_dir. Fine.

But wait: there's a subtlety. If orientation varies within one experiment, the batch dir per orientation differs. The runner currently computes one OUTDIR (batch dir) from the whole config. So the chunking must group by (exp, ff, orientation)? Actually since orientation → data file → batch dir, the natural grouping for jobs is (exp, ff, orientation) OR the runner iterates orientations inside one job, creating separate batch dirs. Since orientation doesn't change cores (grid depends only on ff), multiple orientations can share a job — the runner loops orientations and within each does the sweep. P depends only on ff. So one job per (exp, ff), and within it, loop orientations (each with its own data file and batch dir).

Hmm, but the simdir dedup (mkdir claim) — simdirs under different orientation batch dirs don't collide, so no problem.

Actually wait — do we even need to worry about the orientation affecting the data file? The data file: `box_fill_${fill}_SI_100um_${Orientation}.data`. For fill 0.15 and orientation "90_0_1_0" → `box_fill_0.15_SI_100um_90_0_1_0.data` ✓ exists. For orientation "270_0_1_0" ✓ exists. For "180_1_0_0" ✓. Good. But note: there's no `box_fill_0.35_SI_100um_90_0_1_0.data`! The 0.35 datas only have `box_fill_0.35_SI_100um.data` and `box_fill_0.35_SI_145um.data` — no orientation variants! Hmm interesting. So for ff=0.35, the data file naming without orientation is `box_fill_0.35_SI_100um.data`. 

Wait, let me recheck the datas listing:
```
box_fill_0.15_SI_100um_180_1_0_0.data
box_fill_0.15_SI_100um_270_0_1_0.data
box_fill_0.15_SI_100um_270_1_0_0.data
box_fill_0.15_SI_100um_90_0_1_0.data
box_fill_0.15_SI_100um_90_1_0_0.data
box_fill_0.15_SI_100um.data
box_fill_0.15_SI_145um.data
box_fill_0.35_SI_100um.data
box_fill_0.35_SI_145um.data
```

So ff=0.35 has NO orientation-suffixed files. That means orientation rotations currently only exist for ff=0.15 (the 100um variants: 90_0_1_0, 90_1_0_0, 180_1_0_0, 270_0_1_0, 270_1_0_0). And there's the base `box_fill_0.15_SI_100um.data` (no orientation).

So the orientation handling: if orientation is specified, use `box_fill_${fill}_SI_100um_${orientation}.data`; if "none" or empty, use `box_fill_${fill}_SI_100um.data`. The controller/runner should check file existence and error/warn. Good design detail to note.

Also "pueden haber varias rotaciones en el .toml" — so the experiment form should allow `orientations = ["90_0_1_0", "270_0_1_0"]`. And possibly the rotations relate to the `angle` sweep? No — `angle` is a separate sweep param (the rotation angle of the projectile?). Actually in the .input, `angle` is used in `displace_atoms sphere rotate ... ${angle}` (commented out). And simdir name includes `angle_${angle}`. And the sweep has `angles`. So `angle` is a distinct parameter. The orientation (data file rotation) is different from the projectile rotation angle. OK.

Hmm, actually wait. Let me reconsider: is "rotaciones" the orientation (data file suffix) or the angle? The user said "pueden haber varias rotaciones en el .toml". Given the context — I proposed `orientation = "90_0_1_0"` as a single value in the experiment form — the user is responding that there can be multiple. So `orientations` becomes a list. Yes.

So update the plan:
- Experiment form: `orientations = ["90_0_1_0", ...]` (list), each swept in the cartesian product.
- Runner: for each orientation → select data file `box_fill_${fill}_SI_100um_${orientation}.data` (or base if orientation empty), batch dir includes orientation.
- Grid depends on ff only → one job per (exp, ff), orientations looped inside.
- Chunking/est time: orientations multiply sims count.

Also note: the batch dir OUTDIR currently: `sims_SI_erosion_bed_100um_${Orientation}_ff_${Fills...}`. With multiple orientations per experiment, either one batch dir per orientation, or a batch dir per experiment. CARBOXI's discover_simulations with prefix "erosion_" finds dirs starting with "erosion_" — the simdirs. Fine.

Actually, let me reconsider the whole OUTDIR naming with multiple orientations. If the runner loops orientations, it should create one batch dir per orientation (since orientation is in the batch dir name). The batch dir name pattern `sims_SI_erosion_bed_100um_${Orientation}_ff_...`. OK.

Now let me also double check: does orientation affect the simdir name? No — simdirs are `erosion_bed_SI_ff_${fill}_np_${NP}_vel_${vel}_angle_${angle}_move_${xm}_${ym}_${zm}` under each orientation's batch dir. So the same sim params under different orientations are distinct by batch dir. CARBOXI processes one batch dir at a time (or the base dir with subdirs). If the user points CARBOXI at the parent containing multiple batch dirs, discover_simulations looks for subdirs starting with "erosion_bed_SI_" recursively? No — it does `base_dir.iterdir()` for dirs starting with prefix. If base is the parent, batch dirs start with "sims_SI_" — not "erosion_". So CARBOXI must be pointed at each batch dir. That's existing behavior, not our concern now.

Now, do multiple orientations interact with the ff grouping for jobs? Jobs are per (exp, ff). Orientations loop inside. The est-time calc: n_sims = n_orientations × n_fills... wait, fills can be a list per experiment too. Let me reconsider grouping.

Experiment row: fills=[0.15], vels=[-50], angles=[90], moves=[7], orientations=[90_0_1_0, 270_0_1_0], np=1. The cartesian product is fills × vels × angles × moves × orientations. Jobs split by ff (grid). Within a job, all combinations of that ff. Sim count = n_fills(within this ff) × vels × angles × moves × orientations.

The chunking by est time: est_wall = ceil(n_sims / P) × est_horas_por_ff. Split into chunks of the move/orientation space. Since splitting the cartesian product into time-bounded chunks is more complex, the simplest: split the flattened sim list into chunks of size floor(P × 72h / est_horas_por_ff). Each chunk job processes its slice in parallel. Clean.

I think I have enough. Let me also reconsider the `angle` parameter: it's in the sweep. And the .input uses `angle` for projectile rotation. With NP1, `displace_atoms sphere rotate` is commented out, so angle only affects simdir naming? Actually the simdir name includes `angle_${angle}` and it's passed as -v angle. But if displace rotate is commented, angle does nothing physically currently. That's a separate concern (maybe discussed later). Not my problem now, but note the sim naming includes angle.

Now the final plan is essentially complete. The user said "el reejectuar me gusta" (likes re-execute fresh). And "pueden haber varias rotaciones en el .toml". 

I'm in plan mode. I should finalize the plan incorporating multiple orientations, present the updated design, and ask if ready to implement (exit plan mode). Or maybe there are remaining loose ends.

Let me think about remaining loose ends to tie:
1. Multiple orientations → update experiment schema (orientations list) and runner (data file selection per orientation, batch dir per orientation).
2. The `.data` file selection logic: orientation present → `box_fill_${fill}_SI_100um_${orientation}.data`; absent/empty → `box_fill_${fill}_SI_100um.data`. Validate existence (0.35 has no orientation variants → warn).
3. Confirm the est-hours default values and max_horas_job=72 config.
4. The `--tasks` manual budget semantics: user passes total concurrent sims? Or per-node? Since jobs request cores via --ntasks-per-node... With one job per (exp,ff) and parallel runner, the job requests ntasks = P × cores_por_ff (all on one node). The controller needs to decide P = floor(--tasks / cores_por_ff)? Let me define `--tasks` as "concurrent sims per job" (manual budget). E.g., --tasks 16 → for ff=0.15 (4c): job requests 64 cores, P=16. For ff=0.35 (8c): P=8, 64 cores. Hmm but user mentioned "32 tareas de 4 nucleos en 2 nodos" — meaning they sometimes spread across 2 nodes. With one job per (exp,ff) we mostly need 1 node. But if a node is shared (32 free), requesting 64 waits. The user chose "manual" budget — so they'll pass --tasks matching what's free. Maybe also `--nodes`? Keep it: `--tasks` = concurrent sims per job (manual), jobs are single-node. If they want 2 nodes, they can set tasks accordingly but a single job can't span... actually one job CAN span 2 nodes (`--nodes 2`) with --ntasks-per-node. The runner uses xargs -P P; mpirun per sim uses cores_por_ff ranks; total ranks = P×cores. If P×cores spans 2 nodes, mpirun scheduling may place ranks across nodes (mpirun -np on an allocated multi-node job → ranks spread across nodes). That's fine for LAMMPS (processors grid 2x2x1 across nodes? weird but works). Hmm, distributing a 4-rank sim across 2 nodes is bad. So for ff=0.15 with 4-rank sims, a sim should fit on one node. The job requests 64 cores on node A; if only 32 free... Slurm queues until 64 free on one node? Not necessarily—Slurm can place a 64-core job on a node with 64 free. If node is shared, job waits. So the user, knowing free cores, passes --tasks such that P×cores ≤ free. Manual. Fine.

I'll define: `--tasks N` = número de sims concurrentes por job (manual). Controlador deriva: ntasks = N × cores_por_ff, y rechaza si > 64 (un nodo) salvo que pase --nodes 2. Keep simple: warn if ntasks > 64.

Actually, I realize I should keep this discussion light and just present the final consolidated plan with the orientation update, and confirm readiness. The user seems ready to move forward. Since plan mode requires me not to implement, I'll present the final plan and ask for approval to implement.

Let me also reconcile: with orientations as a list, the "job por (exp, ff)" still holds. Each job loops orientations. Batch dir per orientation.

One more: the `Orientation` variable in OUTDIR — the batch dir name `sims_SI_erosion_bed_100um_${Orientation}_ff_${fills}_vel_${vels}_move_${moves}_NP_${NP}_angle_${angles}_cores_${CORES}`. With multiple orientations and multiple fills in one experiment, the batch dir name includes all fills/vels etc. concatenated. If orientations vary, batch dir per orientation. So batch dir name should include orientation. Fine — loop orientations, build OUTDIR per orientation.

Hmm, wait: could there be multiple fills in one experiment too? Yes, fills is a list. The batch dir name has `ff_${Fills// /_}` (all fills). So batch dir groups by (experiment) essentially, but orientation breaks it. If an experiment has 2 fills and 2 orientations → 4 batch dirs (one per orientation, each containing both fills' sims). Simdirs differ by fill → no collision within a batch dir. OK.

Let me now also double-check the `.input` `if` for np and the resume removal:
- Remove resume? We're not removing the restart writing (restart 50000 res_A.rst res_B.rst stays — harmless and useful for debugging). We just don't USE them for resume. Actually the fresh re-run approach deletes incomplete dirs and re-runs — the restart files are irrelevant. Keep them or remove? Keeping them wastes disk. The main input writes `restart 50000 res_A.rst res_B.rst` — two binary restart files per sim (~can be large). Since we don't resume, we could remove that line to save disk. But maybe user wants them for OVITO inspection? I'll propose removing them (or keep configurable `guardar_restart = false` default). Let me note it as a small decision; default: keep configurable, default off to save disk. Actually, to avoid over-asking, I'll put it in the plan as "se elimina el restart periódico (se ahorra disco) a menos que lo quieras para debug" — ask inline, minor.

Now let me present the final plan. Keep it tight. Then ask for approval to start implementing (which would exit plan mode).

Let me write the final consolidated plan with:
1. Confirmaciones lock: P1 form central, P2 if/elif, P3 fresh re-run.
2. New: orientations as list.
3. Updated schema example.
4. File changes table.
5. Runner + controller logic.
6. Small open details: restart files removal, est_horas defaults, --tasks semantics.
7. Implementation order.

I'll present and ask for green light.
</thinking>

Confirmado: **P3 = re-ejecutar fresco** (solo se `rm -rf` del simdir incompleto y se corre de nuevo). Y agrego lo de las **rotaciones**: `orientations` pasa a ser lista en el formulario, y cada orientación selecciona su `.data` y genera su propio directorio de lote. Verifiqué que en `datas/` las orientaciones existen solo para ff=0.15 (`box_fill_0.15_SI_100um_90_0_1_0.data`, `270_0_1_0.data`, `180_1_0_0.data`, `90_1_0_0.data`, `270_1_0_0.data`); ff=0.35 **no** tiene variantes → el controlador valida existencia del `.data` y advierte si la orientación no existe para ese ff.

## Formulario central `config.toml` (esquema final)

```toml
[sweep]
grid_por_ff      = { "0.15" = [2,2,1], "0.35" = [2,4,1] }
est_horas_por_ff = { "0.15" = 5.5,     "0.35" = 11.0 }
max_horas_job    = 72
guardar_restart  = false

[[experiment]]
fills       = [0.15]
vels        = [-50]
angles      = [90]
moves       = [-3, -2, -1, 0, 1, 2, 3]
orientations = ["90_0_1_0", "270_0_1_0"]   # ← lista; "" = data base sin orientación
np          = 1
nsteps      = [2000000]   # opcional, default = base

[[experiment]]
fills = [0.35]
...
```

## Plan de implementación

**Archivos en `Programas/cluster/scripts/`:**

| Archivo | Cambio |
|---|---|
| `config.toml` | formulario central (`[[experiment]]` + `grid_por_ff` + `est_horas_por_ff` + `max_horas_job`) |
| `enqueue_sims.py` | **NUEVO**: parsea experimentos → por cada uno, agrupa por ff (resuelve grid/cores) → parte el producto cartesiano (×orientations) en chunks ≤72 h → genera `config_<jobid>.toml` mergeado → 1 `sbatch` por chunk → registra `queue/jobs.tsv` → idempotente (`--dry-run`, presupuesto manual `--tasks N`) |
| `slurm_submit_JKR.sh` | `--ntasks-per-node` y job-name desde el config del job |
| `run_erosion_bed_sims_lammps_move_np3.sh` | loop orientations → selecciona `.data` → `xargs -P` (P = ntasks / cores_por_ff) → worker: completo=skip / incompleto=`rm -rf`+fresco / nuevo=`mkdir` claim → OUTDIR sin `_cores_8`, con orientación |
| `collision_erosion_bed.input` | `np` por `if/elif` (create_atoms NP1/NP3) |
| `queue/jobs.tsv` | manifiesto generado |

**Lógica del worker** (núcleo del fix):
```
si post-run_final.data existe  → skip
si dir existe sin final        → rm -rf y correr fresco
si no existe                    → mkdir (claim) y correr
```

**Verificación previa en cluster (paso 0):** N `mpirun` concurrentes en un job + `create_atoms` dentro de `if` de LAMMPS.

**Orden:** (0) pruebas cluster → (1) config+grid → (2) runner paralelo+orientations → (3) input `np` → (4) `enqueue_sims.py`+manifiesto → (5) `--dry-run` + `ruff`.

Detalles menores a definir al implementar: eliminar el `restart 50000 res_A/res_B.rst` periódico del input (ahorra disco, `guardar_restart=false` por defecto), y la semántica exacta de `--tasks` (sims concurrentes por job, manual).

¿Le doy luz verde y arranco con el paso 0 (validar en cluster) y la implementación?

## User

Bien, adelante

## User

tienes luz verde{}

## User

hola?
