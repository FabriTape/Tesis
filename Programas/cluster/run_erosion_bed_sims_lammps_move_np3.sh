#!/bin/bash
# =============================================================================
# Script de simulaciones LAMMPS — erosión de cama granular
# =============================================================================
# Lee parámetros de config.toml via read_config.py
# Uso: ./run_erosion_bed_sims_lammps_move_np3.sh
# =============================================================================

INPUT="collision_erosion_bed.input"
BASE_DIR=$(pwd)
#source ~/.venv/bin/activate
PY311="python3"
# Cargar configuración centralizada
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
$PY311 ${SCRIPT_DIR}/read_config.py ${SCRIPT_DIR}/configs/config_$1.toml
#$PY311 ${BASE_DIR}/read_config.py ${BASE_DIR}/config.toml
source config_vars.sh

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
OUTDIR="${BASE_DIR}/Buffer/sims_SI_erosion_bed_100um_ff_${Fills// /_}_vel_${Vels// /_}_move_${Moves// /_}_NP_${NP}_angle_${Angles// /_}_cores_${CORES}"
COUNTER=0

mkdir -p "$OUTDIR"
cd "$OUTDIR"
cp "${BASE_DIR}/$0" ./
cp "${SCRIPT_DIR}/configs/config_$1.toml" ./

for fill in "${FILLS[@]}"
do
	DATA="box_fill_${fill}_SI_100um.data"
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
				if [ ! -d "$simdir" ]; then
					mkdir "$simdir"
					cd "$simdir"

					cp "${BASE_DIR}/$INPUT" .

					nsteps=${TOTAL_STEPS[$COUNTER]}

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
				fi
			done
		done
		COUNTER=$((COUNTER + 1))
	done
	COUNTER=0
done
