#!/bin/bash
set -euo pipefail

INPUT="collision_erosion_bed.input"
BASE_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBID="${1:?Uso: $0 <SLURM_JOB_ID>}"
CONFIG="${SCRIPT_DIR}/configs/config_${JOBID}.toml"

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
# Cargar configuración mergeada del job (generada por enqueue_sims.py)
$PY311 ${SCRIPT_DIR}/read_config.py ${CONFIG} ${JOBID}
source ${SCRIPT_DIR}/configs/config_vars_${JOBID}.sh
rm -f ${SCRIPT_DIR}/configs/config_vars_${JOBID}.sh

# Backward-compat de variables opcionales
if [ -z "${NSTEPS[0]:-}" ]; then NSTEPS=(2000000); fi
if ! declare -p ORIENTATIONS >/dev/null 2>&1; then ORIENTATIONS=(""); fi

# Variables derivadas
R=$GRAIN_RADIUS_UM

# Desplazamientos XMOVE, YMOVE a partir de MOVES y R
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

# Concurrencia: P sims en paralelo, cada uno con CORES ranks MPI.
# P lo define el job ([slurm].tasks), con fallback ntasks/cores.
if [ -n "${TASKS:-}" ]; then
	P="$TASKS"
else
	P=$(( ${NTASKS:-1} / CORES ))
fi
[ "$P" -lt 1 ] && P=1

# ---------------------------------------------------------------------------
# Worker: corre UNA simulación. Lógica de idempotencia:
#   si post-run_final.data existe → skip
#   si dir existe sin final       → re-ejecutar fresco (limpiar contenido)
#   si no existe                  → crear y correr
# El flock evita que dos procesos tomen el mismo simdir a la vez.
# ---------------------------------------------------------------------------
run_one_sim() {
	local ori="$1" fill="$2" vel="$3" angle="$4" xm="$5" ym="$6" nsteps="$7"
	local batchdir="sims_SI_erosion_bed_100um_${ori}_ff_${Fills// /_}_vel_${Vels// /_}_move_${Moves// /_}_NP_${NP}_angle_${Angles// /_}"
	local OUTDIR="${BASE_DIR}/Buffer/${batchdir}"
	local data="box_fill_${fill}_SI_100um_${ori}.data"
	local simdir="erosion_bed_SI_ff_${fill}_np_${NP}_vel_${vel}_angle_${angle}_move_${xm}_${ym}_0"

	mkdir -p "$OUTDIR"
	cd "$OUTDIR"

	if [ -f "$simdir/post-run_final.data" ]; then
		echo "OK   (ya completa)   $simdir"
		return 0
	fi

	mkdir -p "$simdir"
	exec 9>"${simdir}/.lock"
	if ! flock -n 9; then
		echo "SKIP (en proceso)   $simdir"
		exec 9>&-
		return 0
	fi

	# Bajo el lock: si apareció el final mientras tanto, ya está
	if [ -f "$simdir/post-run_final.data" ]; then
		echo "OK   (ya completa)   $simdir"
		exec 9>&-
		return 0
	fi

	if [ -n "$(ls -A "$simdir" 2>/dev/null)" ]; then
		echo "RE-RUN (incompleta) $simdir"
		rm -rf "$simdir"/*   # limpia restos del corte (el .lock se conserva)
	else
		echo "RUN   $simdir"
	fi

	cd "$simdir"
	cp "${BASE_DIR}/${INPUT}" .

	echo "  ${batchdir}/${simdir}  angle=${angle} xmove=${xm}e-6 ymove=${ym}e-6 v=${vel} nsteps=${nsteps}"
	if ! $MPIRUN -in "$INPUT" \
		-v data "${BASE_DIR}/datas/${data}" \
		-v np "$NP" \
		-v xrotate "$XROTATE" \
		-v yrotate "$YROTATE" \
		-v zrotate "$ZROTATE" \
		-v angle "$angle" \
		-v xmove "${xm}e-6" \
		-v ymove "${ym}e-6" \
		-v zmove "0e-6" \
		-v velocity "$vel" \
		-v nsteps "$nsteps" \
		-v ndumps "$NDUMPS" \
		-v nthermo "$NTHERMO" \
		-v timestep "$TIMESTEP" \
		-v young_modulus "$YOUNG_MODULUS" \
		-v poisson "$POISSON" \
		-v surface_energy "$SURFACE_ENERGY" \
		-v density_proj "$DENSITY" \
		-v guardar_restart "$GUARDAR_RESTART" \
		-v proc_x "$PROC_X" \
		-v proc_y "$PROC_Y" \
		-v proc_z "$PROC_Z" \
		> salida.log 2>&1; then
		echo "ERROR  $simdir  (ver salida.log)" >&2
		cd "$OUTDIR"
		exec 9>&-
		return 1
	fi
	cd "$OUTDIR"
	exec 9>&-
	rm -f "${simdir}/.lock"
	return 0
}
export -f run_one_sim

# Variables que el worker (subshell de xargs) necesita
export MPIRUN BASE_DIR INPUT NP \
	XROTATE YROTATE ZROTATE NDUMPS NTHERMO TIMESTEP \
	YOUNG_MODULUS POISSON SURFACE_ENERGY DENSITY \
	PROC_X PROC_Y PROC_Z GUARDAR_RESTART \
	Fills Vels Angles Moves

# ---------------------------------------------------------------------------
# Lista de sims a correr para este job.
# Si enqueue_sims.py partió el (exp, ff) en chunks, usa la lista exacta
# (configs/sims_<JOBID>.tsv). Si no (corrida manual), la reconstruye desde
# el config: orientación → fill → vel → angle → moves.
# Formato de cada línea: orientation fill vel angle xm ym nsteps
# ---------------------------------------------------------------------------
SIMS=()
SIMS_FILE="${SCRIPT_DIR}/configs/sims_${JOBID}.tsv"
if [ -f "$SIMS_FILE" ]; then
	while IFS= read -r line; do
		[ -n "$line" ] && SIMS+=("$line")
	done < "$SIMS_FILE"
else
	COUNTER=0
	for orientation in "${ORIENTATIONS[@]}"
	do
		for fill in "${FILLS[@]}"
		do
			DATA="box_fill_${fill}_SI_100um_${orientation}.data"
			if [ ! -f "${BASE_DIR}/datas/$DATA" ]; then
				echo "ERROR: no existe ${BASE_DIR}/datas/$DATA" >&2
				exit 1
			fi
			for vel in "${VELS[@]}"
			do
				nsteps="${NSTEPS[$COUNTER]:-${NSTEPS[0]}}"
				len=${#XMOVE[@]}
				for angle in "${ANGLES[@]}"
				do
					for (( move=0; move<len; move++ )); do
						SIMS+=( "${orientation} ${fill} ${vel} ${angle} ${XMOVE[$move]} ${YMOVE[$move]} ${nsteps}" )
					done
				done
				COUNTER=$((COUNTER + 1))
			done
			COUNTER=0
		done
	done
fi

echo "=== ${#SIMS[@]} sims pendientes para el job ${JOBID}, P=${P} ==="
if [ "${#SIMS[@]}" -gt 0 ]; then
	# Los fallos individuales no cortan el job; las sims incompletas
	# se re-envían frescas en la próxima pasada de enqueue_sims.py.
	printf '%s\n' "${SIMS[@]}" | xargs -P "$P" -n 7 bash -c 'run_one_sim "$@"' _ || true
fi
