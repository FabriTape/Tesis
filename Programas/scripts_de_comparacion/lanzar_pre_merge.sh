#!/bin/bash
# Lanza el análisis CARBOXI con la versión PRE-MERGE (snapshot del 2026-06-20)
# para generar el baseline "antes" de la reconciliación.
#
# Uso:
#   ./lanzar_pre_merge.sh 9_moves
#   ./lanzar_pre_merge.sh 25_moves
#
# Salida: test_reconciliacion/pre_merge/<set>/resultados/*.parquet  (sin columnas rot_*)
#
# Es resume-safe: si ya existe el parquet de una sim, la saltea.

set -uo pipefail

SET="${1:-}"
case "$SET" in
  9_moves)  SCRATCH="/home/fbazzoni/scratch/sims_9_moves" ;;
  25_moves) SCRATCH="/home/fbazzoni/scratch/sims_25_moves" ;;
  *) echo "Uso: $0 {9_moves|25_moves}"; exit 1 ;;
esac

PYTHON_BIN="/home/fbazzoni/.venv_py311/bin/python"
PRE_PKG="/home/fbazzoni/CARBOXI_PRE"
OUT_DIR="/home/fbazzoni/test_reconciliacion/pre_merge/${SET}"
MAX_JOBS="${MAX_JOBS:-1}"

# El entorno de login tiene un LD_LIBRARY_PATH que rompe el python del venv.
export LD_LIBRARY_PATH=""
export QT_QPA_PLATFORM=offscreen
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PRE_PKG"

mkdir -p "$OUT_DIR/logs"
echo "cwd: $(pwd)"
echo "paquete pre-merge: $PRE_PKG/CARBOXI"
echo "scratch: $SCRATCH"
echo "MAX_JOBS=$MAX_JOBS"
echo "raíz del proyecto (salidas): $OUT_DIR"

echo "=== diagnóstico ==="
# -P aísla sys.path[0] del CWD para que nunca se cargue el paquete canónico ~/CARBOXI.
"$PYTHON_BIN" -P -c "import CARBOXI; print('import OK:', CARBOXI.__file__)" || exit 1
"$PYTHON_BIN" -P -c "
import CARBOXI.config_and_discovery as cd
assert not hasattr(cd, 'extraer_rotacion'), 'ESTE NO ES EL PAQUETE PRE-MERGE'
print('confirmado: versión pre-merge (sin rotación)')
" || exit 1

echo "=== lanzando análisis ==="
n=0
skipped=0
for dir in "$SCRATCH"/*/erosion_bed_SI_*; do
    [ -d "$dir" ] || continue
    simname=$(basename "$dir")
    if ls "$OUT_DIR"/resultados/${simname}_TOTAL_F2C_*.parquet >/dev/null 2>&1; then
        skipped=$((skipped + 1))
        continue
    fi
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
    "$PYTHON_BIN" -P -m CARBOXI "$dir" --project-root "$OUT_DIR" > "logs/out_${simname}.log" 2>&1 &
    n=$((n + 1))
done
echo "lanzados: $n, salteados (ya existían): $skipped"
wait
echo "=== finalizado: $(date) ==="
echo "Resultados en: $OUT_DIR/resultados"
