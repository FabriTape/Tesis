#!/bin/bash
# Compara una corrida "antes" (pre-merge) contra una "después" (post-merge),
# verifica que no haya filas duplicadas en el merge del "después" y genera el CSV.
#
# Uso:
#   ./comparar_y_mergear.sh <dir_antes> <dir_despues>
#
# Ejemplo 9_moves (control junio vs hoy):
#   ./comparar_y_mergear.sh resultados_final_9 \
#       test_reconciliacion/9_moves_all/resultados
#
# Ejemplo 9_moves (baseline real pre-merge vs post-merge):
#   ./comparar_y_mergear.sh test_reconciliacion/pre_merge/9_moves/resultados \
#       test_reconciliacion/9_moves_all/resultados

set -uo pipefail
export LD_LIBRARY_PATH=""

PYTHON_BIN="/home/fbazzoni/.venv_py311/bin/python"
ANTES="${1:?Uso: $0 <dir_antes> <dir_despues>}"
DESPUES="${2:?Uso: $0 <dir_antes> <dir_despues>}"

[ -d "$ANTES" ] || { echo "No existe: $ANTES"; exit 1; }
[ -d "$DESPUES" ] || { echo "No existe: $DESPUES"; exit 1; }

echo "=== 1) Regenerando merge del después (excluyendo *_ONDA_*) ==="
TMP="$(dirname "$DESPUES")/.merge_tmp/resultados"
rm -rf "$(dirname "$DESPUES")/.merge_tmp"; mkdir -p "$TMP"
find "$DESPUES" -maxdepth 1 -name "*.parquet" ! -name "*_ONDA_*" -exec cp {} "$TMP"/ \;
PYTHONPATH=/home/fbazzoni QT_QPA_PLATFORM=offscreen \
  "$PYTHON_BIN" -m CARBOXI.merge_all "$TMP" --output "$DESPUES"
rm -rf "$(dirname "$DESPUES")/.merge_tmp"

CSV="$DESPUES/merge_$(basename "$DESPUES").csv"
[ -f "$CSV" ] || CSV="$DESPUES/merge_$(basename "$TMP").csv"
echo "merge CSV: $CSV"

echo
echo "=== 2) Verificando duplicados en el merge ==="
"$PYTHON_BIN" - "$CSV" <<'EOF'
import sys, polars as pl
csv = sys.argv[1]
df = pl.read_csv(csv)
key = ["ff", "np", "vel0", "angulo", "posx", "posy", "posz"]
print(f"  filas: {df.height}, columnas: {df.width}")
print(f"  sims únicas: {df.group_by(key).len().height}")
dup = df.height - df.unique().height
print(f"  filas duplicadas exactas: {dup}")
if dup:
    print("  !! HAY DUPLICADOS")
    sys.exit(1)
print("  OK sin duplicados")
EOF

echo
echo "=== 3) Comparación antes vs después ==="
"$PYTHON_BIN" /home/fbazzoni/comparar_versiones.py "$ANTES" "$DESPUES" \
  --out "$DESPUES" --verbose
rc=$?
echo
if [ "$rc" -eq 0 ]; then
    echo ">>> RESULTADO: sin diferencias inesperadas."
elif [ "$rc" -eq 2 ]; then
    echo ">>> RESULTADO: hay diferencias INESPERADAS. Revisar el detalle."
else
    echo ">>> RESULTADO: error en la comparación (rc=$rc)."
fi
exit "$rc"
