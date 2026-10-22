#!/bin/bash
# Limpieza de scripts, logs y archivos temporales obsoletos.
# Uso:  DRY_RUN=1 ./limpieza_fase1.sh   (solo lista)
#       ./limpieza_fase1.sh             (borra)
#
# NO toca: lanzar_analisis_9moves.sh, slurm_sumbit*.sh, ~/CARBOXI,
#          ~/resultados (dataset de junio), test_reconciliacion/9_moves_all/resultados,
#          logs/out_*.log, logs/job_*.out, /tmp/procesados_9moves.txt

set -uo pipefail
HOME_DIR="/home/fbazzoni"
DRY="${DRY_RUN:-0}"
DELETED=0
KEPT=0

remove() {
  local path="$1"
  if [ ! -e "$path" ]; then
    return 0
  fi
  if [ "$DRY" = "1" ]; then
    echo "  [DRY] borraría: $path"
  else
    rm -rf -- "$path" && echo "  [OK] borrado: $path"
  fi
  DELETED=$((DELETED + 1))
}

keep() {
  echo "  [KEEP] $1"
  KEPT=$((KEPT + 1))
}

echo "=== 1) Scripts de prueba obsoletos (home) ==="
for f in run_9moves_all.sh run_9moves_missing.sh run_9moves_bruteforce.sh \
         do_all_9.sh do_rest_9.sh \
         run_9moves_all.sh.out run_9moves_missing.sh.out run_9moves_bruteforce.sh.out \
         do_all_9.sh.out do_rest_9.sh.out \
         test_env_sb.slurm test_env_sb2.slurm test_py311_direct.slurm test_venv_repair.slurm; do
  remove "$HOME_DIR/$f"
done

echo "=== 2) Temporales de /tmp (conservando procesados_9moves.txt) ==="
for f in fix_9moves.sh fix_9moves.sh.out run_missing.sh run_missing.sh.out \
         ok.txt ok_9.txt processed.txt processed_fix.txt done_9.txt done_9_rest.txt \
         missing_9.txt L_out.txt L_output.txt A_home.txt B_test.txt \
         S9.txt S9b.txt S25.txt EXTRA_home.txt; do
  remove "/tmp/$f"
done
keep "/tmp/procesados_9moves.txt"

echo "=== 3) Smoke test local ==="
remove "$HOME_DIR/test_reconciliacion/local_smoke"

echo "=== 4) Logs obsoletos de 9_moves_all/logs (conservando out_*.log) ==="
LOGS="$HOME_DIR/test_reconciliacion/9_moves_all/logs"
if [ -d "$LOGS" ]; then
  while IFS= read -r -d '' f; do remove "$f"; done < <(find "$LOGS" -maxdepth 1 -type f ! -name "out_*" -print0)
  echo "  conservados: $(find "$LOGS" -maxdepth 1 -type f -name 'out_*' | wc -l) archivos out_*.log"
fi

echo "=== 5) ~/logs: solo los de pruebas de entorno descartadas ==="
for f in test_env_99237.out test_env2_99238.out repair_venv_99239.out test_py311d_99240.out; do
  remove "$HOME_DIR/logs/$f"
done
while IFS= read -r -d '' f; do remove "$f"; done < <(find "$HOME_DIR/logs" -maxdepth 1 -type f -name "9_output_*.log" -print0)
echo "  conservados: $(find "$HOME_DIR/logs" -maxdepth 1 -type f -name 'job_*.out' | wc -l) archivos job_*.out (historial SLURM)"

echo
echo "=== Resumen ==="
echo "  elementos procesados para borrar: $DELETED"
echo "  elementos conservados explícitamente: $KEPT"
[ "$DRY" = "1" ] && echo "  (DRY RUN: no se borró nada)"
exit 0
