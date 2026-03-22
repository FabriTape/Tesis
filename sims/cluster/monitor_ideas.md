# Ideas para monitor.sh — integración SLURM

## Opción A: squeue (jobs activos)

```bash
echo "========================================="
echo "  SLURM JOBS"
echo "========================================="
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

## Opción B: tail del log de SLURM (último job ejecutado)

```bash
SLURM_LOG=$(ls -t ~/SIMULACIONES/logs/job_exec_*.out 2>/dev/null | head -n 1)
if [ -n "$SLURM_LOG" ]; then
    echo "========================================="
    echo "  SLURM LOG (último job)"
    echo "========================================="
    tail -n 15 "$SLURM_LOG"
fi
```

## Opción C: Ambos (squeue + log del job más reciente)
