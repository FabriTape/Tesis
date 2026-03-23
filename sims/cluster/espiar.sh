#!/bin/bash

# 1. Encuentra el directorio de la simulación más reciente
DIR=$(dirname $(ls -t ~/SIMULACIONES/Buffer/*/*/salida.log | head -n 1))

echo "========================================="
echo "  LOG DE LAMMPS (Últimas 15 líneas)"
echo "========================================="
tail -n 15 "$DIR/salida.log"

echo ""
echo "========================================="
echo "  VELOCIDADES (Últimos 15 pasos)"
echo "========================================="

# 2. Buscamos los últimos 15 timesteps únicos desde archivos .gz
TIMESTEPS=$(ls -t "$DIR"/output.*.gz 2>/dev/null | head -n 15 | awk -F'/' '{print $NF}' | sed 's/output\.//;s/\.gz//')

if [ -n "$TIMESTEPS" ]; then
    # 3. Hacemos el bucle calculando máximo y promedio
    for TS in $TIMESTEPS; do
        gzip -dc "$DIR/output.${TS}.gz" | awk -v ts="$TS" 'NR>9 {
            v = sqrt($6*$6 + $7*$7 + $8*$8);
            sum += v;
            count++;
            if(v>max) max=v
        } END {
            avg = (count > 0) ? sum / count : 0;
            printf "Paso %8s -> Max: %.4f m/s | Prom: %.6f m/s\n", ts, max, avg
        }'
    done
else
    echo "Esperando a que LAMMPS escriba los primeros dumps..."
fi
