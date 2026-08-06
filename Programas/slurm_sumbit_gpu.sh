#!/bin/bash
#SBATCH -J analisis_CARBOXI_MULTI
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -o ./logs/job_exec_%j.out
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu2

## 1. Configuración de entorno
cd /home/fbazzoni

# Activamos el entorno correctamente
source /home/fbazzoni/.venv_gpu/bin/activate
export QT_QPA_PLATFORM=offscreen
which python
python --version
## 2. Crear carpeta de logs si no existe
mkdir -p /home/fbazzoni/logs

## 3. Loop sobre las carpetas con control de concurrencia
BASE_DIR="/home/fbazzoni"
MAX_JOBS=8

# Verificación de diagnóstico (revisá el .out para ver qué imprime esto)
echo "Revisando si el directorio NAS existe en gpu2:"
ls $BASE_DIR/scratch/sims_9_moves  | head -n 5

for dir in $BASE_DIR/scratch/sims_9_moves/*/*; do
    if [ -d "$dir" ]; then
        # Esperar si ya hay MAX_JOBS procesos en ejecución
        while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 10
        done
        
        echo "Lanzando análisis para: $dir"
        
        # Al estar el venv activado, 'python' ya apunta al correcto
        python -m CARBOXI "$dir" > "logs/9_output_$(basename "$dir").log" 2>&1 &
        
        sleep 30
    fi
done

## 4. Esperar a que terminen
wait

echo "Todos los análisis han finalizado."
python -m CARBOXI.merge_all $BASE_DIR/resultados
