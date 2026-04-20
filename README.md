# BPM Toolkit

Biblioteca unificada para simulaciones LAMMPS de erosión granular.

## Requisitos

- Python 3.11+
- LAMMPS con soporte MPI
- Acceso a cluster SLURM (nodos toko01-toko09)

## Instalación

```bash
# Clonar o copiar a ~/BPM_toolkit
cd ~/BPM_toolkit
pip install -e .
```

## Estructura

```
BPM_toolkit/
├── config.toml          # Configuración única
├── execution/          # CLI de ejecución
├── stats/              # Análisis estadístico
├── viz/                # Visualizaciones
├── io/                 # Merge y I/O
└── topology/          # Análisis topológico (OVITO)
```

## Flujo de Trabajo

### 1. Configuración

Editar `config.toml` con los parámetros deseados:

```toml
[simulation]
np = 1                  # Proyectil: 1 o 3
cores = 8                # Núcleos por job

[sweep]
fills = [0.15, 0.35]    # Fracciones de llenado
vels = [-50, -25, -10, -5]  # Velocidades
angles = [0, 90]        # Ángulos (90 solo para NP3)
moves = [-3, -2, -1, 0, 1, 2, 3]  # Grilla de位移amientos

[cluster]
partition = "XL"
```

### 2. Dry-Run (Probar sin ejecutar)

```bash
python3 -m BPM_toolkit.execution dry-run --config config.toml
```

### 3. Ejecutar Simulaciones

```bash
python3 -m BPM_toolkit.execution run --config config.toml
```

### 4. Verificar Estado

```bash
python3 -m BPM_toolkit.execution status 123456 123457
```

### 5. Verificar Integridad (Post-Simulación)

```bash
python3 -m BPM_toolkit.execution check --results ~/resultados/resultados_NP_...
```

### 6. Analizar Resultados

```bash
python3 -m BPM_toolkit.execution analyze --results ~/resultados/resultados_NP_...
```

## Comandos CLI

| Comando | Descripción |
|----------|-------------|
| `dry-run` | Mostrar jobs sin ejecutar |
| `run` | Ejecutar simulaciones en SLURM |
| `status` | Ver estado de jobs |
| `check` | Verificar integridad post-simulación |
| `analyze` | Merge + stats + visualizaciones |

## Rutas de Salida

```
~/resultados/resultados_NP_{np}_vels_{...}_ff_{...}_movs_{...}/
├── config_{job_id}.toml      # Copia config para reproducibilidad
├── slurm_{job_id}.out       # Log SLURM
├── erosion_bed_SI_.../      # Directorios de simulación
│   └── output_trayectoria.dump
├── resultados/             # Archivos parquet
├── merge_resultados.csv   # CSV合并
└── figuras/               # Gráficos generados
```

## Parámetros del Config

| Sección | Parámetros |
|----------|------------|
| `simulation` | np, cores, processors, ndumps, nthermo, exe, timestep |
| `sweep` | fills, vels, angles, moves, total_steps, grain_radius_um |
| `material` | young_modulus, poisson, surface_energy, density |
| `cluster` | nodes, partition |

## Nombres de Jobs

Formato legacy: `erosion_bed_SI_ff_{fill}_np_{np}_vel_{vel}_angle_{angle}_move_{x}_{y}_{z}`

Ejemplos:
- `erosion_bed_SI_ff_0.15_np_1_vel_-50_angle_0_move_0_0_0`
- `erosion_bed_SI_ff_0.35_np_3_vel_-10_angle_90_move_2.28_2.28_0`

## Troubleshoot

### Job no inicia
- Verificar que nodos estén disponibles: `sinfo`
- Verificar partición: `squeue -p XL`

### Simulación falla
- Revisar log: `cat slurm_{job_id}.out`
- Verificar ejecutable: `which lmp_g++_openmpi-4.0.2_stableSep2021`

### Merge falla
- Verificar que existen parquet: `ls resultados/*.parquet`
- Verificar permisos: `ls -la resultados/`

## Licencia

MIT