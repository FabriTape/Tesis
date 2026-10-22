# Tesis — Análisis de erosión en medios granulares

Herramientas en **Python** para el análisis de simulaciones de **erosión en medios granulares** hechas con **LAMMPS**. El proyecto cubre todo el ciclo de trabajo:

1. **Simular**: lanzamiento de simulaciones LAMMPS de impacto de proyectil sobre una cama de partículas de sílice (modelo JKR), con envío a cluster vía Slurm.
2. **Analizar**: pipeline **CARBOXI** que detecta la onda de choque, filtra las partículas erosionadas y calcula su coordinación y tamaño de cluster con **OVITO**.
3. **Comparar**: evaluación de distintas estrategias de filtrado de erosión (`comparar_filtros`).
4. **Estadística**: intervalos de confianza, distribuciones de clusters y figuras para la tesis (`cluster/`).

El nombre de las versiones sigue los dígitos de π: **3 → 3.1 → 3.14 → 3.141** … cada versión agrega un dígito de precisión (y un hito del proyecto). Ver [Releases](#lanzamientos).

## Estructura del repositorio

```
Tesis/
├── README.md                        # Esta guía
├── LICENSE                          # Licencia MIT
├── Documentos/                      # Material de la tesis
│   ├── linea_temporal.tex           #   Cronología de avances del proyecto
│   ├── errores_timeline.tex         #   Registro de errores y fixes
│   ├── Imagenes/                    #   Figuras comparativas (filtros con/sin tracking)
│   └── resultados/                  #   CSVs de resultados de comparación de filtros
└── Programas/                       # Todo el código
    ├── CARBOXI/                     # ─── P A S O  2 · ANÁLISIS ────────────────
    │   │                            # Pipeline de análisis de erosión (paquete instalable)
    │   ├── __main__.py              #   `python -m CARBOXI [ruta]` (punto de entrada)
    │   ├── main_pipeline.py         #   Orquesta el pipeline: onda + OVITO + guardado
    │   ├── config_and_discovery.py  #   Config desde config.toml + descubre simulaciones
    │   ├── config.toml              #   Parámetros: umbrales, material JKR, rutas
    │   ├── wave_analysis.py         #   Detecta la onda de choque (fase 1)
    │   ├── erosion_analysis.py      #   Coordinación y clusters con OVITO (fase 2)
    │   ├── storage_and_viz.py       #   Lee dumps, guarda parquet, gráficos SVG
    │   ├── trajectory_converter.py  #   parquet → dump/xyz para OVITO
    │   ├── merge_all.py             #   Junta parquet de muchas sims en un CSV
    │   └── pyproject.toml           #   Empaquetado (pip install ./CARBOXI)
    │
    ├── cluster/                     # ─── P A S O  1 · SIMULACIÓN ──────────────
    │   ├── scripts/                 #   Scripts para correr sims en el cluster
    │   │   ├── config.toml          #     Config central (barrido, material, filtros)
    │   │   ├── read_config.py       #     TOML → variables de shell (config_vars.sh)
    │   │   ├── encolar_sims.sh      #     Envía un job a Slurm por lote
    │   │   ├── enqueue_sims.py      #     Controlador idempotente de sims
    │   │   ├── run_erosion_bed_sims_lammps_move_np3.sh  # Lanzador LAMMPS con barrido
    │   │   ├── slurm_submit_JKR.sh  #     Definición del job (nodos, partición)
    │   │   └── collision_erosion_bed.input              # Input LAMMPS (JKR)
    │   ├── datas/                   #   Cajas de partículas generadas (.data)
    │   ├── datos/                   #   Resultados agregados (9 y 49 movimientos)
    │   │   ├── procesados/          #     CSVs filtrados reutilizables (df_dist, df_cluster, …)
    │   │   └── 9_mov/, 49_mov/      #     Ajustes, métricas y tablas LaTeX por set
    │   ├── figuras/                 #   PNGs de la tesis (9_mov/ y 49_mov/)
    │   ├── generador_figuras.py     #   Genera TODAS las figuras de la tesis (unifica 9 y 49, polars)
    │   └── verificar_polars.py      #   Verifica la equivalencia polars vs pandas (vs legacy/)
    │
    ├── comparar_filtros/            # ─── P A S O  3 · VALIDACIÓN ──────────────
    │   ├── main.py                  #   Ejecuta la comparación de filtros
    │   ├── filtros.py               #   6 filtros: F1I/F1C, F2I/F2C, F3I/F3C
    │   ├── lammps_reader.py         #   Lectura de dumps con polars
    │   ├── metricas.py              #   Métricas por step con OVITO
    │   ├── plots.py                 #   Gráficos comparativos
    │   ├── solo_graficos.py         #   Regenera gráficos desde CSV ya calculados
    │   └── run_all.py               #   Corre las variaciones de margen
    │
    ├── scripts_de_comparacion/      # Verificación antes/después del merge F2C
    │   ├── comparar_versiones.py    #   Compara parquet de dos versiones (por clave y frames comunes)
    │   ├── comparar_y_mergear.sh    #   Regenera el merge del "después" y ejecuta la comparación
    │   ├── lanzar_pre_merge.sh      #   Corre el baseline pre-merge (paquete viejo) sobre 9 o 25 movimientos
    │   ├── limpieza_fase1.sh        #   Elimina archivos obsoletos tras la reconciliación
    │   ├── reubicar_9moves.py       #   Reubica y deduplica los parquet de 9 movimientos
    │   ├── suma_tiempos.py          #   Suma horas y core-horas desde nas/
    │   └── analisis_completo.py     #   Suma horas y core-horas en nas/, sims_emmanuel/, SIMULACIONES/Buffer/
    │
    ├── Ideas_de_implementacion/     # Prototipos y experimentos en curso
    │   ├── fabric_tensor*.py        #   Tensor de fábrica (v1…v5) con OVITO
    │   ├── crater_analisis.py       #   Análisis de cráteres
    │   ├── aggregate_parquets.py    #   Agrega parquet de un filtro
    │   ├── analisis_ocupacion.py    #   Ocupación del área XY por altura
    │   ├── bayesian_sandbox.py      #   Modelos bayesianos (arviz/pymc)
    │   ├── censored.py              #   Datos censurados (pymc)
    │   └── plot_fabric.py           #   Grafica el tensor de fábrica
    │
    ├── legacy/                      # Scripts obsoletos o de prueba (referencia)
    │   ├── analisis_Tesis.ipynb     #   Notebook de análisis original
    │   ├── pdf_quitadas.py          #   Figuras removidas del pipeline principal
    │   ├── analyze.py               #   Detección de formato y pipeline automático
    │   ├── CARBOXIMINI/             #   Versión previa del pipeline de clusters
    │   ├── solid/                   #   Módulos SOLID de un refactor descartado
    │   ├── final/                   #   Últimos scripts de visualización de clusters
    │   ├── prueba_*.py              #   Scripts de prueba documentados
    │   ├── nuevo_plot_erosion/      #   Prototipo de figuras de balance de momento (removidas del pipeline)
    │   ├── resume_controller.py     #   Retomar sims incompletas desde restart
    │   ├── slurm_resume.sh          #   Job Slurm para reanudar simulaciones
    │   ├── erosion.py               #   Figuras de la tesis, set de 9 (previo a generador_figuras.py)
    │   ├── erosion_49.py            #   Figuras de la tesis, set de 49 (previo a generador_figuras.py)
    │   └── ANALISIS/                #   Análisis estadístico previo (pandas): main.py, CI, heatmaps
    │
    └── resultados/                  # Salidas de CARBOXI (parquet + SVG)
```

## Cómo empezar

### 1. Análisis de simulaciones (CARBOXI)

```bash
# Instalar el paquete (dependencias: polars, ovito, matplotlib, numpy)
pip install ./Programas/CARBOXI

# Procesar una carpeta de simulaciones LAMMPS
python -m CARBOXI /ruta/a/simulaciones

# Sin argumento: abre un navegador interactivo para elegir la carpeta
python -m CARBOXI

# Convertir trayectorias a formato OVITO
python -m CARBOXI.trajectory_converter resultados.parquet --format dump
```

`CARBOXI` detecta automáticamente la onda de choque, filtra las partículas
erosionadas (criterio **F3C** por defecto) y guarda resultados en `resultados/`
como `.parquet` y gráficos SVG.

### 2. Simulación en el cluster (LAMMPS + Slurm)

Todo el barrido de parámetros se configura en `Programas/cluster/scripts/config.toml`
(única fuente de verdad). Desde esa carpeta:

```bash
# Enviar un lote de simulaciones a Slurm
./encolar_sims.sh

# Alternativa controlada por Python (idempotente, evita duplicados)
python enqueue_sims.py
```

### 3. Figuras de la tesis

```bash
# Figuras de la tesis a partir de los CSV mergeados (script unificado, polars)
cd Programas/cluster
python generador_figuras.py            # procesa ambos sets y genera todas las figuras
python generador_figuras.py 9_mov      # solo el set de 9 repeticiones
python generador_figuras.py 49_mov     # solo el set de 49 repeticiones
python generador_figuras.py 9_mov 49_mov --forzar  # regenera ignorando el caché

# Chequeo de equivalencia con las versiones originales (pandas, en legacy/)
python verificar_polars.py
```

`generador_figuras.py` reemplaza a `erosion.py` y `erosion_49.py` (ahora en
`legacy/`): procesa los datos con **polars**, guarda los resultados intermedios
en `datos/procesados/` (se reutilizan si no se fuerza la recarga) y escribe las
figuras en `figuras/9_mov/` y `figuras/49_mov/`.

### 4. Verificación de la reconciliación (antes/después del merge F2C)

`Programas/scripts_de_comparacion/` reúne los scripts usados para validar que el
merge del filtro **F2C** (coordinación por capas) no altera los resultados
físicos de las simulaciones:

- `lanzar_pre_merge.sh`: corre el paquete **pre-merge** (snapshot viejo) sobre un
  set de simulaciones (`9_moves` o `25_moves`) para generar el baseline "antes".
- `comparar_versiones.py`: compara dos versiones de resultados parquet,
  indexando por clave de simulación y contrastando solo los frames comunes.
  Clasifica las diferencias como esperadas (columnas `rot_*`, `umbral_*`) o
  inesperadas (cambios de valores o de filas).
- `comparar_y_mergear.sh`: orquesta lo anterior; regenera el merge del estado
  "después" (excluyendo los `_ONDA_`), verifica duplicados y ejecuta la
  comparación.
- `limpieza_fase1.sh` y `reubicar_9moves.py`: limpieza y reubicación de los
  parquet de 9 movimientos tras la reconciliación.

Las salidas de estos scripts (`.txt`/`.csv` de comparación) no se versionan.

### 5. Cálculo de tiempos de simulación

`Programas/scripts_de_comparacion/` incluye además dos scripts para estimar el tiempo
total empleado en las simulaciones a partir de las marcas temporales (`mtime`) de los
archivos:

- **`suma_tiempos.py`** — Escanea `nas/` (excluyendo `sims_25_moves`). Para cada carpeta
  de simulación busca el `output` final (`output.N.gz` más reciente, o
  `output_trayectoria.dump`). Calcula la duración entre `pre-run_with_projectile.data`
  y ese archivo. Obtiene el número de núcleos desde `log.lammps` (grilla MPI) o del
  nombre de la carpeta (`cores_N`). Imprime el detalle por simulación, la suma de
  horas, el total en *core-horas* y el equivalente con 32, 64 o 128 núcleos.
- **`analisis_completo.py`** — Versión general que recorre `nas/`, `sims_emmanuel/`
  y `SIMULACIONES/Buffer/` con sus listas de exclusión. Usa el mismo criterio para
  duración y núcleos, y además extrae el coeficiente de fricción (`ff = 0.15` o
  `0.35`) del nombre de la subcarpeta. Útil para obtener un total global entre
  distintos discos/directorios.

## Lanzamientos

| Versión | Fecha | Qué incluye |
|---------|-------|-------------|
| [v3](https://github.com/FabriTape/Tesis/releases/tag/v3) | Dic 2025 | Lectura de datos con **parquet** en lugar de CSV (3 s → ~0.6 s). |
| [v3.1](https://github.com/FabriTape/Tesis/releases/tag/v3.1) | Dic 2025 | Módulo **Lector** con **polars**, paralelización y cálculos con **OVITO**. |
| [v3.14](https://github.com/FabriTape/Tesis/releases/tag/v3.14) | Mar 2026 | **CARBOXI**: pipeline automático (máquina de estados, config externalizada, detección de onda). |
| [v3.141](https://github.com/FabriTape/Tesis/releases/tag/v3.141) | Ago 2026 | Estado actual: scripts de simulación unificados, comparación de filtros, análisis estadístico. |

## Ramas

- **`main`**: rama estable. Apunta al estado actual del trabajo.
- **`desarrollo`**: rama de desarrollo activo (Se está trabajando en el desarrollo de una interfaz gráfica que centralice el trabajo. Este programa que se encuentra en desarrollo y revisión, se llamará LENGA+).

Los experimentos descartados quedan archivados como tags `archive/*` para
conservar su historial sin ensuciar las ramas.

## Licencia

MIT — ver [LICENSE](LICENSE).
