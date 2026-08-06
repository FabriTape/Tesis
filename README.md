# Tesis — Análisis de erosión en medios granulares

Herramientas en **Python** para el análisis de simulaciones de **erosión en medios granulares** hechas con **LAMMPS**. El proyecto cubre todo el ciclo de trabajo:

1. **Simular**: lanzamiento de simulaciones LAMMPS de impacto de proyectil sobre una cama de partículas de sílice (modelo JKR), con envío a cluster vía Slurm.
2. **Analizar**: pipeline **CARBOXI** que detecta la onda de choque, filtra las partículas erosionadas y calcula su coordinación y tamaño de cluster con **OVITO**.
3. **Comparar**: evaluación de distintas estrategias de filtrado de erosión (`comparar_filtros`).
4. **Estadística**: intervalos de confianza, distribuciones de clusters y figuras para la tesis (`cluster/`).
5. **LENGA**: plataforma con interfaz gráfica (GUI) que integra autenticación SSH, monitoreo de nodos del cluster, formulario de simulaciones, análisis CARBOXI, descargas SCP y generador de figuras.

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
    ├── LENGA/                       # ─── G U I  I N T E G R A D A ───────────────
    │   ├── lenga/                   #   Paquete de la interfaz gráfica PyQt5
    │   ├── datos/                   #   Resultados descargados
    │   ├── figuras/                 #   Figuras generadas TAL CUAL
    │   ├── run_lenga.sh             #   Script ejecutable del programa
    │   └── lenga.desktop            #   Lanzador de escritorio
    │
    ├── CARBOXI/                     # ─── P A S O  2 · ANÁLISIS ────────────────
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
    │   ├── datas/                   #   Cajas de partículas generadas (.data)
    │   ├── datos/                   #   Resultados agregados (9 y 49 movimientos)
    │   ├── figuras/                 #   PNGs de la tesis (9_mov/ y 49_mov/)
    │   └── generador_figuras.py     #   Genera TODAS las figuras de la tesis (unifica 9 y 49, polars)
    │
    └── comparar_filtros/            # ─── P A S O  3 · VALIDACIÓN ──────────────
```

## LENGA — Plataforma Unificada GUI

LENGA unifica el flujo completo de simulación, análisis y gráficos en una aplicación de escritorio PyQt5:

1. **Autenticación SSH y Monitoreo del Cluster**:
   - Inicio de sesión seguro vía SSH (correo/usuario y contraseña).
   - Monitoreo en tiempo real de nodos del cluster (`sinfo`), uso de CPU/GPU y colas (`squeue`).
   - Selección de nodos destino con validación estricta de **máximo 2 nodos**.

2. **Lanzador de Simulaciones LAMMPS**:
   - Formulario de parámetros (`fills`, `vels`, `angles`, `moves`, `orientations`, `np`, `nsteps`).
   - Generación automática de `config.toml` y scripts SLURM (`sbatch`).

3. **Análisis Automático con CARBOXI**:
   - Ejecución remota de `CARBOXI` y `python -m CARBOXI.merge_all`.

4. **Gestor de Descargas SCP**:
   - Descarga automática hacia `LENGA/datos/`.

5. **Generador y Visor Interactivo de Figuras**:
   - Renderizado de figuras **TAL CUAL** `generador_figuras.py` con lienzo Matplotlib interactivo.

## Cómo empezar

### 1. Ejecutar LENGA (GUI)

```bash
# Mediante el script ejecutable
/home/tutturuu/Documentos/Universidad/Tesis/Programas/LENGA/run_lenga.sh

# O mediante Python
cd Programas/LENGA
python3 -m lenga.main
```

### 2. Análisis de simulaciones (CARBOXI)

```bash
python -m CARBOXI /ruta/a/simulaciones
```

### 3. Figuras de la tesis

```bash
cd Programas/cluster
python generador_figuras.py
```

## Lanzamientos

| Versión | Fecha | Qué incluye |
|---------|-------|-------------|
| [v3](https://github.com/FabriTape/Tesis/releases/tag/v3) | Dic 2025 | Lectura de datos con **parquet** en lugar de CSV. |
| [v3.1](https://github.com/FabriTape/Tesis/releases/tag/v3.1) | Dic 2025 | Módulo **Lector** con **polars**, paralelización y **OVITO**. |
| [v3.14](https://github.com/FabriTape/Tesis/releases/tag/v3.14) | Mar 2026 | **CARBOXI**: pipeline automático. |
| [v3.141](https://github.com/FabriTape/Tesis/releases/tag/v3.141) | Ago 2026 | Estado actual: **LENGA GUI**, scripts de simulación unificados, gráficos TAL CUAL. |

## Licencia

MIT — ver [LICENSE](LICENSE).
