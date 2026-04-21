"""Módulo de constantes y configuración para análisis de datos.

Define los parámetros globales y rutas utilizadas en el análisis.
"""

from pathlib import Path

# Rutas
CARBOXI_DIR = Path("CARBOXI")
RESULTADOS_DIR = Path("resultados")
CSV_PATH = RESULTADOS_DIR / "merge_resultados.csv"
SALIDA_DIR = RESULTADOS_DIR / "figuras_pdf"

# Parámetros de simulación (se usa I_VALUE dinámico en main.py, mantener para otros módulos)
I_VALUE = 80
THRESHOLD_Z = 1e-4
THRESHOLD_Z = 1e-4

# Valores de fricción a analizar
FF_VALUES = [0.15, 0.35]

# Filtros
EXCLUIR_EROSION = "NOT (type = 2 AND (z < 1e-4 OR vz < 0))"
FILTRO_PENETRACION = "type = 2 AND (z < 1e-4 OR vz < 0)"

# Colores para gráficos
COLORS = {
    "primary": "#3b82f6",
    "power_law": "#e74c3c",
    "truncated": "#2ecc71",
    "exponential": "#f39c12",
    "gamma": "#9b59b6",
    "secondary": "#3498db",
}
