"""Módulo de visualizaciones para análisis de erosión.

Contiene:
- basic: Gráficos boxplot, heatmap y comparaciones.
- trajectory: Visualización de trayectorias.
"""

from .basic import (
    plot_erosion_boxplot,
    plot_erosion_heatmap,
    plot_erosion_vs_angulo,
    plot_erosion_vs_energia,
    plot_erosion_vs_ff,
    plot_erosion_vs_np,
    save_all_figures_pdf,
)
from .trajectory import plot_erosion as plot_erosion_trajectory

__all__ = [
    "plot_erosion_boxplot",
    "plot_erosion_heatmap",
    "plot_erosion_vs_energia",
    "plot_erosion_vs_ff",
    "plot_erosion_vs_np",
    "plot_erosion_vs_angulo",
    "plot_erosion_trajectory",
    "save_all_figures_pdf",
]