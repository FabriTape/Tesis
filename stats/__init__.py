"""Módulo de estadísticas para análisis de erosión y clusters.

Contiene:
- clusters: Cálculos de tamaño y distribución de clusters.
- confidence: Intervalos de confianza con métodos robustos.
- kinematics: Datos cinemáticos para heatmaps.
"""

from .clusters import (
    compute_cluster_frequency,
    compute_cluster_size_stats,
    compute_cluster_sizes,
    compute_erosion_counts_with_zeros,
    compute_unique_ids_by_velocity,
    compute_unique_ids_raw,
    compute_unique_ids_raw_with_positions,
    filter_erosion,
)
from .confidence import calc_ic, run_confidence_intervals
from .kinematics import compute_heatmap_data, get_grid_pairs

__all__ = [
    "filter_erosion",
    "compute_unique_ids_by_velocity",
    "compute_unique_ids_raw_with_positions",
    "compute_erosion_counts_with_zeros",
    "compute_unique_ids_raw",
    "compute_cluster_frequency",
    "compute_cluster_sizes",
    "compute_cluster_size_stats",
    "calc_ic",
    "run_confidence_intervals",
    "get_grid_pairs",
    "compute_heatmap_data",
]