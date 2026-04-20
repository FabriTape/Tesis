"""
Análisis topológico de partículas.

Submódulos:
- erosion: Filtro F3C y análisis de clusters.
- wave: Detección de onda de choque.
"""

from .erosion import (
    analyze_erosion,
    build_ovito_pipeline,
    run_topology_analysis,
)
from .wave import detect_wave_particles, is_critical_simulation, run_wave_scan

__all__ = [
    "build_ovito_pipeline",
    "analyze_erosion",
    "run_topology_analysis",
    "detect_wave_particles",
    "is_critical_simulation",
    "run_wave_scan",
]