"""
BPM_toolkit - Biblioteca de análisis de simulaciones LAMMPS.

Módulos:
- config: Carga de configuración TOML y descubrimiento de simulaciones.
- loader: Lectura de trayectorias LAMMPS (OVITO + Polars).
- filters: Filtros cinemáticos para detección de erosión.
- topology: Análisis topológico (clusters, onda de choque).

Física:
- Análisis de erosión de partículas en lechos de sílice.
- Detección de ondas de choque macroscópicas.
- Tracking de partículas eyectadas entre timesteps.
"""

__version__: str = "0.1.0"