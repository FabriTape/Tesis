"""Parámetros de ejecución de simulaciones LAMMPS encluster.

Carga y valida los parámetros de simulación, sweep y material desde
archivos TOML o configuración por defecto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class SimulationParams:
    """Parámetros de configuración de LAMMPS.

    Attributes:
        np: Número de partículas del proyectil (1 o 3).
        ndumps: Frecuencia de guardado de dump [timesteps].
        nthermo: Frecuencia de impresión thermo [timesteps].
        cores: Núcleos MPI asignados.
        processors: Distribución de processors [Px, Py, Pz].
        exe: Nombre del ejecutable LAMMPS.
        timestep: Paso temporal [s].
    """

    np: int
    ndumps: int
    nthermo: int
    cores: int
    processors: tuple[int, int, int]
    exe: str
    timestep: float


@dataclass(frozen=True)
class SweepParams:
    """Parámetros del barrido de parámetros.

    Attributes:
        fills: Lista de fracciones de llenar.
        vels: Lista de velocidades de impacto [m/s].
        angles: Lista de ángulos de rotación [deg].
        moves: Lista de desplazamientos en la grilla.
        total_steps: Lista de pasos por combinación.
        grain_radius_um: Radio del grano [μm].
        xrotate: Rotación en X (0 o 1).
        yrotate: Rotación en Y (0 o 1).
        zrotate: Rotación en Z (0 o 1).
    """

    fills: tuple[float, ...]
    vels: tuple[int, ...]
    angles: tuple[int, ...]
    moves: tuple[int, ...]
    total_steps: tuple[int, ...]
    grain_radius_um: float
    xrotate: int
    yrotate: int
    zrotate: int


@dataclass(frozen=True)
class MaterialParams:
    """Parámetros del material (sílice).

    Attributes:
        young_modulus: Módulo de Young [Pa].
        poisson: Coeficiente de Poisson.
        surface_energy: Energía superficial JKR [J/m²].
        density: Densidad del proyectil [kg/m³].
    """

    young_modulus: float
    poisson: float
    surface_energy: float
    density: float


@dataclass(frozen=True)
class ClusterConfig:
    """Configuración del cluster SLURM.

    Attributes:
        nodes: Lista de nodos disponibles.
        partition: Nombre de la partición.
    """

    nodes: tuple[str, ...]
    partition: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuración completa para ejecución de simulaciones.

    Attributes:
        simulation: Parámetros de LAMMPS.
        sweep: Parámetros del barrido.
        material: Parámetros del material.
        cluster: Configuración del cluster.
    """

    simulation: SimulationParams
    sweep: SweepParams
    material: MaterialParams
    cluster: ClusterConfig
    config_path: Path = field(default_factory=lambda: Path("."))


def load_toml(path: Path) -> dict[str, Any]:
    """Carga un archivo TOML.

    Args:
        path: Ruta al archivo TOML.

    Returns:
        Diccionario con el contenido parseado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    with path.open("rb") as f:
        return tomllib.load(f)


def parse_simulation(raw: dict[str, Any]) -> SimulationParams:
    """Parsea la sección [simulation] del TOML.

    Args:
        raw: Diccionario con las claves de la sección simulation.

    Returns:
        Instancia de SimulationParams.
    """
    sim = raw["simulation"]
    return SimulationParams(
        np=int(sim["np"]),
        ndumps=int(sim["ndumps"]),
        nthermo=int(sim["nthermo"]),
        cores=int(sim["cores"]),
        processors=tuple(int(x) for x in sim["processors"]),
        exe=str(sim["exe"]),
        timestep=float(sim["timestep"]),
    )


def parse_sweep(raw: dict[str, Any]) -> SweepParams:
    """Parsea la sección [sweep] del TOML.

    Args:
        raw: Diccionario con las claves de la sección sweep.

    Returns:
        Instancia de SweepParams.
    """
    sw = raw["sweep"]
    return SweepParams(
        fills=tuple(float(x) for x in sw["fills"]),
        vels=tuple(int(x) for x in sw["vels"]),
        angles=tuple(int(x) for x in sw["angles"]),
        moves=tuple(int(x) for x in sw["moves"]),
        total_steps=tuple(int(x) for x in sw["total_steps"]),
        grain_radius_um=float(sw["grain_radius_um"]),
        xrotate=int(sw["xrotate"]),
        yrotate=int(sw["yrotate"]),
        zrotate=int(sw["zrotate"]),
    )


def parse_material(raw: dict[str, Any]) -> MaterialParams:
    """Parsea la sección [material] del TOML.

    Args:
        raw: Diccionario con las claves de la sección material.

    Returns:
        Instancia de MaterialParams.
    """
    mat = raw["material"]
    return MaterialParams(
        young_modulus=float(mat["young_modulus"]),
        poisson=float(mat["poisson"]),
        surface_energy=float(mat["surface_energy"]),
        density=float(mat["density"]),
    )


def parse_cluster(raw: dict[str, Any]) -> ClusterConfig:
    """Parsea la sección [cluster] del TOML.

    Args:
        raw: Diccionario con las claves de la sección cluster.

    Returns:
        Instancia de ClusterConfig.
    """
    if "cluster" not in raw:
        return ClusterConfig(
            nodes=("toko01", "toko02", "toko03", "toko04", "toko05",
                  "toko06", "toko07", "toko08", "toko09"),
            partition="XL",
        )
    cl = raw["cluster"]
    return ClusterConfig(
        nodes=tuple(str(x) for x in cl["nodes"]),
        partition=str(cl["partition"]),
    )


def load_execution_config(path: Path | None = None) -> ExecutionConfig:
    """Carga y valida la configuración de ejecución desde un TOML.

    Args:
        path: Ruta al archivo de configuración. Si None, usa config.toml embebido.

    Returns:
        Instancia completa de ExecutionConfig.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        KeyError: Si faltan secciones obligatorias.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "cluster" / "config.toml"
        if not path.exists():
            path = Path(__file__).parent / "templates" / "config.toml"

    raw = load_toml(path)
    return ExecutionConfig(
        simulation=parse_simulation(raw),
        sweep=parse_sweep(raw),
        material=parse_material(raw),
        cluster=parse_cluster(raw),
        config_path=path,
    )


def validate_config(cfg: ExecutionConfig) -> list[str]:
    """Valida la configuración y retorna errores encontrados.

    Args:
        cfg: Configuración a validar.

    Returns:
        Lista de mensajes de error. Vacía si es válida.
    """
    errors: list[str] = []

    if cfg.simulation.np not in (1, 3):
        errors.append(f"np debe ser 1 o 3, recibido {cfg.simulation.np}")

    if cfg.simulation.cores not in (8, 16, 32):
        errors.append(f"cores debe ser 8, 16 o 32, recibido {cfg.simulation.cores}")

    proc_x, proc_y, proc_z = cfg.simulation.processors
    if proc_x * proc_y * proc_z != cfg.simulation.cores:
        errors.append(
            f"processors {cfg.simulation.processors} "
            f"no produce {cfg.simulation.cores} cores"
        )

    if cfg.sweep.fills and not all(0.0 < f < 1.0 for f in cfg.sweep.fills):
        errors.append("fills debe estar entre 0 y 1")

    if cfg.sweep.vels and any(v > 0 for v in cfg.sweep.vels):
        errors.append("vels debe ser negativo (velocidad hacia abajo)")

    if cfg.sweep.grain_radius_um <= 0:
        errors.append("grain_radius_um debe ser positivo")

    return errors
