"""Planificador de jobs para ejecución en cluster SLURM.

Calcula las combinaciones fill×vel×angle y genera los nombres
de job según la convención establecida.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

from config import ExecutionConfig


@dataclass(frozen=True)
class JobSpec:
    """Especificación de un job individual.

    Attributes:
        job_id: Identificador único del job.
        fill: Fracción de llenar.
        vel: Velocidad de impacto [m/s].
        angle: Ángulo de rotación [deg].
        move_grid: Tamaño de la grilla de moves.
        xmove: Desplazamiento en X [m].
        ymove: Desplazamiento en Y [m].
        zmove: Desplazamiento en Z [m].
        nsteps: Pasos de simulación.
        sim_dir: Directorio de simulación.
    """

    job_id: str
    fill: float
    vel: int
    angle: int
    move_grid: int
    xmove: float
    ymove: float
    zmove: float
    nsteps: int
    sim_dir: str


def _make_job_id(
    np: int,
    vel: int,
    fill: float,
    angle: int,
    xmove: float,
    ymove: float,
    zmove: float,
) -> str:
    """Genera el identificador del job según la convención legacy.

    Formato: erosion_bed_SI_ff_{fill}_np_{np}_vel_{vel}_angle_{angle}_move_{x}_{y}_{z}

    Ejemplo: erosion_bed_SI_ff_0.35_np_3_vel_-10_angle_90_move_0_0_0

    Args:
        np: Número de partículas.
        vel: Velocidad de impacto.
        fill: Fracción de llenar.
        angle: Ángulo de rotación.
        xmove: Desplazamiento en X.
        ymove: Desplazamiento en Y.
        zmove: Desplazamiento en Z.

    Returns:
        Identificador de job formateado.
    """
    return f"erosion_bed_SI_ff_{fill}_np_{np}_vel_{vel}_angle_{angle}_move_{xmove}_{ymove}_{zmove}"


def _make_sim_dir(fill: float, np: int, vel: int, angle: int, xmove: float, ymove: float, zmove: float) -> str:
    """Genera el nombre del directorio de simulación.

    Formato: erosion_bed_SI_ff_{fill}_np_{np}_vel_{vel}_angle_{angle}_move_{x}_{y}_{z}

    Args:
        fill: Fracción de llenar.
        np: Número de partículas.
        vel: Velocidad de impacto.
        angle: Ángulo de rotación.
        xmove: Desplazamiento en X [m].
        ymove: Desplazamiento en Y [m].
        zmove: Desplazamiento en Z [m].

    Returns:
        Nombre del directorio de simulación.
    """
    return f"erosion_bed_SI_ff_{fill}_np_{np}_vel_{vel}_angle_{angle}_move_{xmove}_{ymove}_{zmove}"


def compute_displacement(move: int, grain_radius_um: float) -> float:
    """Calcula el desplazamiento lineal a partir del parámetro de move.

    Args:
        move: Valor entero de la grilla de moves.
        grain_radius_um: Radio del grano [μm].

    Returns:
        Desplazamiento en μm.
    """
    return move * grain_radius_um


def generate_jobs(cfg: ExecutionConfig) -> Iterator[JobSpec]:
    """Genera todas las combinaciones de jobs según la configuración.

    Itera sobre fill × vel × angle × move_grid, calculando los
    desplazamientos y nombres apropiados.

    Args:
        cfg: Configuración de ejecución.

    Yields:
        JobSpec para cada combinación válida.
    """
    sweep = cfg.sweep
    np = cfg.simulation.np
    grain_radius = sweep.grain_radius_um

    move_grid = len(sweep.moves)

    move_combinations = list(product(sweep.moves, sweep.moves))

    index = 0
    total_steps = sweep.total_steps

    for fill in sweep.fills:
        for vel in sweep.vels:
            for angle in sweep.angles:
                for move_idx, (mx, my) in enumerate(move_combinations):
                    xmove = compute_displacement(mx, grain_radius)
                    ymove = compute_displacement(my, grain_radius)
                    zmove = 0.0

                    job_id = _make_job_id(np, vel, fill, angle, xmove, ymove, zmove)

                    nsteps = total_steps[index] if index < len(total_steps) else total_steps[0]

                    sim_dir = _make_sim_dir(fill, np, vel, angle, xmove, ymove, zmove)

                    yield JobSpec(
                        job_id=job_id,
                        fill=fill,
                        vel=vel,
                        angle=angle,
                        move_grid=move_grid,
                        xmove=xmove,
                        ymove=ymove,
                        zmove=zmove,
                        nsteps=nsteps,
                        sim_dir=sim_dir,
                    )

                index += 1


def count_jobs(cfg: ExecutionConfig) -> int:
    """Cuenta el número total de jobs generados.

    Args:
        cfg: Configuración de ejecución.

    Returns:
        Número total de jobs.
    """
    sweep = cfg.sweep
    move_count = len(sweep.moves) ** 2
    return (
        len(sweep.fills) * len(sweep.vels) * len(sweep.angles) * move_count
    )


def select_node(cfg: ExecutionConfig, job_index: int) -> str:
    """Selecciona un nodo del cluster usando round-robin.

    Args:
        cfg: Configuración de ejecución.
        job_index: Índice del job (0-indexed).

    Returns:
        Nombre del nodo seleccionado.
    """
    nodes = cfg.cluster.nodes
    if not nodes:
        return "toko01"
    return nodes[job_index % len(nodes)]


def select_ntasks(cfg: ExecutionConfig) -> int:
    """Selecciona el número de tasks MPI según np.

    Args:
        cfg: Configuración de ejecución.

    Returns:
        Número de tasks para mpirun.
    """
    np = cfg.simulation.np
    if np == 1:
        return 8
    if np == 3:
        return 16
    return cfg.simulation.cores
