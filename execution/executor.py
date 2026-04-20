"""Ejecutor de jobs en cluster SLURM.

Maneja el ciclo de vida de los jobs: submit, wait, status
y gestión de-directorios de simulación.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from config import ExecutionConfig
from scheduler import JobSpec


@dataclass(frozen=True)
class JobStatus:
    """Estado de un job en SLURM.

    Attributes:
        job_id: ID del job en SLURM.
        slurm_id: Identificador del job (job_name).
        state: Estado del job (PENDING, RUNNING, COMPLETED, FAILED, etc).
        exit_code: Código de salida.
    """

    slurm_id: str
    state: str
    exit_code: int | None = None


def submit_job(
    job_spec: JobSpec,
    cfg: ExecutionConfig,
    base_dir: Path,
    script_content: str,
    config_path: Path | None = None,
) -> int:
    """Envía un job a SLURM usando sbatch.

    Args:
        job_spec: Especificación del job.
        cfg: Configuración de ejecución.
        base_dir: Directorio raíz del toolkit.
        script_content: Contenido del script SLURM.
        config_path: Ruta al config.toml a copiar (None = usar default).

    Returns:
        Job ID asignado por SLURM.

    Raises:
        RuntimeError: Si sbatch falla.
    """
    script_path = base_dir / "execution" / "generated" / f"{job_spec.job_id}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    script_path.write_text(script_content)

    configs_dir = base_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        cwd=base_dir,
    )

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    slurm_id: int | None = None
    output = result.stdout.strip()
    for line in output.split("\n"):
        if line.startswith("Submitted batch job "):
            parts = line.split()
            slurm_id = int(parts[-1])

    if slurm_id is None:
        raise RuntimeError(f"No job ID in sbatch output: {output}")

    if config_path and config_path.exists():
        dst = configs_dir / f"config_{slurm_id}.toml"
        shutil.copy(config_path, dst)

    return slurm_id


def wait_for_job(slurm_id: int, timeout: int = 3600) -> JobStatus:
    """Espera a que un job termine.

    Args:
        slurm_id: ID del job en SLURM.
        timeout: Tiempo máximo de espera [s].

    Returns:
        Estado final del job.

    Raises:
        TimeoutError: Si excede el timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = check_job_status(slurm_id)
        if status.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            return status
        time.sleep(10)

    raise TimeoutError(f"Job {slurm_id} timeout after {timeout}s")


def check_job_status(slurm_id: int) -> JobStatus:
    """Consulta el estado de un job en SLURM.

    Args:
        slurm_id: ID del job en SLURM.

    Returns:
        Estado actual del job.
    """
    result = subprocess.run(
        ["squeue", "-j", str(slurm_id), "--noheader", "-o", "%T"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return JobStatus(
            slurm_id=str(slurm_id),
            state="UNKNOWN",
            exit_code=-1,
        )

    state = result.stdout.strip() or "UNKNOWN"

    exit_code: int | None = None
    if state in ("COMPLETED", "FAILED", "CANCELLED"):
        exit_code = get_exit_code(slurm_id)

    return JobStatus(
        slurm_id=str(slurm_id),
        state=state,
        exit_code=exit_code,
    )


def get_exit_code(slurm_id: int) -> int | None:
    """Obtiene el código de salida de un job completado.

    Args:
        slurm_id: ID del job en SLURM.

    Returns:
        Código de salida o None si no se puede determinar.
    """
    result = subprocess.run(
        [
            "sacct",
            "-j",
            str(slurm_id),
            "--noheader",
            "-o",
            "exitcode",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    lines = result.stdout.strip().split("\n")
    if not lines:
        return None

    first = lines[0].split("|")
    if len(first) < 2:
        return None

    try:
        exit_code = int(first[1].split(":")[0])
        return exit_code
    except (ValueError, IndexError):
        return None


def check_jobs_status(slurm_ids: list[int]) -> Iterator[tuple[int, JobStatus]]:
    """Itera sobre los estados de múltiples jobs.

    Args:
        slurm_ids: Lista de IDs de jobs en SLURM.

    Yields:
        Tupla (slurm_id, estado).
    """
    for sid in slurm_ids:
        yield sid, check_job_status(sid)


def create_sim_dir(base_dir: Path, sim_dir: str) -> Path:
    """Crea el directorio de simulación.

    Args:
        base_dir: Directorio base.
        sim_dir: Nombre del directorio.

    Returns:
        Ruta al directorio creado.
    """
    path = base_dir / sim_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_sim_input(
    sim_dir: Path,
    input_template: Path,
    job_spec: JobSpec,
    cfg: ExecutionConfig,
) -> None:
    """Prepara los archivos de entrada para una simulación.

    Args:
        sim_dir: Directorio de la simulación.
        input_template: Ruta al template LAMMPS.
        job_spec: Especificación del job.
        cfg: Configuración de ejecución.
    """
    content = input_template.read_text()

    replacements = {
        "xmove": f"{job_spec.xmove}e-6",
        "ymove": f"{job_spec.ymove}e-6",
        "zmove": f"{job_spec.zmove}e-6",
        "velocity": str(job_spec.vel),
        "nsteps": str(job_spec.nsteps),
        "ndumps": str(cfg.simulation.ndumps),
        "nthermo": str(cfg.simulation.nthermo),
        "timestep": str(cfg.simulation.timestep),
        "young_modulus": str(cfg.material.young_modulus),
        "poisson": str(cfg.material.poisson),
        "surface_energy": str(cfg.material.surface_energy),
        "density_proj": str(cfg.material.density),
        "proc_x": str(cfg.simulation.processors[0]),
        "proc_y": str(cfg.simulation.processors[1]),
        "proc_z": str(cfg.simulation.processors[2]),
    }

    for var, val in replacements.items():
        content = content.replace(f"${{{var}}}", val)
        content = content.replace(f"${{{var.upper()}}}", val)

    (sim_dir / "collision_erosion_bed.input").write_text(content)


def run_mpirun(
    sim_dir: Path,
    cfg: ExecutionConfig,
    job_spec: JobSpec,
    exe: Path,
) -> subprocess.CompletedProcess:
    """Ejecuta mpirun para una simulación.

    Args:
        sim_dir: Directorio de simulación.
        cfg: Configuración.
        job_spec: Especificación del job.
        exe: Ruta al ejecutable LAMMPS.

    Returns:
        Resultado del proceso.
    """
    ntasks = cfg.simulation.cores
    input_file = sim_dir / "collision_erosion_bed.input"

    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-np", str(ntasks),
        str(exe),
        "-in", str(input_file),
    ]

    return subprocess.run(
        cmd,
        cwd=sim_dir,
        capture_output=True,
        text=True,
    )
