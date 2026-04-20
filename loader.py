"""
Lectura de trayectorias LAMMPS.

Combina:
- LammpsTrajectoryReader: Motor C++ de OVITO para archivos unificados.
- read_dump: Lector legacy para archivos .gz comprimidos.

Física: Los archivos dump contienen posiciones y velocidades de partículas.
El lector abstrae el formato subyacente, permitiendo que el código
de análisis trabaje con frames sin conocer la estructura de archivos.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from ovito.io import import_file  # type: ignore[import-untyped]

_ALL_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

_STEP_TO_US: float = 1.25


class LammpsTrajectoryReader:
    """Lector de trayectorias LAMMPS usando el motor C++ de OVITO.

    Soporta tanto archivos individuales (.dump) como secuencias legacy (.gz).
    El lector abstrae el formato subyacente, permitiendo que el código de
    análisis trabaje con frames sin conocer la estructura de archivos.

    Attributes:
        num_frames: Número de frames en la trayectoria.
        timestep_interval: Intervalo de timesteps entre frames consecutivos.
    """

    def __init__(self, pattern: str) -> None:
        """Inicializa el lector con un patrón de archivo.

        Args:
            pattern: Ruta a archivo único ('output.dump') o comodín ('output.*.gz').
        """
        self.pipeline = import_file(pattern)
        self.num_frames = self.pipeline.source.num_frames

        t0 = self.pipeline.compute(0).attributes.get("Timestep", 0)
        if self.num_frames > 1:
            t1 = self.pipeline.compute(1).attributes.get("Timestep", 0)
            self.timestep_interval = t1 - t0
        else:
            self.timestep_interval = 0

    def get_frame_lazy(self, frame_index: int) -> pl.LazyFrame:
        """Lee un frame específico hacia un LazyFrame de Polars.

        Args:
            frame_index: Índice del frame a leer (0 a num_frames-1).

        Returns:
            LazyFrame con columnas: id, type, x, y, z, vx, vy, vz.
        """
        data = self.pipeline.compute(frame_index)

        return pl.DataFrame(
            {
                "id": data.particles["Particle Identifier"][:],
                "type": data.particles["Particle Type"][:],
                "x": data.particles.positions[:, 0],
                "y": data.particles.positions[:, 1],
                "z": data.particles.positions[:, 2],
                "vx": data.particles.velocities[:, 0],
                "vy": data.particles.velocities[:, 1],
                "vz": data.particles.velocities[:, 2],
            }
        ).lazy()


def detect_and_create_reader(sim_dir: Path) -> LammpsTrajectoryReader | None:
    """Detecta automáticamente el formato de los datos de trayectoria.

    Busca primero el archivo unificado (metodología nueva), luego la secuencia
    legacy de archivos .gz. Cada carpeta contiene un solo tipo de formato.

    Args:
        sim_dir: Directorio de la simulación a procesar.

    Returns:
        LammpsTrajectoryReader configurado, o None si no se encuentran datos.
    """
    single_dump = sim_dir / "output_trayectoria.dump"
    if single_dump.exists():
        print(
            f"  [Auto-detect] Archivo unificado: {single_dump.name} "
            f"({single_dump.stat().st_size / 1024**2:.1f} MB)"
        )
        return LammpsTrajectoryReader(str(single_dump))

    gz_files = list(sim_dir.glob("output.*.gz"))
    if gz_files:
        wildcard = str(sim_dir / "output.*.gz")
        print(f"  [Auto-detect] Secuencia legacy: {len(gz_files)} archivos .gz")
        return LammpsTrajectoryReader(wildcard)

    return None


def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lee un archivo dump comprimido de LAMMPS como LazyFrame de Polars.

    Args:
        path: Ruta al archivo ``.gz`` a leer.
        columns: Lista de columnas a seleccionar.

    Returns:
        LazyFrame con las columnas solicitadas.
    """
    lf = pl.scan_csv(
        path,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,
        new_columns=_ALL_COLUMNS,
        infer_schema_length=10_000,
        schema_overrides={"type": pl.UInt8},
    )
    return lf.select(columns or _ALL_COLUMNS)


def list_dump_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Lista y ordena cronológicamente los archivos dump de una simulación.

    Args:
        directory: Directorio donde buscar.
        recursive: Si True, busca en subdirectorios también.
    """
    files: list[Path] = []
    for f in directory.iterdir():
        if f.is_file() and f.name.startswith("output") and f.name.endswith(".gz"):
            files.append(f)
    if recursive:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.is_file() and f.name.startswith("output") and f.name.endswith(".gz"):
                        files.append(f)
    files.sort(key=lambda p: _extract_timestep(p.name))
    return files


def get_timestep_interval(files: list[Path]) -> int:
    """Extrae el intervalo de volcado desde la lista de archivos."""
    if len(files) < 2:
        return 0
    return _extract_timestep(files[1].name)


def save_wave_data(
    wave_frame: pl.LazyFrame,
    sim_id: str,
    step_index: int,
    timestamp: str,
    output_dir: Path,
    filter_name: str = "F3C",
) -> Path:
    """Guarda las partículas detectadas como parte de la onda de choque.

    Args:
        wave_frame: LazyFrame con las partículas de la onda.
        sim_id: Identificador de la simulación.
        step_index: Índice del frame.
        timestamp: Marca temporal.
        output_dir: Directorio de salida.
        filter_name: Nombre del filtro aplicado.

    Returns:
        Ruta al archivo guardado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_ONDA_{filter_name}_{step_index}_{timestamp}.parquet"
    wave_frame.sink_parquet(path)
    return path


def _extract_timestep(filename: str) -> int:
    """Extrae el número de timestep del nombre de un archivo dump."""
    match = re.search(r"output\.(\d+)\.gz", filename)
    return int(match.group(1)) if match else 0