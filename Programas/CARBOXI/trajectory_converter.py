"""
Conversor de resultados CARBOXI a formato compatible con OVITO.

Convierte archivos .parquet con trayectorias de partículas a formatos
que OVITO puede leer y visualizr:
- LAMMPS dump (.dump)
- XYZ (.xyz)

Uso:
    python3 -m CARBOXI.trajectory_converter <archivo.parquet> [--format dump|xyz]
    python3 -m CARBOXI.trajectory_converter <directorio> [--format dump|xyz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def read_parquet(path: Path) -> pl.DataFrame:
    """Lee archivo parquet de resultados CARBOXI."""
    return pl.read_parquet(path)


def get_timesteps(df: pl.DataFrame) -> list[int]:
    """Obtiene lista de timesteps únicos."""
    return sorted(df["i"].unique().to_list())


def export_to_dump(df: pl.DataFrame, timestep: int, output_path: Path) -> None:
    """Exporta un timestep a formato LAMMPS dump."""
    df_step = df.filter(pl.col("i") == timestep)
    
    if df_step.is_empty():
        return
    
    n_atoms = len(df_step)
    
    lines = [
        "ITEM: TIMESTEP",
        str(timestep),
        "ITEM: NUMBER OF ATOMS",
        str(n_atoms),
        "ITEM: ATOMS id type x y z vx vy vz fx fy fz cluster coordinacion",
    ]
    
    for row in df_step.iter_rows():
        id, type_, x, y, z, vx, vy, vz, fz, coord, cluster = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]
        coord_val = coord if coord is not None else 0
        cluster_val = cluster if cluster is not None else 0
        lines.append(f"{id} {type_} {x:.6e} {y:.6e} {z:.6e} {vx:.6e} {vy:.6e} {vz:.6e} {0.0:.6e} {0.0:.6e} {fz:.6e} {cluster_val} {coord_val}")
    
    output_path.write_text("\n".join(lines))


def export_to_xyz(df: pl.DataFrame, timestep: int, output_path: Path) -> None:
    """Exporta un timestep a formato XYZ."""
    df_step = df.filter(pl.col("i") == timestep)
    
    if df_step.is_empty():
        return
    
    n_atoms = len(df_step)
    
    lines = [
        str(n_atoms),
        f"Timestep {timestep}",
    ]
    
    for row in df_step.iter_rows():
        _, type_, x, y, z, coord, cluster = row[0], row[1], row[2], row[3], row[4], row[9], row[10]
        element = "Si" if type_ == 1 else "C"  # Si = sílice (cama), C = proyectil
        coord_val = coord if coord is not None else 0
        cluster_val = cluster if cluster is not None else 0
        lines.append(f"{element} {x:.6e} {y:.6e} {z:.6e} {cluster_val} {coord_val}")
    
    output_path.write_text("\n".join(lines))


def convert_file(
    parquet_path: Path,
    output_dir: Path | None = None,
    format: str = "dump",
) -> list[Path]:
    """Convierte un archivo parquet a trajectory.
    
    Args:
        parquet_path: Ruta al archivo .parquet
        output_dir: Directorio de salida. Si None, usa mismo que parquet.
        format: Formato de salida ('dump' o 'xyz')
    
    Returns:
        Lista de archivos generados.
    """
    df = read_parquet(parquet_path)
    timesteps = get_timesteps(df)
    
    if output_dir is None:
        output_dir = parquet_path.parent
    output_dir = output_dir / f"{parquet_path.stem}_{format}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "dump":
        export_fn = export_to_dump
        ext = "dump"
    else:
        export_fn = export_to_xyz
        ext = "xyz"
    
    generated = []
    for t in timesteps:
        output_path = output_dir / f"traj_{t:04d}.{ext}"
        export_fn(df, t, output_path)
        generated.append(output_path)
    
    return generated


def convert_directory(
    dir_path: Path,
    output_dir: Path | None = None,
    format: str = "dump",
) -> list[Path]:
    """Convierte todos los archivos .parquet en un directorio."""
    parquet_files = list(dir_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No se encontraron archivos .parquet en {dir_path}")
        return []
    
    all_files = []
    for pf in parquet_files:
        print(f"Convirtiendo {pf.name}...")
        files = convert_file(pf, output_dir, format)
        all_files.extend(files)
        print(f"  → {len(files)} archivos generados")
    
    return all_files


def main() -> None:
    """Punto de entrada del conversor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convierte resultados CARBOXI a formato OVITO"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Archivo .parquet o directorio con archivos .parquet",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["dump", "xyz"],
        default="dump",
        help="Formato de salida (default: dump)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Directorio de salida (default: mismo que entrada)",
    )
    
    args = parser.parse_args()
    
    if args.input.is_file():
        files = convert_file(args.input, args.output, args.format)
        print(f"Generados {len(files)} archivos")
    elif args.input.is_dir():
        files = convert_directory(args.input, args.output, args.format)
        print(f"Total: {len(files)} archivos")
    else:
        print(f"Entrada no válida: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
