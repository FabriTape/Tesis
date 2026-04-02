#!/usr/bin/env python3.11
"""Agrega todos los Parquets de un filtro en un solo DataFrame.

Lee todos los archivos ``*_TOTAL_{FILTRO}_*.parquet`` de un directorio
de resultados y los concatena en un único DataFrame, guardándolo como
``aggregated_{FILTRO}.parquet`` en el mismo directorio.

Uso:
    python3.11 scripts/aggregate_parquets.py resultados/ --filter F3C
    python3.11 scripts/aggregate_parquets.py resultados/ --filter F3C --output custom.parquet
"""

import argparse
from pathlib import Path

import polars as pl


def aggregate_parquets(
    results_dir: Path,
    filter_name: str = "F3C",
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Concatena todos los Parquets de un filtro en un solo DataFrame.

    Args:
        results_dir: Directorio que contiene los archivos Parquet.
        filter_name: Nombre del filtro a agregar (F1I, F1C, F2I, F2C, F3I, F3C).
        output_path: Ruta de salida personalizada. Si es None, se guarda en
            ``results_dir/aggregated_{filter_name}.parquet``.

    Returns:
        DataFrame concatenado con todos los resultados.

    Raises:
        FileNotFoundError: Si no se encuentran Parquets del filtro especificado.
    """
    pattern = f"*_TOTAL_{filter_name}_*.parquet"
    parquet_files = sorted(results_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No se encontraron archivos con patrón '{pattern}' en '{results_dir}'."
        )

    print(f"Encontrados {len(parquet_files)} archivos para filtro {filter_name}:")
    for f in parquet_files:
        print(f"  - {f.name}")

    frames = [pl.read_parquet(f) for f in parquet_files]
    df = pl.concat(frames, how="vertical_relaxed")

    if output_path is None:
        output_path = results_dir / f"aggregated_{filter_name}.parquet"

    df.write_parquet(output_path)
    print(f"\nDataFrame agregado: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Guardado en: {output_path}")

    return df


def main() -> None:
    """Punto de entrada CLI para agregar Parquets."""
    parser = argparse.ArgumentParser(
        description="Agrega todos los Parquets de un filtro en un solo DataFrame."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directorio que contiene los archivos Parquet de resultados.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="F3C",
        choices=["F1I", "F1C", "F2I", "F2C", "F3I", "F3C"],
        help="Nombre del filtro de erosión a agregar (por defecto: F3C).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de salida personalizada para el Parquet agregado.",
    )

    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"Error: '{args.results_dir}' no es un directorio válido.")
        return

    aggregate_parquets(args.results_dir, args.filter, args.output)


if __name__ == "__main__":
    main()
