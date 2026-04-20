"""Funciones de merge para archivos parquet.

Combina múltiples archivos parquet en un solo DataFrame,
incluyendo marcadores de 'sin erosión'.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..config import discover_simulations, extraer_metadatos


def merge_all_parquet(path_file: Path) -> pl.DataFrame:
    """Merge de todos los archivos parquet de un directorio.

    Args:
        path_file: Ruta al directorio con archivos .parquet.

    Returns:
        DataFrame consolidado.
    """
    datos = discover_simulations(path_file, "ff_", "file")
    temp = []
    for dato in datos:
        metadatos = extraer_metadatos(str(dato.name).replace(".parquet", ""))
        df = pl.read_parquet(dato)
        df = df.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(df)

    casted = []
    for df in temp:
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in [pl.UInt8, pl.UInt16, pl.UInt32]
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def merge_with_no_erosion(path_dir: Path) -> pl.DataFrame:
    """Merge de archivos parquet incluyendo marcadores de 'sin erosión'.

    Args:
        path_dir: Ruta al directorio.

    Returns:
        DataFrame consolidado con marcadores de sin erosión.
    """
    datos = list(path_dir.glob("*.parquet"))
    nada = list(path_dir.glob("*NADA*.txt"))

    temp = []
    for dato in datos:
        metadatos = extraer_metadatos(str(dato.name).replace(".parquet", ""))
        df = pl.read_parquet(dato)
        df = df.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(df)

    for marker in nada:
        marker.read_text()
        sim_id = marker.stem.replace("_NADA_F3C_", "")
        metadatos = extraer_metadatos(sim_id)
        metadatos["proyectil_erosion"] = "false"
        row = pl.DataFrame({
            "id": [0],
            "type": [0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "vx": [0.0],
            "vy": [0.0],
            "vz": [0.0],
            "fz": [0.0],
            "coordinacion": [0.0],
            "cluster": [0.0],
            "proyectil_erosion": [False],
            "paso": [0],
            "i": [0],
        })
        row = row.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(row)

    if not temp:
        return pl.DataFrame()

    casted = []
    for df in temp:
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in [pl.UInt8, pl.UInt16, pl.UInt32]
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def main() -> None:
    """Punto de entrada del conversor."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Junta todos los .parquet de un directorio en un .csv"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directorio con archivos .parquet",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Directorio de salida (default: mismo que entrada)",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Entrada no válida: {args.input} (debe ser un directorio)")
        sys.exit(1)

    output_dir = args.output or args.input

    try:
        merged_df = merge_with_no_erosion(args.input)
        if merged_df.is_empty():
            print("No se encontraron datos para fusionar")
            sys.exit(1)
        output_path = output_dir / f"merge_{args.input.name}.csv"
        merged_df.write_csv(output_path)
        print(
            f"Total: {len(list(args.input.glob('*.parquet')) + list(args.input.glob('*NADA*.txt')))} archivos"
        )
        print(f"Guardado en: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()