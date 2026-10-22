# -*- coding: utf-8 -*-
from pathlib import Path

import polars as pl

from .config_and_discovery import extraer_metadatos


def merge_parquets(dir_path: Path) -> pl.DataFrame:
    """Lee todos los .parquet de un directorio y los fusiona en un DataFrame."""
    temp = []
    for path in sorted(dir_path.glob("*.parquet")):
        metadatos = extraer_metadatos(path.stem.replace(".parquet", ""))
        df = pl.read_parquet(path)
        df = df.with_columns(pl.lit(metadatos[k]).alias(k) for k in metadatos)
        temp.append(df)

    if not temp:
        return pl.DataFrame()

    all_cols = sorted({c for df in temp for c in df.columns})

    casted = []
    for df in temp:
        missing = [c for c in all_cols if c not in df.columns]
        df = df.with_columns(pl.lit(None).alias(c) for c in missing)
        df = df.select(all_cols)
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in {pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32}
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def merge_with_no_erosion(path_dir: Path, filter_name: str = "F2C") -> pl.DataFrame:
    """Merge de .parquet incluyendo marcadores de simulaciones sin erosión."""
    datos = sorted(path_dir.glob("*.parquet"))
    nada = sorted(path_dir.glob(f"*NADA_{filter_name}_*.txt"))

    temp = []
    for path in datos:
        metadatos = extraer_metadatos(path.stem.replace(".parquet", ""))
        df = pl.read_parquet(path)
        df = df.with_columns(pl.lit(metadatos[k]).alias(k) for k in metadatos)
        temp.append(df)

    for marker in nada:
        sim_id = marker.stem.replace(f"_NADA_{filter_name}_", "")
        metadatos = extraer_metadatos(sim_id)
        metadatos["proyectil_erosion"] = "false"
        row = pl.DataFrame(
            {
                "id": [0],
                "type": [0],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "vx": [0.0],
                "vy": [0.0],
                "vz": [0.0],
                "coordinacion": [0.0],
                "cluster": [0],
                "proyectil_erosion": [False],
                "lecho_erosion": [None],
                "paso": [0],
                "i": [0],
            }
        )
        row = row.with_columns(
            pl.lit(metadatos[k]).alias(k) for k in metadatos
        )
        temp.append(row)

    if not temp:
        return pl.DataFrame()

    all_cols = sorted({c for df in temp for c in df.columns})

    casted = []
    for df in temp:
        missing = [c for c in all_cols if c not in df.columns]
        df = df.with_columns(pl.lit(None).alias(c) for c in missing)
        df = df.select(all_cols)
        cast_expr = [
            pl.col(c).cast(pl.Int64)
            if df[c].dtype in {pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32}
            else pl.col(c)
            for c in df.columns
        ]
        casted.append(df.with_columns(cast_expr))
    return pl.concat(casted)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Junta todos los .parquet de un directorio en un .csv"
    )
    parser.add_argument("input", type=Path, help="Directorio con archivos .parquet")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Directorio de salida"
    )

    args = parser.parse_args()
    if not args.input.is_dir():
        print(f"Entrada no válida: {args.input}")
        sys.exit(1)

    output_dir = args.output or args.input
    merged_df = merge_with_no_erosion(args.input)
    if merged_df.is_empty():
        print("No se encontraron datos para fusionar")
        sys.exit(1)

    output_path = output_dir / f"merge_{args.input.name}.csv"
    merged_df.write_csv(output_path)
    total_parquets = len(list(args.input.glob("*.parquet")))
    total_nada = len(list(args.input.glob("*NADA_*.txt")))
    print(f"Total: {total_parquets} parquet + {total_nada} marcadores NADA")
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    main()
