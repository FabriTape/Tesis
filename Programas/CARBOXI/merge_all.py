from pathlib import Path

import polars as pl

from .config_and_discovery import discover_simulations, extraer_metadatos


def merge_all_parquet(path_file: Path) -> pl.DataFrame:
    datos = discover_simulations(path_file,"ff_", "file")
    temp = []
    for dato in datos:
        metadatos = extraer_metadatos(str(dato.name).replace(".parquet", ""))
        df = pl.read_parquet(dato)
        df = df.with_columns([pl.lit(metadatos[k]).alias(k) for k in metadatos.keys()])
        temp.append(df)
        # En merge_all_parquet, antes del concat:
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

    output_dir = args.output or args.input  # fallback a input

    try:
        merged_df = merge_all_parquet(args.input)
        if merged_df.is_empty():
            print("No se encontraron datos para fusionar")
            sys.exit(1)
        output_path = output_dir / f"merge_{args.input.name}.csv"
        merged_df.write_csv(output_path)
        print(f"Total: {len(discover_simulations(args.input, 'ff', 'file'))} archivos")
        print(f"Guardado en: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
