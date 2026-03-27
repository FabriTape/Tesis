"""
Comparador de filtros de erosión granular.

Compara 6 configuraciones de filtrado sobre simulaciones LAMMPS:
  F1I/F1C: solo z (individual/cluster)
  F2I/F2C: z + vz (individual/cluster)
  F3I/F3C: z + vz + Δv (individual/cluster)

Uso:
    python main.py
    python main.py /ruta/a/simulaciones
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from filtros import (
    CONFIG_NAMES,
    build_ovito_pipeline,
    filter_F1C,
    filter_F1I,
    filter_F2C,
    filter_F2I,
    filter_F3C,
    filter_F3I,
)
from lammps_reader import list_dump_files, read_dump
from metricas import (
    build_cluster_pipeline,
    compute_gaps,
    compute_gaps_per_step,
    compute_metrics,
)
from plots import plot_comparison, plot_summary_table

# Parámetros físicos (extraer de la simulación)
BED_HEIGHT_FF_015 = 9.99989e-5
BED_HEIGHT_FF_035 = 9.99999e-5
RADIO_UM = 0.7601551  ## Mas un 0.02% para que me tome las vibraciones
RADIO_M = RADIO_UM * 1e-6
CUTOFF = RADIO_M * 2
TOLERANCIA_V_CM = 1e-6

_COLUMNS = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]

# Mapeo nombre → función de filtrado
FILTER_FUNCS = {
    "F1I": filter_F1I,
    "F1C": filter_F1C,
    "F2I": filter_F2I,
    "F2C": filter_F2C,
    "F3I": filter_F3I,
    "F3C": filter_F3C,
}

# Configs que usan Ovito (cluster)
CLUSTER_CONFIGS = {"F1C", "F2C", "F3C"}


def _detect_bed_height(sim_dir: Path) -> float:
    """Detecta la altura de la cama según el nombre de la carpeta.

    Aplica margen de 5 radios sobre la altura cruda.

    Args:
        sim_dir: Directorio de la simulación.

    Returns:
        Altura de la cama con margen [m].
    """
    name = sim_dir.name
    if "ff_0.15" in name:
        base = BED_HEIGHT_FF_015
    elif "ff_0.35" in name:
        base = BED_HEIGHT_FF_035
    else:
        base = BED_HEIGHT_FF_015
    return base + 2 * RADIO_M


def run_simulation(sim_dir: Path, output_dir: Path, image_dir: Path) -> None:
    """Ejecuta las 6 configuraciones sobre una simulación.

    Args:
        sim_dir: Directorio con los dumps.
        output_dir: Directorio para CSVs.
        image_dir: Directorio para plots.
    """
    sim_name = sim_dir.name
    print(f"\n  Simulación: {sim_name}")

    bed_height = _detect_bed_height(sim_dir)
    dump_files = list_dump_files(sim_dir)
    print(f"  Dumps encontrados: {len(dump_files)}")
    print(f"  bed_height: {bed_height * 1e6:.3f} µm")

    if not dump_files:
        print("  Sin dumps. Saltando...")
        return

    # Pipelines de Ovito
    pipeline_filter = build_ovito_pipeline(
        cutoff=CUTOFF
    )  # para filtros cluster (con coordinación)
    pipeline_metrics = build_cluster_pipeline(
        cutoff=CUTOFF
    )  # para métricas (solo clusters)

    # Resultados por config
    all_metrics: dict[str, list[dict]] = {}
    all_gaps: dict[str, list[dict]] = {}
    all_results: dict[str, list[pl.DataFrame]] = {}

    for config_name in CONFIG_NAMES:
        print(f"\n  Procesando {config_name}...")
        filter_func = FILTER_FUNCS[config_name]
        use_cluster = config_name in CLUSTER_CONFIGS
        use_dv = config_name in {"F3I", "F3C"}

        metrics_list: list[dict] = []
        results_list: list[pl.DataFrame] = []
        prev_dump: pl.DataFrame | None = None

        for i, dump_file in enumerate(dump_files):
            if i % 10 == 0:
                print(f"    step {i}/{len(dump_files)}")

            lf = read_dump(dump_file, _COLUMNS)

            if use_cluster:
                df = filter_func(
                    lf,
                    bed_height,
                    prev_dump=prev_dump,
                    tol=TOLERANCIA_V_CM,
                    pipeline=pipeline_filter,
                )
            elif use_dv:
                df = filter_func(lf, bed_height, prev_dump, TOLERANCIA_V_CM)
            else:
                df = filter_func(lf, bed_height)

            df = df.with_columns(i=pl.lit(i, dtype=pl.Int64))
            metrics_list.append(compute_metrics(df, i, pipeline=pipeline_metrics))
            results_list.append(df)

            # Guardar dump actual para filtros con Δv
            df_raw = lf.collect()
            prev_dump = df_raw.select(["id", "vx", "vy", "vz"])

        # Concatenar resultados
        df_total = (
            pl.concat(results_list, how="diagonal_relaxed")
            if results_list
            else pl.DataFrame()
        )
        gaps = compute_gaps(df_total) if not df_total.is_empty() else 0
        fluct = sum(
            1
            for j in range(1, len(metrics_list))
            if metrics_list[j]["n_bed"] < metrics_list[j - 1]["n_bed"]
        )

        all_metrics[config_name] = metrics_list
        all_results[config_name] = results_list
        all_gaps[config_name] = (
            compute_gaps_per_step(df_total) if not df_total.is_empty() else []
        )

        # Guardar CSV
        csv_path = output_dir / f"{sim_name}_{config_name}.csv"
        if not df_total.is_empty():
            df_total.write_csv(csv_path)
        print(
            f"    → IDs: {metrics_list[-1]['n_unique_ids']}, gaps: {gaps}, fluct: {fluct}"
        )

    # Generar plots
    plot_comparison(all_metrics, all_gaps, sim_name, image_dir)

    # Tabla resumen
    summaries = {}
    for config_name in CONFIG_NAMES:
        metrics = all_metrics[config_name]
        df_total = pl.concat(all_results[config_name], how="diagonal_relaxed")
        summaries[config_name] = {
            "n_ids": metrics[-1]["n_unique_ids"],
            "gaps": compute_gaps(df_total),
            "fluct": sum(
                1
                for j in range(1, len(metrics))
                if metrics[j]["n_bed"] < metrics[j - 1]["n_bed"]
            ),
            "n_clusters_final": metrics[-1]["n_clusters"],
        }

    plot_summary_table(summaries, sim_name, image_dir)


def main() -> None:
    """Punto de entrada del comparador de filtros."""
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(__file__).parent

    sim_dir = base_dir / "simulaciones"
    output_dir = base_dir / "resultados"
    image_dir = base_dir / "Imagenes"

    output_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)

    sim_dirs = sorted([d for d in sim_dir.iterdir() if d.is_dir()])
    print(f"  Encontradas {len(sim_dirs)} simulaciones en {sim_dir}")

    for sim_d in sim_dirs:
        run_simulation(sim_d, output_dir, image_dir)

    print("\n  Comparación completada.")


if __name__ == "__main__":
    main()
