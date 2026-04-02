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
    compute_gaps_per_step_raw,
    compute_metrics,
)
from ovito.pipeline import Pipeline  # type: ignore[import-untyped]
from plots import plot_comparison, plot_summary_table

# Parámetros físicos (extraer de la simulación)
RADIO_UM = 0.7603  ## Mas un 0.03% para que me tome las vibraciones
RADIO_M = RADIO_UM * 1e-6
CUTOFF = RADIO_M * 2

_COLUMNS = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

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


def _detect_bed_height(radios: int = 5) -> tuple[float, float]:
    """Detecta la altura de la cama según el nombre de la carpeta.

    Args:
        radios: Número de radios de margen sobre la cama cruda.

    Returns:
        Tupla (bed_height crudo, max_height con margen).
    """
    base = 1e-4
    if radios < 0:
        return base + radios * RADIO_M, base + radios * RADIO_M
    else:
        return base, base + radios * RADIO_M


def _process_config(
    config_name: str,
    dump_files: list[Path],
    bed_height: float,
    max_height: float,
    tol: float,
    pipeline_filter: Pipeline,
    pipeline_metrics: Pipeline,
) -> dict:
    """Procesa una configuración: filtra, calcula métricas y gaps.

    Args:
        config_name: Nombre de la configuración (F1I, F1C, ...).
        dump_files: Lista de archivos dump ordenados.
        bed_height: Altura de la cama cruda [m].
        max_height: Umbral CM del cluster [m].
        tol: Tolerancia de |Δv| [m/s].
        pipeline_filter: Pipeline de Ovito para filtros cluster.
        pipeline_metrics: Pipeline de Ovito para métricas de clusters.

    Returns:
        Dict con keys: metrics_list, results_list, df_total, gaps, fluct.
    """
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
        df_raw = lf.collect()
        df = filter_func(
            lf,
            bed_height,
            max_height,
            prev_dump=prev_dump if use_dv else None,
            tol=tol,
            pipeline=pipeline_filter,
        )
        if not df.is_empty():
            df = df.with_columns(i=pl.lit(i, dtype=pl.Int64))
        else:
            df = pl.DataFrame().with_columns(
                i=pl.lit(i, dtype=pl.Int64), id=pl.lit(None, dtype=pl.Int64)
            )
        metrics_list.append(
            compute_metrics(df, i, pipeline=pipeline_metrics if use_cluster else None)
        )
        results_list.append(df)

        prev_dump = df_raw.select(["id", "vx", "vy", "vz"])

    df_total = (
        pl.concat(results_list, how="diagonal_relaxed")
        if results_list
        else pl.DataFrame()
    )
    gaps = compute_gaps(df_total) if not df_total.is_empty() else 0
    fluct = sum(
        1
        for j in range(1, len(metrics_list))
        if metrics_list[j]["n_total"] < metrics_list[j - 1]["n_total"]
    )

    return {
        "metrics_list": metrics_list,
        "results_list": results_list,
        "df_total": df_total,
        "gaps": gaps,
        "fluct": fluct,
    }


def _apply_tracking(
    results_list: list[pl.DataFrame],
    pipeline_metrics: Pipeline,
    use_cluster: bool,
) -> dict:
    """Aplica tracking acumulativo sobre resultados filtrados.

    Toma los IDs del último step y filtra todos los steps para
    mantener solo esas partículas.

    Args:
        results_list: Lista de DataFrames por step.
        pipeline_metrics: Pipeline de Ovito para métricas de clusters.
        use_cluster: Si True, usar Ovito para métricas.

    Returns:
        Dict con keys: metrics_tracking, list_tracking, df_total, gaps, fluct.
    """
    final_df = results_list[-1]
    eroded_ids = final_df["id"].to_list()

    list_tracking: list[pl.DataFrame] = []
    metrics_tracking: list[dict] = []
    for i in range(len(results_list)):
        if not results_list[i].is_empty():
            list_tracking.append(results_list[i].filter(pl.col("id").is_in(eroded_ids)))
            metrics_tracking.append(
                compute_metrics(
                    list_tracking[i],
                    i,
                    pipeline=pipeline_metrics if use_cluster else None,
                )
            )

    df_total = (
        pl.concat(list_tracking, how="diagonal_relaxed")
        if list_tracking
        else pl.DataFrame()
    )
    gaps = compute_gaps(df_total) if not df_total.is_empty() else 0
    fluct = sum(
        1
        for j in range(1, len(metrics_tracking))
        if metrics_tracking[j]["n_total"] < metrics_tracking[j - 1]["n_total"]
    )

    return {
        "metrics_tracking": metrics_tracking,
        "list_tracking": list_tracking,
        "df_total": df_total,
        "gaps": gaps,
        "fluct": fluct,
    }


def _build_summary(
    metrics: list[dict],
    results: list[pl.DataFrame],
) -> dict[str, dict]:
    """Construye tabla resumen para una configuración.

    Args:
        metrics: Lista de métricas por step.
        results: Lista de DataFrames por step.

    Returns:
        Dict con claves de configuración y valores de resumen.
    """
    df_total = pl.concat(results, how="diagonal_relaxed")
    return {
        "n_ids": metrics[-1]["n_total"],
        "gaps": compute_gaps(df_total),
        "fluct": sum(
            1
            for j in range(1, len(metrics))
            if metrics[j]["n_total"] < metrics[j - 1]["n_total"]
        ),
        "n_clusters_final": metrics[-1]["n_clusters"],
    }


def run_simulation(
    sim_dir: Path,
    output_dir: Path,
    image_dir: Path,
    radios: int = 5,
    tol: float = 1e-4,
) -> None:
    """Ejecuta las 6 configuraciones sobre una simulación.

    Args:
        sim_dir: Directorio con los dumps.
        output_dir: Directorio para CSVs.
        image_dir: Directorio para plots.
        radios: Número de radios de margen sobre la cama cruda.
        tol: Tolerancia de |Δv| [m/s].
    """
    sim_name = sim_dir.name
    print(f"\n  Simulación: {sim_name}")

    bed_height, max_height = _detect_bed_height(radios)
    dump_files = list_dump_files(sim_dir)
    print(f"  Dumps encontrados: {len(dump_files)}")
    print(f"  bed_height: {bed_height * 1e6:.3f} µm")
    print(f"  max_height: {max_height * 1e6:.3f} µm (margen: {radios} radios)")

    if not dump_files:
        print("  Sin dumps. Saltando...")
        return

    pipeline_filter = build_ovito_pipeline(cutoff=CUTOFF)
    pipeline_metrics = build_cluster_pipeline(cutoff=CUTOFF)

    all_metrics: dict[str, list[dict]] = {}
    all_gaps: dict[str, list[dict]] = {}
    all_gaps_raw: dict[str, list[dict]] = {}
    all_results: dict[str, list[pl.DataFrame]] = {}
    all_metrics_tracking: dict[str, list[dict]] = {}
    all_results_tracking: dict[str, list[pl.DataFrame]] = {}
    all_gaps_tracking: dict[str, list[dict]] = {}
    all_gaps_raw_tracking: dict[str, list[dict]] = {}

    for config_name in CONFIG_NAMES:
        print(f"\n  Procesando {config_name}...")
        use_cluster = config_name in CLUSTER_CONFIGS

        result = _process_config(
            config_name,
            dump_files,
            bed_height,
            max_height,
            tol,
            pipeline_filter,
            pipeline_metrics,
        )

        all_metrics[config_name] = result["metrics_list"]
        all_results[config_name] = result["results_list"]
        all_gaps[config_name] = (
            compute_gaps_per_step(result["df_total"])
            if not result["df_total"].is_empty()
            else []
        )
        all_gaps_raw[config_name] = (
            compute_gaps_per_step_raw(result["df_total"])
            if not result["df_total"].is_empty()
            else []
        )

        if not result["df_total"].is_empty():
            csv_path = output_dir / f"{sim_name}_{config_name}_sin_tracking.csv"
            result["df_total"].write_csv(csv_path)
        print(
            f"    → IDs: {result['metrics_list'][-1]['n_unique_ids']}, "
            f"gaps: {result['gaps']}, fluct: {result['fluct']}"
        )

        tracking = _apply_tracking(
            result["results_list"], pipeline_metrics, use_cluster
        )

        all_metrics_tracking[config_name] = tracking["metrics_tracking"]
        all_results_tracking[config_name] = tracking["list_tracking"]
        all_gaps_tracking[config_name] = (
            compute_gaps_per_step(tracking["df_total"])
            if not tracking["df_total"].is_empty()
            else []
        )
        all_gaps_raw_tracking[config_name] = (
            compute_gaps_per_step_raw(tracking["df_total"])
            if not tracking["df_total"].is_empty()
            else []
        )

        if not tracking["df_total"].is_empty():
            csv_path = output_dir / f"{sim_name}_{config_name}_con_tracking.csv"
            tracking["df_total"].write_csv(csv_path)
        print(
            f"    → IDs: {tracking['metrics_tracking'][-1]['n_unique_ids']}, "
            f"gaps: {tracking['gaps']}, fluct: {tracking['fluct']}"
        )

    plot_comparison(
        all_metrics_tracking,
        all_gaps_tracking,
        all_gaps_raw_tracking,
        sim_name,
        image_dir,
        "con tracking",
    )
    plot_comparison(
        all_metrics,
        all_gaps,
        all_gaps_raw,
        sim_name,
        image_dir,
        "sin tracking",
    )

    summaries = {}
    for config_name in CONFIG_NAMES:
        summaries[config_name] = _build_summary(
            all_metrics[config_name], all_results[config_name]
        )
    plot_summary_table(summaries, sim_name, image_dir, "sin tracking")

    summaries_tracking = {}
    for config_name in CONFIG_NAMES:
        summaries_tracking[config_name] = _build_summary(
            all_metrics_tracking[config_name], all_results_tracking[config_name]
        )
    plot_summary_table(summaries_tracking, sim_name, image_dir, "con tracking")


def main() -> None:
    """Punto de entrada del comparador de filtros.

    Uso:
        python main.py [radios] [tol] [output_subdir]

    Args de línea de comandos:
        radios: Número de radios de margen (default: 5)
        tol: Tolerancia Δv (default: 1e-6)
        output_subdir: Subdirectorio de salida (default: "")
    """
    radios = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
    subdir = sys.argv[3] if len(sys.argv) > 3 else ""

    base_dir = Path(__file__).parent.parent

    sim_dir = base_dir / "comparar_filtros" / "simulaciones"
    output_dir = (
        base_dir.parent / "Documentos" / "resultados" / subdir
        if subdir
        else base_dir.parent / "Documentos" / "resultados"
    )
    image_dir = (
        base_dir.parent / "Documentos" / "Imagenes" / subdir
        if subdir
        else base_dir.parent / "Documentos" / "Imagenes"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted([d for d in sim_dir.iterdir() if d.is_dir()])
    print(f"  Encontradas {len(sim_dirs)} simulaciones en {sim_dir}")
    print(f"  Radios: {radios} Tol: {tol}")
    print(f"  Salida: {output_dir}")

    for sim_d in sim_dirs:
        run_simulation(sim_d, output_dir, image_dir, radios=radios, tol=tol)

    print("\n  Comparación completada.")


if __name__ == "__main__":
    main()
