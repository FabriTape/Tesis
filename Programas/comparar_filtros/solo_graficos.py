"""
Script rápido para generar gráficos a partir de archivos .csv ya calculados.
Evita todo el procesamiento pesado de Ovito.

Uso:
    python solo_graficos.py
    python solo_graficos.py /ruta/a/resultados
"""

import sys
import re
from pathlib import Path
import polars as pl

# Asegurar que importamos los módulos locales correctamente
sys.path.append(str(Path(__file__).parent))

from plots import plot_comparison, plot_summary_table
from metricas import (
    compute_gaps_per_step, 
    compute_gaps_per_step_raw, 
    compute_gaps, 
    compute_fluctuations
)

def process_directory(results_dir: Path):
    """Genera los gráficos comparativos a partir de CSVs ya calculados.

    Reconstruye las métricas por step (n_total, n_clusters) y los gaps a
    partir de los CSVs generados por main.py (sin volver a correr Ovito),
    y llama a plot_comparison / plot_summary_table por cada grupo
    (directorio, simulación, variante de tracking).

    Args:
        results_dir: Directorio con los CSVs (busca recursivamente *.csv).
    """
    print(f"Buscando archivos CSV en: {results_dir}")
    csv_files = list(results_dir.rglob("*.csv"))

    if not csv_files:
        print("No se encontraron archivos .csv. ¡Corré main.py primero!")
        return

    # Agrupar archivos por (Directorio Salida, Nombre Simulación, Estado de Tracking)
    simulations = {}
    for csv_path in csv_files:
        # Extraer el nombre, filtro y variante del nombre del archivo
        match = re.match(r"^(.+?)_(F[123][IC])_(con_tracking|sin_tracking)$", csv_path.stem)
        if not match:
            continue

        sim_name, config, tracking = match.groups()
        group_key = (csv_path.parent, sim_name, tracking)

        if group_key not in simulations:
            simulations[group_key] = {}

        simulations[group_key][config] = csv_path

    # Generar gráficos para cada grupo encontrado
    for (out_dir, sim_name, tracking), configs_dict in simulations.items():
        print(f"\nGenerando gráficos para: {sim_name} ({tracking.replace('_', ' ')})")
        print(f"  Carpeta: {out_dir.name}")

        all_metrics = {}
        all_gaps = {}
        all_gaps_raw = {}
        summaries = {}

        # Procesar en orden
        for config in ["F1I", "F1C", "F2I", "F2C", "F3I", "F3C"]:
            if config not in configs_dict:
                continue

            df = pl.read_csv(configs_dict[config])

            if df.is_empty():
                all_metrics[config] = []
                all_gaps[config] = []
                all_gaps_raw[config] = []
                summaries[config] = {"n_ids": 0, "gaps": 0, "fluct": 0, "n_clusters_final": 0}
                continue

            max_step = df["i"].max()
            metrics_list = []
            
            # Reconstruir métricas básicas por step para el gráfico (a) y (b)
            for step in range(max_step + 1):
                step_df = df.filter(pl.col("i") == step)
                n_total = step_df.shape[0]
                n_clusters = step_df["cluster"].n_unique() if "cluster" in step_df.columns else 0
                
                metrics_list.append({
                    "step": step,
                    "n_total": n_total,
                    "n_clusters": n_clusters
                })

            # Reconstruir diccionarios para plots
            all_metrics[config] = metrics_list
            all_gaps[config] = compute_gaps_per_step(df)
            all_gaps_raw[config] = compute_gaps_per_step_raw(df)

            # Reconstruir resumen para la tabla
            summaries[config] = {
                "n_ids": metrics_list[-1]["n_total"] if metrics_list else 0,
                "gaps": compute_gaps(df),
                "fluct": compute_fluctuations(metrics_list),
                "n_clusters_final": metrics_list[-1]["n_clusters"] if metrics_list else 0
            }

        # Mandar los datos reconstruidos a tus funciones graficadoras originales
        title_suffix = tracking.replace("_", " ")
        plot_comparison(all_metrics, all_gaps, all_gaps_raw, sim_name, out_dir, title_suffix)
        plot_summary_table(summaries, sim_name, out_dir, title_suffix)

if __name__ == "__main__":
    # Permite pasar el directorio por parámetro o usa el predeterminado por defecto
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        # Sube un nivel desde comparar_filtros y entra a Documentos/resultados
        target_dir = Path(__file__).parent.parent.parent / "Documentos" / "resultados"

    if target_dir.exists():
        process_directory(target_dir)
    else:
        print(f"El directorio de resultados '{target_dir}' no existe.")
