"""
App Streamlit para comparación visual de filtros de erosión.

Uso:
    streamlit run app.py
    streamlit run app.py -- --results-dir /path/to/resultados
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Comparar Filtros", layout="wide")

# ---------------------------------------------------------------------------
# Colores y estilos
# ---------------------------------------------------------------------------

COLORS = {
    "F1I": "#e74c3c",
    "F1C": "#e74c3c",
    "F2I": "#3498db",
    "F2C": "#3498db",
    "F3I": "#2ecc71",
    "F3C": "#2ecc71",
}
LINESTYLES = {
    "F1I": "-",
    "F1C": "--",
    "F2I": "-",
    "F2C": "--",
    "F3I": "-",
    "F3C": "--",
}

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def discover_results(results_dir: Path) -> dict[str, list[str]]:
    """Descubre subdirectorios y sus CSVs disponibles.

    Maneja dos niveles de anidación:
    - Nivel 1: resultados/subdir/*.csv
    - Nivel 2: resultados/subdir/subsubdir/*.csv

    Las claves del dict son rutas RELATIVAS al directorio seleccionado.

    Args:
        results_dir: Directorio raíz de resultados.

    Returns:
        Dict con clave = ruta relativa, valor = lista de nombres de CSV.
    """
    result: dict[str, list[str]] = {}
    if not results_dir.exists():
        return result

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue

        # Nivel 1: CSVs directamente en el subdirectorio
        csvs = [f.stem for f in sorted(d.glob("*.csv"))]
        if csvs:
            # Clave = nombre relativo al directorio padre
            key = d.relative_to(results_dir).as_posix()
            result[key] = csvs
            continue

        # Nivel 2: buscar CSVs en sub-subdirectorios
        for dd in sorted(d.iterdir()):
            if dd.is_dir():
                csvs = [f.stem for f in sorted(dd.glob("*.csv"))]
                if csvs:
                    key = dd.relative_to(results_dir).as_posix()
                    result[key] = csvs

    return result


def load_csv(results_dir: Path, subdir: str, sim_name: str, config: str) -> pl.DataFrame | None:
    """Carga un CSV de resultados.

    Args:
        results_dir: Directorio raíz.
        subdir: Ruta relativa (ej: "0r_no_tracking" o "v3/0r_tracking").
        sim_name: Nombre de simulación.
        config: Nombre de configuración (ej: "F1C").

    Returns:
        DataFrame o None si no existe.
    """
    csv_path = results_dir / subdir / f"{sim_name}_{config}.csv"
    if csv_path.exists():
        return pl.read_csv(csv_path)
    return None


def compute_metrics_from_df(df: pl.DataFrame) -> dict:
    """Calcula métricas de un DataFrame.

    Args:
        df: DataFrame con columna "i" (step).

    Returns:
        Diccionario con métricas por step.
    """
    bed = df.filter(pl.col("type") != 2)

    if bed.is_empty():
        return {"steps": [], "n_bed": [], "n_clusters": [], "gaps_acum": []}

    bed_counts = bed.group_by("i").agg(pl.len().alias("n")).sort("i")
    steps = bed_counts["i"].to_list()
    n_bed = bed_counts["n"].to_list()

    # n_clusters
    n_clusters = []
    if "cluster" in bed.columns:
        nc = bed.group_by("i").agg(pl.col("cluster").n_unique().alias("nc")).sort("i")
        n_clusters = nc["nc"].to_list()

    # gaps acumulados
    gaps_acum = []
    seen_ids: set[int] = set()
    for step in steps:
        step_ids = set(bed.filter(pl.col("i") == step)["id"].to_list())
        gaps = len(seen_ids - step_ids)
        gaps_acum.append(gaps)
        seen_ids.update(step_ids)

    return {
        "steps": steps,
        "n_bed": n_bed,
        "n_clusters": n_clusters,
        "gaps_acum": gaps_acum,
    }


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada de la app Streamlit."""
    st.title("Comparación de Filtros de Erosión")

    # --- Selector de directorio ---
    default_dir = str(Path(__file__).parent / "resultados")
    results_path = st.sidebar.text_input(
        "Directorio de resultados",
        value=default_dir,
        help="Ruta al directorio con subdirectorios de resultados (CSVs)",
    )
    results_dir = Path(results_path)

    if not results_dir.exists():
        st.error(f"El directorio {results_dir} no existe.")
        return

    # --- Descubrir datos ---
    available = discover_results(results_dir)

    if not available:
        st.warning("No se encontraron CSVs en el directorio.")
        return

    # --- Selectores en sidebar ---
    st.sidebar.header("Configuraciones")

    subdirs = list(available.keys())
    selected_subdirs = st.sidebar.multiselect(
        "Variaciones (radios × tracking)",
        options=subdirs,
        default=subdirs[:2],
        help="Selecciona las variaciones a comparar",
    )

    # Simulaciones
    all_csv_names = []
    for s in selected_subdirs:
        all_csv_names.extend(available[s])
    sim_names = extract_sim_names(all_csv_names)
    config_names = extract_config_names(all_csv_names)

    selected_sim = st.sidebar.selectbox(
        "Simulación",
        options=sim_names,
        help="Selecciona la simulación a analizar",
    )

    selected_configs = st.sidebar.multiselect(
        "Filtros (F1/F2/F3 × I/C)",
        options=config_names,
        default=config_names,
        help="Selecciona las configuraciones de filtro",
    )

    # --- Cargar datos ---
    all_data: dict[str, dict] = {}

    for subdir in selected_subdirs:
        for config in selected_configs:
            df = load_csv(results_dir, subdir, selected_sim, config)
            if df is not None:
                key = f"{subdir}/{config}"
                all_data[key] = compute_metrics_from_df(df)

    if not all_data:
        st.warning("No se encontraron datos para la selección actual.")
        return

    # --- Layout: 3 columnas de plots ---
    col1, col2, col3 = st.columns(3)

    # Plot 1: n_bed vs step
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, data in all_data.items():
            if data["steps"]:
                config = key.split("/")[-1]
                ax.plot(
                    data["steps"],
                    data["n_bed"],
                    label=key,
                    color=COLORS.get(config, "gray"),
                    linestyle=LINESTYLES.get(config, "-"),
                    linewidth=1.5,
                )
        ax.set_xlabel("Step")
        ax.set_ylabel("Partículas")
        ax.set_title("N partículas erosionadas")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Plot 2: n_clusters vs step
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        has_clusters = False
        for key, data in all_data.items():
            if data["n_clusters"]:
                config = key.split("/")[-1]
                ax.plot(
                    data["steps"],
                    data["n_clusters"],
                    label=key,
                    color=COLORS.get(config, "gray"),
                    linestyle=LINESTYLES.get(config, "-"),
                    linewidth=1.5,
                )
                has_clusters = True
        if has_clusters:
            ax.set_xlabel("Step")
            ax.set_ylabel("N clusters")
            ax.set_title("Clusters por step")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Sin datos de clusters para esta selección.")
        plt.close(fig)

    # Plot 3: gaps acumulados vs step
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, data in all_data.items():
            if data["gaps_acum"]:
                config = key.split("/")[-1]
                ax.plot(
                    data["steps"],
                    data["gaps_acum"],
                    label=key,
                    color=COLORS.get(config, "gray"),
                    linestyle=LINESTYLES.get(config, "-"),
                    linewidth=1.5,
                )
        ax.set_xlabel("Step")
        ax.set_ylabel("Gaps acumulados")
        ax.set_title("IDs que desaparecen")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # --- Tabla resumen ---
    st.header("Resumen")

    summary_data = []
    for subdir in selected_subdirs:
        for config in selected_configs:
            key = f"{subdir}/{config}"
            if key not in all_data:
                continue

            data = all_data[key]
            n_bed_list = data["n_bed"]

            # Cargar CSV para gaps totales
            df = load_csv(results_dir, subdir, selected_sim, config)
            if df is None:
                continue

            bed = df.filter(pl.col("type") != 2)
            n_ids = bed["id"].n_unique() if not bed.is_empty() else 0

            # Gaps
            gaps = 0
            if not bed.is_empty():
                gap_particles = (
                    bed.group_by("id")
                    .agg(
                        pl.col("i").min().alias("ini"),
                        pl.col("i").max().alias("fin"),
                        pl.col("i").count().alias("n"),
                    )
                    .with_columns((pl.col("fin") - pl.col("ini") + 1).alias("rango"))
                    .filter(pl.col("n") != pl.col("rango"))
                )
                gaps = gap_particles.shape[0]

            # Fluctuaciones
            fluct = sum(
                1 for j in range(1, len(n_bed_list)) if n_bed_list[j] < n_bed_list[j - 1]
            )

            summary_data.append(
                {
                    "Variación": subdir,
                    "Filtro": config,
                    "IDs": n_ids,
                    "Gaps": gaps,
                    "Fluct": fluct,
                    "Max bed": max(n_bed_list) if n_bed_list else 0,
                    "Min bed": min(n_bed_list) if n_bed_list else 0,
                }
            )

    if summary_data:
        st.dataframe(
            pl.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )

    # --- Info ---
    st.sidebar.header("Info")
    st.sidebar.write(f"Subdirectorios: {len(available)}")
    st.sidebar.write(f"Configuraciones: {len(config_names)}")
    st.sidebar.write(f"Simulaciones: {len(sim_names)}")


if __name__ == "__main__":
    main()
