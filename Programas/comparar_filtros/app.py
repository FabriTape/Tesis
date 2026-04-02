"""App Streamlit para comparación visual de filtros de erosión granular.

Permite explorar resultados de las 6 configuraciones de filtrado (F1/F2/F3 × I/C)
mediante gráficos interactivos y tablas resumen.

Uso:
    streamlit run app.py
    streamlit run app.py -- --results-dir /path/to/resultados
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

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
# Constantes
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent / "Documentos" / "resultados"
_PLOT_FIGSIZE = (6, 4)
_LEGEND_FONTSIZE = 7
_GRID_ALPHA = 0.3


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterMetrics:
    """Métricas extraídas de un CSV de filtro para una simulación.

    Attributes:
        steps: Índices de paso temporal.
        n_total: Conteo de partículas por step.
        n_clusters: Conteo de clusters por step (vacío si no aplica).
        gaps_acum: Gaps acumulados por step.
    """

    steps: list[int]
    n_total: list[int]
    n_clusters: list[int]
    gaps_acum: list[int]


@dataclass(frozen=True)
class SummaryRow:
    """Fila de la tabla resumen para una combinación variación×filtro.

    Attributes:
        variacion: Nombre de la variación (ej: "0r", "5r_tracking").
        filtro: Nombre del filtro (ej: "F1C").
        n_ids: Partículas únicas identificadas.
        n_gaps: Partículas con aparición intermitente.
        n_fluct: Disminuciones en el conteo entre steps.
        max_bed: Máximo conteo de partículas.
        min_bed: Mínimo conteo de partículas.
    """

    variacion: str
    filtro: str
    n_ids: int
    n_gaps: int
    n_fluct: int
    max_bed: int
    min_bed: int


# ---------------------------------------------------------------------------
# Descubrimiento de datos
# ---------------------------------------------------------------------------


def discover_results(results_dir: Path) -> dict[str, list[str]]:
    """Descubre subdirectorios y CSVs disponibles en la jerarquía de resultados.

    Soporta hasta dos niveles de anidación:
    - Nivel 1: resultados/subdir/*.csv
    - Nivel 2: resultados/subdir/subsubdir/*.csv

    Args:
        results_dir: Directorio raíz de resultados.

    Returns:
        Dict con clave = ruta relativa, valor = lista de stems de CSVs.
    """
    result: dict[str, list[str]] = {}
    if not results_dir.exists():
        return result

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue

        csvs = [f.stem for f in sorted(d.glob("*.csv"))]
        if csvs:
            result[d.relative_to(results_dir).as_posix()] = csvs
            continue

        for dd in sorted(d.iterdir()):
            if dd.is_dir():
                csvs = [f.stem for f in sorted(dd.glob("*.csv"))]
                if csvs:
                    result[dd.relative_to(results_dir).as_posix()] = csvs

    return result


def extract_sim_names(csv_names: list[str]) -> list[str]:
    """Extrae nombres únicos de simulaciones de los stems de CSVs.

    Args:
        csv_names: Lista de stems (ej: ["sim1_F1C_con_tracking", "sim1_F2I"]).

    Returns:
        Lista ordenada de nombres de simulación (ej: ["sim1"]).
    """
    sims = {
        re.sub(r"_F[123][IC](?:_(?:con|sin)_tracking)?$", "", name)
        for name in csv_names
    }
    return sorted(sims)


def extract_config_names(csv_names: list[str]) -> list[str]:
    """Extrae nombres únicos de configuraciones de filtro.

    Args:
        csv_names: Lista de stems de CSVs.

    Returns:
        Lista ordenada de configuraciones (ej: ["F1C", "F2I", "F3C"]).
    """
    configs = {
        match.group(1)
        for name in csv_names
        if (match := re.search(r"_(F[123][IC])(?:_(?:con|sin)_tracking)?$", name))
    }
    return sorted(configs)


# ---------------------------------------------------------------------------
# Carga y cómputo
# ---------------------------------------------------------------------------


def _parse_stem(stem: str) -> tuple[str, str, str] | None:
    """Descompone un stem de CSV en simulación, filtro y variante.

    Args:
        stem: Nombre del archivo sin extensión (ej: "sim_F1C_con_tracking").

    Returns:
        Tupla (sim_name, config, variant) o None si no coincide el patrón.
    """
    match = re.match(
        r"^(.+?)_(F[123][IC])_(con_tracking|sin_tracking)$", stem
    )
    if not match:
        match = re.match(r"^(.+?)_(F[123][IC])$", stem)
        if not match:
            return None
        return match.group(1), match.group(2), ""
    return match.group(1), match.group(2), match.group(3)


def load_csv(results_dir: Path, subdir: str, stem: str) -> pl.DataFrame | None:
    """Carga un CSV de resultados filtrados por su stem completo.

    Args:
        results_dir: Directorio raíz de resultados.
        subdir: Ruta relativa dentro de results_dir.
        stem: Nombre del archivo sin extensión.

    Returns:
        DataFrame con datos del filtro, o None si no existe.
    """
    csv_path = results_dir / subdir / f"{stem}.csv"
    return pl.read_csv(csv_path) if csv_path.exists() else None


def compute_metrics_from_df(df: pl.DataFrame) -> FilterMetrics:
    """Calcula métricas temporales a partir de un DataFrame de partículas.

    Args:
        df: DataFrame con columnas ["i", "id", "cluster"].

    Returns:
        Métricas por step: conteos, clusters y gaps acumulados.
    """
    if df.is_empty():
        return FilterMetrics([], [], [], [])

    bed_counts = df.group_by("i").agg(pl.len().alias("n")).sort("i")
    steps = bed_counts["i"].to_list()
    n_total = bed_counts["n"].to_list()

    n_clusters = _compute_clusters_per_step(df)
    gaps_acum = _compute_gaps_accumulated(df, steps)

    return FilterMetrics(steps, n_total, n_clusters, gaps_acum)


def _compute_clusters_per_step(df: pl.DataFrame) -> list[int]:
    """Calcula el número de clusters únicos por step.

    Args:
        df: DataFrame con columnas ["i", "cluster"].

    Returns:
        Lista vacía si no hay columna "cluster", sino conteos por step.
    """
    if "cluster" not in df.columns:
        return []

    nc = df.group_by("i").agg(pl.col("cluster").n_unique().alias("nc")).sort("i")
    return nc["nc"].to_list()


def _compute_gaps_accumulated(df: pl.DataFrame, steps: list[int]) -> list[int]:
    """Calcula IDs que desaparecen entre steps (gaps acumulados).

    Un gap es un ID visto en pasos anteriores pero ausente en el step actual.

    Args:
        df: DataFrame con columnas ["i", "id"].
        steps: Lista de steps a evaluar.

    Returns:
        Lista de conteos de gaps por step.
    """
    gaps_acum: list[int] = []
    seen_ids: set[int] = set()

    for step in steps:
        step_ids = set(df.filter(pl.col("i") == step)["id"].to_list())
        gaps_acum.append(len(seen_ids - step_ids))
        seen_ids.update(step_ids)

    return gaps_acum


def compute_summary_row(
    df: pl.DataFrame, subdir: str, config: str, n_total_list: list[int]
) -> SummaryRow | None:
    """Construye una fila de resumen para la tabla comparativa.

    Args:
        df: DataFrame crudo con columna "type" para separar cama de proyectil.
        subdir: Nombre de la variación.
        config: Nombre del filtro.
        n_total_list: Conteos de partículas por step.

    Returns:
        Fila de resumen o None si el DataFrame está vacío.
    """
    if df.is_empty():
        return None

    bed = df.filter(pl.col("type") != 2)
    if bed.is_empty():
        return None

    n_ids = bed["id"].n_unique()
    n_gaps = _compute_total_gaps(bed)
    n_fluct = _compute_fluctuations(n_total_list)
    max_bed = max(n_total_list) if n_total_list else 0
    min_bed = min(n_total_list) if n_total_list else 0

    return SummaryRow(subdir, config, n_ids, n_gaps, n_fluct, max_bed, min_bed)


def _compute_total_gaps(bed: pl.DataFrame) -> int:
    """Cuenta partículas con aparición intermitente (gaps totales).

    Args:
        bed: DataFrame de partículas de cama con columnas ["id", "i"].

    Returns:
        Número de IDs que no aparecen en todos los steps entre su primero y último.
    """
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
    return gap_particles.shape[0]


def _compute_fluctuations(n_total_list: list[int]) -> int:
    """Cuenta disminuciones en el conteo de partículas entre steps consecutivos.

    Args:
        n_total_list: Conteos de partículas por step.

    Returns:
        Número de veces que el conteo disminuyó.
    """
    return sum(
        1 for j in range(1, len(n_total_list)) if n_total_list[j] < n_total_list[j - 1]
    )


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


def _plot_metric(
    all_data: dict[str, FilterMetrics],
    metric_name: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    """Genera un gráfico comparativo de una métrica para todas las configuraciones.

    Args:
        all_data: Dict con clave = "subdir/config", valor = métricas.
        metric_name: Atributo de FilterMetrics a graficar.
        ylabel: Etiqueta del eje Y.
        title: Título del gráfico.

    Returns:
        Figura de matplotlib con las series superpuestas.
    """
    fig, ax = plt.subplots(figsize=_PLOT_FIGSIZE)

    for key, data in all_data.items():
        if not data.steps:
            continue

        config = key.split("/")[-1]
        values = getattr(data, metric_name)
        if not values:
            continue

        ax.plot(
            data.steps,
            values,
            label=key,
            color=COLORS.get(config, "gray"),
            linestyle=LINESTYLES.get(config, "-"),
            linewidth=1.5,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=_LEGEND_FONTSIZE, loc="best")
    ax.grid(alpha=_GRID_ALPHA)

    return fig


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------


def _render_sidebar(
    available: dict[str, list[str]],
) -> tuple[list[str], str, list[str], list[str]]:
    """Renderiza controles de selección en la barra lateral.

    Args:
        available: Dict de subdirectorios y stems de CSVs descubiertos.

    Returns:
        Tupla (subdirs, simulación, filtros, variantes_tracking).
    """
    st.sidebar.header("Configuraciones")

    subdirs = list(available.keys())
    selected_subdirs = st.sidebar.multiselect(
        "Variaciones (radios × tracking)",
        options=subdirs,
        default=subdirs[:2],
        help="Selecciona las variaciones a comparar",
    )

    all_csv_names = [name for s in selected_subdirs for name in available.get(s, [])]
    sim_names = extract_sim_names(all_csv_names)
    config_names = extract_config_names(all_csv_names)
    tracking_variants = sorted({
        match.group(1)
        for name in all_csv_names
        if (match := re.search(r"_(con_tracking|sin_tracking)$", name))
    })

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

    selected_variants = st.sidebar.multiselect(
        "Tracking",
        options=tracking_variants,
        default=tracking_variants,
        help="Selecciona variantes de tracking",
    )

    return selected_subdirs, selected_sim, selected_configs, selected_variants


def _load_all_data(
    results_dir: Path,
    subdirs: list[str],
    sim_name: str,
    configs: list[str],
    variants: list[str],
    available: dict[str, list[str]],
) -> dict[str, FilterMetrics]:
    """Carga y procesa todos los CSVs para la selección actual.

    Args:
        results_dir: Directorio raíz de resultados.
        subdirs: Subdirectorios a incluir.
        sim_name: Nombre de la simulación.
        configs: Filtros a incluir.
        variants: Variantes de tracking a incluir.
        available: Dict de subdirectorios y stems descubiertos.

    Returns:
        Dict con clave = "subdir/config_variant", valor = métricas.
    """
    all_data: dict[str, FilterMetrics] = {}

    for subdir in subdirs:
        csv_stems = available.get(subdir, [])
        for stem in csv_stems:
            parsed = _parse_stem(stem)
            if parsed is None:
                continue
            stem_sim, stem_config, stem_variant = parsed
            if stem_sim != sim_name:
                continue
            if stem_config not in configs:
                continue
            if variants and stem_variant not in variants:
                continue

            df = load_csv(results_dir, subdir, stem)
            if df is not None:
                label = f"{subdir}/{stem_config}"
                if stem_variant:
                    label = f"{label}_{stem_variant}"
                all_data[label] = compute_metrics_from_df(df)

    return all_data


def _render_plots(all_data: dict[str, FilterMetrics]) -> None:
    """Renderiza los tres gráficos comparativos en columnas.

    Args:
        all_data: Métricas cargadas para todas las configuraciones.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = _plot_metric(
            all_data, "n_total", "Partículas", "N partículas erosionadas"
        )
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        has_clusters = any(data.n_clusters for data in all_data.values())
        if has_clusters:
            fig = _plot_metric(
                all_data, "n_clusters", "N clusters", "Clusters por step"
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Sin datos de clusters para esta selección.")

    with col3:
        fig = _plot_metric(
            all_data, "gaps_acum", "Gaps acumulados", "IDs que desaparecen"
        )
        st.pyplot(fig)
        plt.close(fig)


def _render_summary_table(
    results_dir: Path,
    subdirs: list[str],
    configs: list[str],
    all_data: dict[str, FilterMetrics],
    sim_name: str,
    available: dict[str, list[str]],
) -> None:
    """Renderiza la tabla resumen con métricas globales por configuración.

    Args:
        results_dir: Directorio raíz de resultados.
        subdirs: Subdirectorios incluidos.
        configs: Filtros incluidos.
        all_data: Métricas temporales ya calculadas.
        sim_name: Nombre de la simulación.
        available: Dict de subdirectorios y stems descubiertos.
    """
    summary_data: list[SummaryRow] = []

    for subdir in subdirs:
        csv_stems = available.get(subdir, [])
        for stem in csv_stems:
            parsed = _parse_stem(stem)
            if parsed is None:
                continue
            stem_sim, stem_config, stem_variant = parsed
            if stem_sim != sim_name or stem_config not in configs:
                continue

            label = f"{subdir}/{stem_config}"
            if stem_variant:
                label = f"{label}_{stem_variant}"
            if label not in all_data:
                continue

            df = load_csv(results_dir, subdir, stem)
            if df is None:
                continue

            row = compute_summary_row(df, subdir, stem_config, all_data[label].n_total)
            if row is not None:
                summary_data.append(row)

    if summary_data:
        st.dataframe(
            pl.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit.

    Orquesta el flujo: configuración → descubrimiento → selección →
    carga de datos → visualización → tabla resumen.
    """
    st.title("Comparación de Filtros de Erosión")

    # --- Selector de directorio ---
    default_dir = str(_DEFAULT_RESULTS_DIR)
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

    # --- Selección del usuario ---
    selected_subdirs, selected_sim, selected_configs, selected_variants = (
        _render_sidebar(available)
    )

    if not selected_subdirs or not selected_configs or not selected_variants:
        st.warning("Selecciona al menos una variación, un filtro y una variante.")
        return

    # --- Cargar datos ---
    all_data = _load_all_data(
        results_dir,
        selected_subdirs,
        selected_sim,
        selected_configs,
        selected_variants,
        available,
    )

    if not all_data:
        st.warning("No se encontraron datos para la selección actual.")
        return

    # --- Visualización ---
    _render_plots(all_data)

    # --- Tabla resumen ---
    st.header("Resumen")
    _render_summary_table(
        results_dir, selected_subdirs, selected_configs, all_data, selected_sim, available
    )

    # --- Info ---
    st.sidebar.header("Info")
    st.sidebar.write(f"Subdirectorios: {len(available)}")
    all_csv_names = [name for s in available for name in available[s]]
    st.sidebar.write(f"Configuraciones: {len(extract_config_names(all_csv_names))}")
    st.sidebar.write(f"Simulaciones: {len(extract_sim_names(all_csv_names))}")


if __name__ == "__main__":
    main()
