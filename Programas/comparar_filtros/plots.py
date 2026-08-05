"""Visualización comparativa de filtros de erosión."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_STEP_TO_US = 1.25
_AXIS_FONTSIZE = 14


def plot_comparison(
    all_metrics: dict[str, list[dict]],
    all_gaps: dict[str, list[dict]],
    all_gaps_raw: dict[str, list[dict]] | None = None,
    sim_name: str = "",
    output_dir: Path | None = None,
    title_suffix: str = "",
) -> None:
    """Genera plots comparativos de las 6 configuraciones de filtro.

    Args:
        all_metrics: Dict con clave = nombre config (F1I, F1C, ...)
            y valor = lista de métricas por step.
        all_gaps: Dict con clave = nombre config y valor = lista de gaps
            acumulados por step.
        all_gaps_raw: Dict con clave = nombre config y valor = lista de gaps
            por step (no acumulados).
        sim_name: Nombre de la simulación (para título).
        output_dir: Directorio de salida para los SVG.
        title_suffix: Sufijo para el título del plot.
    """
    if output_dir is None:
        return

    colors = {
        "F1I": "#e74c3c",
        "F1C": "#e74c3c",
        "F2I": "#3498db",
        "F2C": "#3498db",
        "F3I": "#2ecc71",
        "F3C": "#2ecc71",
    }
    linestyles = {
        "F1I": "-",
        "F1C": "--",
        "F2I": "-",
        "F2C": "--",
        "F3I": "-",
        "F3C": "--",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.90, right=0.82, wspace=0.28)
    fig.suptitle(
        f"Comparación de filtros — {sim_name.replace('_', ' ')} — {title_suffix.replace('_', ' ')}",
        fontsize=14,
    )

    # Plot (a): n_bed vs tiempo
    ax = axes[0, 0]
    for config, metrics in all_metrics.items():
        time_us = [m["step"] * _STEP_TO_US for m in metrics]
        n_bed = [m["n_total"] for m in metrics]
        ax.plot(
            time_us,
            n_bed,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Y", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02,
        0.98,
        "(a)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.grid(alpha=0.3)

    # Plot (b): n_clusters vs tiempo
    ax = axes[0, 1]
    for config, metrics in all_metrics.items():
        time_us = [m["step"] * _STEP_TO_US for m in metrics]
        n_clusters = [m["n_clusters"] for m in metrics]
        ax.plot(
            time_us,
            n_clusters,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("N clusters", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02,
        0.98,
        "(b)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.grid(alpha=0.3)

    # Plot (c): gaps acumulados vs tiempo
    ax = axes[1, 0]
    for config, gaps_data in all_gaps_raw.items():
        if not gaps_data:
            continue
        time_us = [g["step"] * _STEP_TO_US for g in gaps_data]
        gaps = np.cumulative_sum([g["gaps"] for g in gaps_data])
        ax.plot(
            time_us,
            gaps,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("fluctuaciones acumuladas", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02,
        0.98,
        "(c)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.grid(alpha=0.3)

    # Plot (d): gaps por step vs tiempo
    ax = axes[1, 1]
    if all_gaps_raw:
        for config, gaps_data in all_gaps_raw.items():
            if not gaps_data:
                continue
            time_us = [g["step"] * _STEP_TO_US for g in gaps_data]
            gaps = [g["gaps"] for g in gaps_data]
            ax.plot(
                time_us,
                gaps,
                label=config,
                color=colors.get(config, "gray"),
                linestyle=linestyles.get(config, "-"),
                linewidth=1.5,
            )
        ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
        ax.set_ylabel("fluctuaciones", fontsize=_AXIS_FONTSIZE)
        ax.tick_params(labelsize=_AXIS_FONTSIZE)
        ax.text(
            0.02,
            0.98,
            "(d)",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Sin datos de gaps por step", ha="center", va="center")

    # Leyenda única a la derecha, arriba en dos columnas
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.84, 0.92),
        fontsize=12,
        ncol=2,
        frameon=True,
    )

    output_path = (
        output_dir / f"comparacion_{sim_name}_{title_suffix.replace(' ', '_')}.svg"
    )
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Plot guardado: {output_path}")


def plot_summary_table(
    summaries: dict[str, dict],
    sim_name: str,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Genera una tabla resumen en formato SVG.

    Args:
        summaries: Dict con clave = config, valor = {gaps, fluct, n_ids, ...}.
        sim_name: Nombre de la simulación (para nombre de archivo).
        output_dir: Directorio de salida.
    """
    configs = list(summaries.keys())
    data = []
    for c in configs:
        s = summaries[c]
        data.append(
            [
                c,
                str(s.get("n_ids", 0)),
                str(s.get("gaps", 0)),
                str(s.get("fluct", 0)),
                str(s.get("n_clusters_final", 0)),
            ]
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=["Config", "IDs", "Gaps", "Fluct", "Clusters"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title(f"Resumen — {sim_name} — {title_suffix}", fontsize=14, pad=20)

    plt.tight_layout()
    output_path = (
        output_dir / f"tabla_resumen_{sim_name}_{title_suffix.replace(' ', '_')}.svg"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tabla guardada: {output_path}")
