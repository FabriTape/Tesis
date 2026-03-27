"""Visualización comparativa de filtros de erosión."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_comparison(
    all_metrics: dict[str, list[dict]],
    sim_name: str,
    output_dir: Path,
) -> None:
    """Genera plots comparativos de las 6 configuraciones de filtro.

    Args:
        all_metrics: Dict con clave = nombre config (F1I, F1C, ...)
            y valor = lista de métricas por step.
        sim_name: Nombre de la simulación (para título).
        output_dir: Directorio de salida para los SVG.
    """
    colors = {
        "F1I": "#e74c3c",
        "F1C": "#e74c3c",
        "F2I": "#3498db",
        "F2C": "#3498db",
        "F3I": "#2ecc71",
        "F3C": "#2ecc71",
    }
    linestyles = {"F1I": "-", "F1C": "--", "F2I": "-", "F2C": "--", "F3I": "-", "F3C": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparación de filtros — {sim_name}", fontsize=14)

    # Plot 1: n_bed vs step
    ax = axes[0, 0]
    for config, metrics in all_metrics.items():
        steps = [m["step"] for m in metrics]
        n_bed = [m["n_bed"] for m in metrics]
        ax.plot(
            steps,
            n_bed,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Partículas de cama")
    ax.set_title("N partículas erosionadas por step")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Plot 2: n_clusters vs step
    ax = axes[0, 1]
    for config, metrics in all_metrics.items():
        steps = [m["step"] for m in metrics]
        n_clusters = [m["n_clusters"] for m in metrics]
        ax.plot(
            steps,
            n_clusters,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("N clusters")
    ax.set_title("Número de clusters por step")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Plot 3: z_max vs step
    ax = axes[1, 0]
    for config, metrics in all_metrics.items():
        steps = [m["step"] for m in metrics]
        z_max = [m["z_max"] * 1e6 for m in metrics]
        ax.plot(
            steps,
            z_max,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("z_max [µm]")
    ax.set_title("Altura máxima del ejecta")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Plot 4: IDs únicos acumulados vs step
    ax = axes[1, 1]
    for config, metrics in all_metrics.items():
        steps = [m["step"] for m in metrics]
        ids = [m["n_unique_ids"] for m in metrics]
        ax.plot(
            steps,
            ids,
            label=config,
            color=colors.get(config, "gray"),
            linestyle=linestyles.get(config, "-"),
            linewidth=1.5,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("IDs únicos")
    ax.set_title("Partículas únicas acumuladas")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"comparacion_{sim_name}.svg"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot guardado: {output_path}")


def plot_summary_table(
    summaries: dict[str, dict],
    output_dir: Path,
) -> None:
    """Genera una tabla resumen en formato SVG.

    Args:
        summaries: Dict con clave = config, valor = {gaps, fluct, n_ids, ...}.
        output_dir: Directorio de salida.
    """
    configs = list(summaries.keys())
    data = []
    for c in configs:
        s = summaries[c]
        data.append([
            c,
            str(s.get("n_ids", 0)),
            str(s.get("gaps", 0)),
            str(s.get("fluct", 0)),
            str(s.get("n_clusters_final", 0)),
        ])

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

    ax.set_title("Resumen de configuraciones", fontsize=14, pad=20)

    plt.tight_layout()
    output_path = output_dir / "tabla_resumen.svg"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tabla guardada: {output_path}")
