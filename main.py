"""Script principal de análisis de datos LAMMPS.

Orquesta la generación de visualizaciones para el análisis de erosión
en simulaciones de impacto proyectil-cama.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data_loader import load_data
from constants import SALIDA_DIR, FF_VALUES
from cluster_viz import plot_cluster_frequency, plot_cluster_ccdf
from erosion_viz import plot_erosion_boxplot, plot_erosion_heatmap
from kinematics_viz import plot_velocity_angle, plot_kinetic_energy, plot_penetration
from params_viz import plot_params_vs_energy


def save_figure(fig: plt.Figure, path: str) -> None:
    """Guarda una figura en archivo PNG.

    Args:
        fig: Figura de matplotlib.
        path: Ruta del archivo de salida.
    """
    if fig and fig.get_axes():
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def run_analysis() -> None:
    """Ejecuta el pipeline completo de análisis."""
    print("Cargando datos...")
    df = load_data()
    print(f"Datos cargados: {len(df)} filas")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    valores_vel0 = sorted(df["vel0"].unique())
    figuras: list[plt.Figure] = []

    for ff in FF_VALUES:
        dir_ff = SALIDA_DIR / f"ff_{ff}"
        dir_ff.mkdir(parents=True, exist_ok=True)

        print(f"Procesando ff={ff}...")

        # Boxplot de erosión
        fig_box = plot_erosion_boxplot(df, ff)
        if fig_box and fig_box.get_axes():
            figuras.append(fig_box)
            save_figure(fig_box, dir_ff / "ids_boxplot.png")

        # Frecuencia vs tamaño de cluster
        fig_freq = plot_cluster_frequency(df, ff)
        if fig_freq and fig_freq.get_axes():
            figuras.append(fig_freq)
            save_figure(fig_freq, dir_ff / "ids_por_vel0.png")

        # Por cada vel0
        for vel0 in valores_vel0:
            print(f"  vel0={vel0}...")

            # CCDF de clusters
            figs_ccdf = plot_cluster_ccdf(df, ff, vel0)
            for i, fig_cc in enumerate(figs_ccdf):
                if fig_cc and fig_cc.get_axes():
                    figuras.append(fig_cc)
                    if i == 0:
                        save_figure(fig_cc, dir_ff / f"cluster_ccdf_{vel0}.png")
                    else:
                        plt.close(fig_cc)

            # Ángulo de velocidad
            fig_ang = plot_velocity_angle(df, ff, vel0)
            if fig_ang and fig_ang.get_axes():
                figuras.append(fig_ang)
                save_figure(fig_ang, dir_ff / f"angulo_vel_cm_{vel0}.png")

            # Heatmap de erosión
            fig_heat = plot_erosion_heatmap(df, ff, vel0)
            if fig_heat and fig_heat.get_axes():
                figuras.append(fig_heat)
                save_figure(fig_heat, dir_ff / f"heatmap_erosion_{vel0}.png")

        # Energía cinética
        fig_ek = plot_kinetic_energy(df, ff)
        if fig_ek and fig_ek.get_axes():
            figuras.append(fig_ek)
            save_figure(fig_ek, dir_ff / "energia_km.png")

        # Penetración
        fig_pen = plot_penetration(df, ff)
        if fig_pen and fig_pen.get_axes():
            figuras.append(fig_pen)
            save_figure(fig_pen, dir_ff / "penetracion.png")

        # Parámetros vs energía
        fig_params = plot_params_vs_energy(df, ff)
        if fig_params and fig_params.get_axes():
            figuras.append(fig_params)
            save_figure(fig_params, dir_ff / "parametros_vs_energia.png")

    # PDF final
    pdf_final = SALIDA_DIR / "resultado_final.pdf"
    print(f"Generando PDF final: {pdf_final}")
    with PdfPages(pdf_final) as pdf:
        for fig in figuras:
            if fig and fig.get_axes():
                pdf.savefig(fig)

    print(f"¡Listo! Figuras en {SALIDA_DIR}")


if __name__ == "__main__":
    run_analysis()