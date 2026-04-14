"""Script principal de análisis de datos LAMMPS.

Orquesta la generación de visualizaciones para el análisis de erosión
en simulaciones de impacto proyectil-cama.
"""

import matplotlib.pyplot as plt
from confidence_intervals import run_confidence_intervals
from constants import FF_VALUES, SALIDA_DIR
from data_loader import load_data
from erosion_viz import plot_erosion_boxplot, plot_erosion_heatmap
from kinematics_calc import get_grid_pairs
from matplotlib.backends.backend_pdf import PdfPages


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

    # I_VALUE dinámico: usar el paso temporal máximo del dataset
    I_VALUE = 80
    print(f"I_VALUE dinámico: {I_VALUE}\n")

    # Grid de posiciones (calculado una vez para todas las simulaciones)
    grid_pairs = get_grid_pairs(df)
    print(f"Grid de posiciones: {len(grid_pairs)} pares\n")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    # Tablas de intervalos de confianza (van primero en el PDF)
    print("Generando tablas de intervalos de confianza...")
    figuras = run_confidence_intervals(I_VALUE)
    print(f"Tablas de IC generadas: {len(figuras)}\n")

    valores_vel0 = sorted(df["vel0"].unique())

    for ff in FF_VALUES:
        dir_ff = SALIDA_DIR / f"ff_{ff}"
        dir_ff.mkdir(parents=True, exist_ok=True)

        print(f"Procesando ff={ff}...")

        # Boxplot de erosión
        fig_box = plot_erosion_boxplot(df, ff)
        if fig_box and fig_box.get_axes():
            figuras.append(fig_box)
            save_figure(fig_box, dir_ff / "ids_boxplot.png")

        # Por cada vel0
        for vel0 in valores_vel0:
            print(f"  vel0={vel0}...")

            # Heatmap de erosión
            fig_heat = plot_erosion_heatmap(df, ff, vel0, grid_pairs)
            if fig_heat and fig_heat.get_axes():
                figuras.append(fig_heat)
                save_figure(fig_heat, dir_ff / f"heatmap_erosion_{vel0}.png")

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
