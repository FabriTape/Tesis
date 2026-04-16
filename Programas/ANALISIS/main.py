import matplotlib.pyplot as plt
from confidence_intervals import run_confidence_intervals
from constants import CSV_PATH, FF_VALUES, SALIDA_DIR
from erosion_viz import plot_erosion_boxplot, plot_erosion_heatmap
from kinematics_calc import get_grid_pairs
from matplotlib.backends.backend_pdf import PdfPages
from pandas import read_csv


def save_figure(fig: plt.Figure, path: str) -> None:
    if fig and fig.get_axes():
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def run_analysis() -> None:
    print("Cargando datos...")
    df = read_csv(CSV_PATH)

    grid_pairs = get_grid_pairs(df)
    print(f"Datos: {len(df)} filas. Grid: {len(grid_pairs)} pares\n")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    figuras = run_confidence_intervals()
    valores_vel0 = sorted(df["vel0"].unique())

    for ff in FF_VALUES:
        dir_ff = SALIDA_DIR / f"ff_{ff}"
        dir_ff.mkdir(parents=True, exist_ok=True)

        if fig_box := plot_erosion_boxplot(df, ff):
            figuras.append(fig_box)
            save_figure(fig_box, dir_ff / "ids_boxplot.png")

        for vel0 in valores_vel0:
            if fig_heat := plot_erosion_heatmap(df, ff, vel0):
                figuras.append(fig_heat)
                save_figure(fig_heat, dir_ff / f"heatmap_erosion_{vel0}.png")

    with PdfPages(SALIDA_DIR / "resultado_final.pdf") as pdf:
        for fig in figuras:
            if fig and fig.get_axes():
                pdf.savefig(fig)
    print(f"¡Listo! Figuras en {SALIDA_DIR}")


if __name__ == "__main__":
    run_analysis()
