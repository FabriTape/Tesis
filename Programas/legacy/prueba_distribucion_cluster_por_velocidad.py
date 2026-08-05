"""Prueba: distribución de tamaños de cluster por velocidad.

Explora si la distribución de tamaños de los clusters eyectados en cada
impacto (PDF y CCDF) sigue una ley de potencia con exponente -2.8.

Por cada velocidad de impacto y cada caso (N_p, ángulo) se grafica la
distribución local P(n) = F(n)_j / M_t, con relleno de ceros para los
tamaños no observados, y se superpone una curva de referencia
y ~ n^-2.8 en escala log-log.

Genera dos figuras:
    plot_PDF_por_vel_cluster.png
    plot_CCDF_por_vel_cluster.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def grilla(np, df):
    """Construye una grilla (cuadrícula) de celdas [grid_x, grid_y] para el
    caso con N_p proyectiles.

    Args:
        np: número de proyectiles del caso a considerar.
        df: DataFrame completo de la simulación.

    Returns:
        base_grid: DataFrame indexado por (grid_x, grid_y) con la columna
            booleana "existe" (True si esa celda fue alcanzada por algún
            proyectil).
        data: df filtrado al caso "np", con las columnas extra grid_x/grid_y.
    """
    data = df[df["np"] == np].copy()

    # Paso de la grilla = mínima separación positiva entre posiciones
    step = data[data["posx"] > 0]["posx"].min()
    # Convertimos cada posición continua (posx, posy) a índices enteros de grilla
    data["grid_x"] = (data["posx"] / step).round().astype(int)
    data["grid_y"] = (data["posy"] / step).round().astype(int)

    x_min = int(data["grid_x"].min())
    y_min = int(data["grid_y"].min())
    x_max = int(data["grid_x"].max())
    y_max = int(data["grid_y"].max())

    # Celdas realmente ocupadas por al menos un proyectil
    reales = (
        data[["grid_x", "grid_y"]]
        .drop_duplicates()
        .assign(existe=True)
        .set_index(["grid_x", "grid_y"])
    )

    # Grilla rectangular completa que contiene a todas las celdas ocupadas
    x_r, y_r = range(x_min, x_max + 1), range(y_min, y_max + 1)
    base_grid = (
        pd.MultiIndex.from_product([x_r, y_r], names=["grid_x", "grid_y"])
        .to_frame(index=False)
        .set_index(["grid_x", "grid_y"])
    )
    # Marcamos "existe" solo donde hay datos reales
    base_grid = base_grid.join(reales).fillna(False).infer_objects(copy=False)
    return base_grid, data


def plot_cluster_size_distribution(casos, df_dist, df):
    """Grafica la distribución de tamaños de cluster por velocidad.

    Args:
        casos: lista de tuplas (n_proyectiles, ángulo).
        df_dist: DataFrame con un cluster por fila (columnas posx, posy, np,
            angulo, ff, count, v, ...).
        df: DataFrame completo de la simulación (usado para la grilla).
    """

    def _prepare():
        """Agrupa los clusters por impacto y rellena con ceros las
        combinaciones (posx, posy, phi, count, v) no observadas.

        Returns:
            dfs: lista (una por caso) de DataFrames con las columnas
                posx, posy, np, angulo, ff, count, v, F(n)_j y M_t.
        """
        dfs = []
        for n, alfa in casos:
            subset = df_dist[(df_dist["angulo"] == alfa) & (df_dist["np"] == n)].copy()
            df_plot = subset.rename(columns={"ff": r"$\phi$"})

            # F(n)_j: número de clusters de tamaño n eyectados en el impacto j
            # (identificado por posx, posy), para cada velocidad v.
            df_por_impacto = (
                df_plot.groupby(["posx", "posy", "np", "angulo", r"$\phi$", "count", "v"])
                .size()
                .reset_index(name="F(n)_j")
            )
            # Todos los tamaños de cluster que aparecen en el caso
            df_tamaños_unicos = sorted(df_plot["count"].unique())

            # M_t: total de clusters eyectados en el impacto j (todas las tallas).
            # Nota: este cálculo previo se recalcula después del relleno con
            # ceros; aquí queda como referencia.
            df_por_impacto["M_t"] = df_por_impacto.groupby(["posx", "posy", r"$\phi$", "v"])[
                "F(n)_j"
            ].transform("sum")

            # Grilla de celdas alcanzadas en este caso
            base_grid, df_grid = grilla(n, df)
            grid_existe = df_grid[["posx", "posy"]].drop_duplicates()

            # Producto cartesiano: (celdas reales) x (phi) x (tamaños) x (v)
            # para que ninguna combinación falte en el DataFrame.
            pos_phi = grid_existe[["posx", "posy"]].merge(
                df_plot[[r"$\phi$"]].drop_duplicates(), how="cross"
            )
            all_sizes = pd.DataFrame({"count": df_tamaños_unicos})
            all_combos = pos_phi.merge(all_sizes, how="cross")
            all_combos = all_combos.merge(
                df_plot[["v"]].drop_duplicates(), how="cross"
            )

            # Left join: las combinaciones sin datos quedan con NaN -> 0
            df_por_impacto = all_combos.merge(
                df_por_impacto, how="left", on=["posx", "posy", r"$\phi$", "count", "v"]
            )
            df_por_impacto = df_por_impacto.sort_values(
                ["posx", "posy", "v", r"$\phi$", "count"]
            )

            # Relleno de ceros: tamaños no observados valen 0
            df_por_impacto["F(n)_j"] = df_por_impacto["F(n)_j"].fillna(0)
            # Recalcular M_t después del relleno (ahora incluye los ceros)
            df_por_impacto["M_t"] = df_por_impacto.groupby(["posx", "posy", r"$\phi$", "v"])[
                "F(n)_j"
            ].transform("sum")
            dfs.append(df_por_impacto)
        return dfs

    def _panel(df_por_impacto, y_col, ax, n, alfa, yscale="symlog"):
        """Dibuja un panel individual (una velocidad x un caso).

        Args:
            df_por_impacto: datos del caso/velocidad actual.
            y_col: columna a graficar ("Prob_local" o "CCDF").
            ax: ejes de matplotlib donde dibujar.
            n, alfa: caso (N_p, ángulo) para el título.
            yscale: escala del eje y ("symlog" por defecto).

        Returns:
            handles, labels: leyenda del panel (para reutilizarla a nivel figura).
        """
        palette = sns.color_palette("tab10")

        # Datos puntuales con barras de error (error estándar de la media,
        # calculado sobre los distintos impactos de esa celda).
        sns.lineplot(
            data=df_por_impacto,
            x="count",
            y=y_col,
            errorbar="se",
            palette="tab10",
            hue=r"$\phi$",
            marker="o",
            markersize=8,
            linestyle="-",
            err_style="bars",
            err_kws={"capsize": 5},
            ax=ax,
        )

        # Curva de referencia de ley de potencia con exponente -2.8:
        # se ancla al primer punto observado (x0, y0) y se extrapola
        # y = y0 * (x/x0)^-2.8 sobre el rango de tamaños del panel.
        phi_vals = sorted(df_por_impacto[r"$\phi$"].unique())
        mean_df = df_por_impacto.groupby(["count", r"$\phi$"])[y_col].mean().reset_index()
        for idx, phi_val in enumerate(phi_vals):
            phi_data = mean_df[mean_df[r"$\phi$"] == phi_val].sort_values("count")
            mask = phi_data[y_col] > 0
            if mask.any():
                first = phi_data[mask].iloc[0]
                x0, y0 = first["count"], first[y_col]
                x_max = phi_data["count"].max()
                x_trend = np.logspace(np.log10(x0), np.log10(x_max), 100)
                y_trend = y0 * (x_trend / x0) ** (-2.8)
                ax.plot(x_trend, y_trend, "--", color=palette[idx], alpha=0.5, linewidth=1.5)

        # Escalas logarítmicas para visualizar la ley de potencia
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.set_yscale(yscale)
        ax.set_ylim(1e-2, 1)
        ax.set_xlabel("n", fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.grid(which="minor", linestyle=":", alpha=0.3)
        ax.minorticks_on()
        # Quitamos la leyenda del panel (va unificada en la figura)
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        return handles, labels

    # Preparación de los datos (relleno con ceros) para todos los casos
    dfs = _prepare()

    # Velocidades presentes (una fila de paneles por velocidad)
    velocities = sorted(set(v for df in dfs for v in df["v"].unique()))
    n_vel = len(velocities)

    # Generamos una figura por variable: PDF y CCDF
    for y_col, ylabel, suffix in [
        ("Prob_local", r"$P(n)$", "PDF"),
        ("CCDF", r"$1 - F(n)/N$", "CCDF"),
    ]:
        # Grilla: filas = velocidades, columnas = casos
        fig, axes = plt.subplots(
            n_vel, 3,
            figsize=(15, 5 * n_vel),
            sharex=True, sharey=True, squeeze=False,
        )

        legend_h, legend_l = [], []
        for i, v in enumerate(velocities):
            for j, (n, alfa) in enumerate(casos):
                ax = axes[i, j]
                df_v = dfs[j][dfs[j]["v"] == v].copy()
                if df_v.empty:
                    ax.set_visible(False)
                    continue

                # PDF local: P(n) = F(n)_j / M_t  (probabilidad por impacto)
                if y_col == "Prob_local":
                    df_v["Prob_local"] = np.where(
                        df_v["M_t"] > 0,
                        df_v["F(n)_j"] / df_v["M_t"],
                        0.0,
                    )
                # CCDF: 1 - (suma acumulada de F(n)_j hasta n) / M_t
                else:
                    df_v["M_j"] = (
                        df_v.sort_values("count")
                        .groupby(["posx", "posy", r"$\phi$"])["F(n)_j"]
                        .transform(lambda x: x.cumsum())
                    )
                    df_v["CCDF"] = np.where(
                        df_v["M_t"] > 0,
                        1 - df_v["M_j"] / df_v["M_t"],
                        1.0,
                    )

                yscale = "symlog"
                h, lbs = _panel(df_v, y_col, ax, n, alfa, yscale)
                # Guardamos la leyenda de la primera figura para reutilizarla
                if h and not legend_h:
                    legend_h, legend_l = h, lbs

                # Título de columna (solo en la primera fila)
                if i == 0:
                    if n == 1 and alfa == 0:
                        ax.set_title(rf"$N_p$: {n}", fontsize=16)
                    elif n == 3 and alfa == 0:
                        ax.set_title(rf"$N_p$: {n} H", fontsize=16)
                    else:
                        ax.set_title(rf"$N_p$: {n} V", fontsize=16)
                # Etiqueta de fila (velocidad) solo en la primera columna
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n{ylabel}", fontsize=16)
                else:
                    ax.set_ylabel("")

        # Leyenda única arriba de la figura
        if legend_h:
            fig.legend(
                legend_h, legend_l,
                loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.02 * n_vel),
                ncol=len(legend_h), fontsize=12,
            )
        fig.tight_layout()
        fig.savefig(f"plot_{suffix}_por_vel_cluster.png", bbox_inches="tight")
        plt.close(fig)
