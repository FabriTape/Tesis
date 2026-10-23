from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

CSV_PATH = Path("resultados/merge_resultados.csv").expanduser()


def plot_heatmap(sim, ff, v, base_grid):
    i_max = sim["i"].max()
    mask = (sim["i"] == i_max) & ((sim["type"] == 2) & (sim["z"] > 1e-4)) | (sim["type"] == 1)
    yields = sim[mask].groupby(["x", "y"])["id"].nunique().rename("yield")

    res = base_grid.join(yields)
    res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0

    valores_reales = res.loc[res["existe"], "yield"].reset_index(drop=True)

    # Generar Heatmap
    matrix = res["yield"].unstack(level="x").sort_index(ascending=False)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        square=True,
        cbar_kws={"label": "Yield"},
    )
    plt.title(f"Erosión (ff={ff}, v={v})")
    plt.xlabel(r"posx ($\sigma /2$)", fontsize=12)
    plt.ylabel(r"posy ($\sigma /2$)", fontsize=12)
    plt.savefig(f"resultados/heatmap_ff_{ff}_v{v}.png", bbox_inches="tight")
    return pd.DataFrame({"vel0": abs(v), "yield": valores_reales})


def histograma_vels(df, v, color, ax, i):
    subset = df[df["vel0"] == v]
    sns.histplot(data=subset, x="yield", discrete=True, color=color, ax=ax)
    ax.set_title(f"v = {v}", fontsize=14)
    ax.set_xlabel(r"$Y$", fontsize=12)
    if i == 0:
        ax.set_ylabel("Frecuencia")
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_erosion_distribution(df, vels, ff, df_dist):
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, len(vels), height_ratios=[1, 1.2], figure=fig)

    # 1. Boxplot (Ocupa toda la fila superior)
    ax_box = fig.add_subplot(gs[0, :])
    sns.boxplot(data=df_dist, x="vel0", y="yield", ax=ax_box, palette="tab10", hue=df_dist["vel0"], legend=False)
    ax_box.set_title(
        fr"Distribución de Erosión ($\phi$={ff})", fontsize=16
    )
    ax_box.set_xlabel(r"$v$", fontsize=16)
    ax_box.set_ylabel(r"$Y$", fontsize=16)
    ax_box.grid(axis="y", linestyle="--", alpha=0.7)
    return gs, fig


def main():
    df = pd.read_csv(CSV_PATH)

    # 1. Grilla Global Vectorizada
    step = df[df["posx"] > 0]["posx"].min()
    df["x"] = df["posx"].div(step)
    df["y"] = df["posy"].div(step)

    x_min = int(df["x"].min())
    y_min = int(df["y"].min())
    x_max = int(df["x"].max())
    y_max = int(df["y"].max())

    reales = df[["x", "y"]].drop_duplicates().assign(existe=True).set_index(["x", "y"])
    x_r, y_r = range(x_min, x_max + 1), range(y_min, y_max + 1)
    base_grid = (
        pd.MultiIndex.from_product([x_r, y_r], names=["x", "y"])
        .to_frame(index=False)
        .set_index(["x", "y"])
    )
    base_grid = base_grid.join(reales).fillna(False)

    # 2. Análisis iterativo
    for ff in [0.15, 0.35]:
        datos_distribucion = []
        vels = sorted(df.abs().loc[df["ff"] == ff, "vel0"].unique())

        for v in vels:
            sim = df[(df["ff"] == ff) & (df["vel0"] == -1*v)]
            if not sim.empty:
                datos_distribucion.append(plot_heatmap(sim, ff, v, base_grid))
                plt.close()
        if datos_distribucion:
            df_dist = pd.concat(datos_distribucion)

            gs, fig = plot_erosion_distribution(df_dist, vels, ff, df_dist)
            # 2. Histogramas Side-by-Side (Fila inferior)
            paleta = sns.color_palette(
                "viridis", n_colors=len(vels)
            )  # Mismos colores que el boxplot

#            for i, v in enumerate(vels):
#                histograma_vels(df_dist, v, paleta[i], fig.add_subplot(gs[1, i]), i)

            plt.tight_layout()
            plt.savefig(f"resultados/distribucion_ff_{ff}.png", bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
    print("exito")
