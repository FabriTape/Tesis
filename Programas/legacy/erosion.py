"""Análisis de erosión para el set de 9 repeticiones.

Genera las figuras de la tesis: distribuciones de erosión y de tamaño de
cluster (PDF, CCDF, P(k)), heatmaps, violines, tablas LaTeX de profundidades
y ajustes de modelos (ley de potencia).

Uso:
    python erosion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("future.no_silent_downcasting", True)
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from uncertainties import unumpy as unp

MK_SIZE = 15
PHI_VALS = [0.15, 0.35]

FONT_AXES_LABEL = 20
FONT_TICK_LABEL = 13
FONT_LEGEND = 23
FONT_TITLE = 21
FONT_SUBPLOT_LABEL = 20
FONT_ANNOTATION = 20

CSV_PATH = Path("datos/resultados_9_mov/merge_resultados.csv").expanduser()

# Mapping de (np, ángulo) a marcadores
MARKER_STYLES = {
    (1, 0): "o",
    (3, 90): "^",
    (3, 0): "s",
}

COL_LABELS = ["(a)", "(b)", "(c)"]


def _marker_kwargs(n, angle):
    kwargs = {"marker": MARKER_STYLES[(n, angle)]}
    return kwargs


def _make_hollow_markers(ax):
    for line in ax.lines:
        line.set_markerfacecolor("none")
        line.set_markeredgecolor(line.get_color())
        line.set_markeredgewidth(3)
    for collection in ax.collections:
        edge = collection.get_edgecolors()
        color = collection.get_facecolors()
        collection.set_facecolor("none")
        collection.set_edgecolor(
            color if len(edge) == 0 or (len(edge) > 0 and edge[0, 3] == 0) else edge
        )
        collection.set_linewidth(1.5)


def _ff_marker_handles():
    palette = sns.color_palette("bright")
    markers = ["o", "^", "s"]
    case_labels = [r"$N_p: 1$", r"$N_p: 3 V$", r"$N_p: 3 H$"]
    handles, labels = [], []
    for phi_idx, phi_val in enumerate(PHI_VALS):
        handles.append(Line2D([0], [0], color=palette[phi_idx], linewidth=0))
        labels.append(rf"$\phi$={phi_val}")
    for marker, lbl in zip(markers, case_labels):
        for phi_idx, phi_val in enumerate(PHI_VALS):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=palette[phi_idx],
                    markerfacecolor="none",
                    markeredgewidth=3,
                    markersize=MK_SIZE+5,
                    linestyle="none",
                )
            )
            labels.append(lbl)
    return handles, labels


plt.rcParams.update(
    {
        "axes.labelsize": FONT_AXES_LABEL,
        "xtick.labelsize": FONT_TICK_LABEL,
        "ytick.labelsize": FONT_TICK_LABEL,
        "legend.fontsize": FONT_LEGEND,
    }
)


def plot_heatmap(sim, ff, v, np, angle, base_grid):
    i_max = sim["i"].max()
    erosion_mask = (sim["i"] == i_max) & (sim["proyectil_erosion"] != False)
    erosion_count = sim[erosion_mask].groupby(["grid_x", "grid_y"])["id"].nunique()
    yields = erosion_count.rename("yield")
    res = base_grid.join(yields)
    res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0

    valores_reales = res.loc[res["existe"], "yield"].reset_index(drop=True)

    # Generar Heatmap
    matrix = res["yield"].unstack(level="grid_x").sort_index(ascending=False)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        square=True,
        cbar_kws={"label": "Yield"},
    )
    plt.title(f"Erosión (ff={ff}, v={v}, np={np}, angle={angle})")
    plt.xlabel(r"posx ($\sigma /2$)")
    plt.ylabel(r"posy ($\sigma /2$)")
    plt.savefig(
        f"figuras/heatmap_ff_{ff}_v{v}_np{np}_angle{angle}.png", bbox_inches="tight"
    )
    return pd.DataFrame(
        {
            "vel0": abs(v),
            "yield": valores_reales,
            "ff": ff,
            "np": np,
            "angulo": angle,
        }
    )


def histograma_vels(df, v, i):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(1, 1, 1)
    paleta = sns.color_palette(
        "viridis", n_colors=len(v)
    )  # Mismos colores que el boxplot
    color = paleta[i]
    subset = df[df["vel0"] == v]
    sns.histplot(data=subset, x="yield", discrete=True, color=color, ax=ax)
    ax.set_title(f"v = {v} [m/s]", fontsize=FONT_SUBPLOT_LABEL)
    ax.set_xlabel(r"$Y$")
    if i == 0:
        ax.set_ylabel("Frecuencia")
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_erosion_loglog(casos, df_dist):
    palette = sns.color_palette("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for i, (n, alfa) in enumerate(casos):
        ax = axes[i]
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})
        df_plot[r"$v$"] = df_plot["vel0"].abs()

        m_kw = _marker_kwargs(n, alfa)
        sns.lineplot(
            data=df_plot,
            x=r"$v$",
            y="yield",
            errorbar="se",
            palette="tab10",
            hue=r"$\phi$",
            markersize=MK_SIZE,
            linestyle="",
            linewidth=3.8,
            err_style="bars",
            err_kws={"capsize": 5},
            ax=ax,
            **m_kw,
        )
        _make_hollow_markers(ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ff_vals = sorted(df_plot[r"$\phi$"].unique())
        for idx, ff in enumerate(ff_vals):
            ff_data = df_plot[df_plot[r"$\phi$"] == ff]
            x_fit = ff_data[r"$v$"].values
            y_fit = ff_data["yield"].values

            mask_fit = (y_fit > 0) & (x_fit > 0)
            x_fit, y_fit = x_fit[mask_fit], y_fit[mask_fit]

            if len(x_fit) < 3:
                continue

            try:
                popt, _ = curve_fit(
                    power_law,
                    x_fit,
                    y_fit,
                    p0=[1, 2, 1],
                    maxfev=5000,
                    bounds=([0, 0, 0], [np.inf, 5.1, 4]),
                )
                _, _, c = popt
                x_sm = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
                ax.plot(
                    x_sm,
                    power_law(x_sm, *popt),
                    ":",
                    color=palette[idx],
                    alpha=0.9,
                    linewidth=3.8,
                )
                ax.text(
                    0.97,
                    0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                    rf"$c={c:.2f}$",
                    transform=ax.transAxes,
                    color=palette[idx],
                    fontsize=FONT_ANNOTATION,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )
            except (RuntimeError, ValueError):
                pass

        _style_subplot(ax, n, alfa, i == 0, i)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylim(bottom=1e-10)
        ax.set_xlabel(r"$v$ [$m/s$]", fontsize=FONT_TITLE)
        ax.set_ylabel(r"$Y$", fontsize=FONT_TITLE)
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
    )
    phi_palette = sns.color_palette("bright")
    for i in range(len(PHI_VALS)):
        leg.get_texts()[i].set_color(phi_palette[i])


    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig("figuras/plot_erosion_loglog.png", bbox_inches="tight")
    plt.close(fig)


def plot_erosion_distribution(casos, df_dist):
    palette = sns.color_palette("bright")
    ff_vals = sorted(df_dist["ff"].unique())

    fig1, ax1 = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    filas = []

    for i, (n, alfa) in enumerate(casos):
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})
        df_plot[r"$v$"] = df_plot["vel0"].abs()
        m_kw = _marker_kwargs(n, alfa)
        sns.lineplot(
            data=df_plot,
            x=r"$v$",
            y="yield",
            errorbar="se",
            palette=palette[: len(ff_vals)],
            hue=r"$\phi$",
            markersize=MK_SIZE,
            markeredgewidth=3,
            linestyle="",
            linewidth=3.8,
            err_style="bars",
            err_kws={"capsize": 5},
            ax=ax1[i],
            **m_kw,
        )
        _make_hollow_markers(ax1[i])

        if ax1[i].get_legend() is not None:
            ax1[i].get_legend().remove()

        c_vals = {}
        for idx, ff in enumerate(ff_vals):
            ff_data = df_plot[df_plot[r"$\phi$"] == ff]
            x_fit = ff_data[r"$v$"].values
            y_fit = ff_data["yield"].values

            mask_fit = (y_fit > 0) & (x_fit > 0)
            x_fit, y_fit = x_fit[mask_fit], y_fit[mask_fit]

            if len(x_fit) < 2:
                popt = [np.nan, np.nan, np.nan]
                perr = [np.nan, np.nan, np.nan]
                a, b, c = np.nan, np.nan, np.nan
                c_vals[ff] = np.nan
                ang_s = "H" if alfa == 0 else "V"
                filas.append((n, ang_s, ff, *popt, *perr))
                continue

            try:
                popt, pcov = curve_fit(
                    power_law,
                    x_fit,
                    y_fit,
                    p0=[1, 0, 1],
                    maxfev=5000,
                    bounds=([0, -0.001, 0], [np.inf, 0, 4]),
                )
                dof = len(x_fit) - 3
                t_val = t.ppf(0.975, max(dof, 1))
                perr = np.sqrt(np.diag(pcov)) * t_val
                a, b, c = popt
                c_vals[ff] = c

                x_sm = np.linspace(0, x_fit.max(), 200)
                ax1[i].plot(
                    x_sm,
                    power_law(x_sm, *popt),
                    ":",
                    color=palette[idx],
                    alpha=0.9,
                    linewidth=3.8,
                )
                ax1[i].text(
                    0.97,
                    0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                    rf"$c={c:.2f}$",
                    transform=ax1[i].transAxes,
                    color=palette[idx],
                    fontsize=FONT_ANNOTATION,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )
            except (RuntimeError, ValueError):
                popt = [np.nan, np.nan, np.nan]
                perr = [np.nan, np.nan, np.nan]
                a, b, c = np.nan, np.nan, np.nan
                c_vals[ff] = np.nan

            ang_s = "H" if alfa == 0 else "V"
            filas.append((n, ang_s, ff, *popt, *perr))

        _style_subplot(ax1[i], n, alfa, i == 0, i)
        ax1[i].set_ylim(-2, 36.5)
        ax1[i].set_xlabel(r"$v$ [$m/s$]", fontsize=FONT_TITLE+5)
        ax1[i].set_ylabel(r"$Y$", fontsize=FONT_TITLE+5)
        ax1[i].tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles()
    leg = fig1.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
    )
    phi_palette = sns.color_palette("bright")
    for i in range(len(PHI_VALS)):
        leg.get_texts()[i].set_color(phi_palette[i])


    plt.tight_layout()
    fig1.subplots_adjust(bottom=0.2)
    fig1.savefig("figuras/plot_power_law_todos.png", bbox_inches="tight")
    plt.close(fig1)


def _style_subplot(ax, n, alfa, show_ylabel, col_idx):
    label = COL_LABELS[col_idx]
    if n == 1 and alfa == 0:
        ax.set_title(f"{label} $N_p$: 1", fontsize=FONT_TITLE)
    elif n == 3 and alfa == 0:
        ax.set_title(f"{label} $N_p$: 3 H", fontsize=FONT_TITLE)
    else:
        ax.set_title(f"{label} $N_p$: 3 V", fontsize=FONT_TITLE)
    ax.set_xlabel(r"v [$m/s$]", fontsize=FONT_SUBPLOT_LABEL)
    ax.set_ylabel(r"$Y$", fontsize=FONT_SUBPLOT_LABEL) if show_ylabel else ax.set_ylabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()


def plot_erosion_violin(casos, df_dist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, (n, alfa) in enumerate(casos):
        ax = axes[i]
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        subset = subset.rename(columns={"ff": r"$\phi$"})
        subset["vel_str"] = subset["vel0"].astype(int).astype(str)

        sns.violinplot(
            data=subset,
            x="vel_str",
            y="yield",
            hue=r"$\phi$",
            palette="tab10",
            ax=ax,
            split=True,
            inner="quartile",
            density_norm="width",
            bw_method=0.3,
        )

        label = COL_LABELS[i]
        if n == 1 and alfa == 0:
            ax.set_title(rf"{label} $N_p$: {n}", fontsize=FONT_TITLE)
        elif n == 3 and alfa == 0:
            ax.set_title(rf"{label} $N_p$: {n} V", fontsize=FONT_TITLE)
        else:
            ax.set_title(rf"{label} $N_p$: {n} H", fontsize=FONT_TITLE)
        ax.set_xlabel(r"$v$ [m/s]")
        if i == 0:
            ax.set_ylabel(r"$Y$")
        if ax.get_legend() is not None:
            if i != 2:
                ax.get_legend().remove()
    global_min = df_dist["yield"].min() - 1
    global_max = df_dist["yield"].max() + 1
    for ax in axes:
        ax.set_ylim(global_min, global_max)
    plt.tight_layout()
    plt.savefig("figuras/plot_violin_erosion.png", bbox_inches="tight")
    plt.close()


def _prepare_cluster_data(casos, df_dist, df):
    dfs = []
    for n, alfa in casos:
        subset = df_dist[(df_dist["angulo"] == alfa) & (df_dist["np"] == n)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})

        df_por_impacto = (
            df_plot.groupby(["posx", "posy", "np", "angulo", r"$\phi$", "count", "v"])
            .size()
            .reset_index(name="F(n)_j")
        )
        df_por_impacto["M_t"] = df_por_impacto.groupby(
            ["posx", "posy", r"$\phi$", "v"]
        )["F(n)_j"].transform("sum")
        dfs.append(df_por_impacto)
    return dfs


def _prepare_cluster_data_sin_relleno(casos, df_dist):
    dfs = []
    for n, alfa in casos:
        subset = df_dist[(df_dist["angulo"] == alfa) & (df_dist["np"] == n)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})

        df_por_impacto = (
            df_plot.groupby(["posx", "posy", "np", "angulo", r"$\phi$", "count", "v"])
            .size()
            .reset_index(name="F(n)_j")
        )
        df_por_impacto["M_t"] = df_por_impacto.groupby(
            ["posx", "posy", r"$\phi$", "v"]
        )["F(n)_j"].transform("sum")
        dfs.append(df_por_impacto)
    return dfs


def _plot_distribution_panel(df_por_impacto, y_col, ax, n, alfa):
    palette = sns.color_palette("tab10")
    m_kw = _marker_kwargs(n, alfa)
    m_kw["markerfacecolor"] = "none"
    m_kw["markeredgewidth"] = 3

    # 1. Agrupación global del ensamble: sumamos los conteos F(n)_j de todas las simulaciones
    stats = (
        df_por_impacto.groupby(["count", r"$\phi$"])["F(n)_j"]
        .sum()
        .reset_index(name="N_k")
    )

    # 2. Suma total de clústeres eyectados en el ensamble para cada factor de llenado
    phi_sums = stats.groupby(r"$\phi$")["N_k"].transform("sum")

    # 3. Probabilidad física real P(k) del ensamble (suma exactamente 1.0 por definición)
    stats["mean"] = np.where(phi_sums > 0, stats["N_k"] / phi_sums, 0.0)
    stats["mean_plot"] = stats["mean"].replace(0, np.nan)

    phi_vals = sorted(df_por_impacto[r"$\phi$"].unique())

    for idx, phi_val in enumerate(phi_vals):
        subset = stats[stats[r"$\phi$"] == phi_val].sort_values("count")

        valid = subset[subset["mean_plot"].notna() & (subset["mean_plot"] > 0)]

        x_vals = valid["count"].values
        y_vals = valid["mean_plot"].values

        ax.plot(
            x_vals,
            y_vals,
            label=phi_val,
            color=palette[idx % len(palette)],
            markersize=MK_SIZE,
            linestyle="",
            **m_kw,
        )
        if not valid.empty:
            first = valid.iloc[0]
            x0, y0 = first["count"], first["mean_plot"]
            x_trend = np.logspace(0, 1.1, 100)
            y_trend = y0 * (x_trend / x0) ** (-2.8)
            ax.plot(
                x_trend,
                y_trend,
                ":",
                color=palette[idx],
                alpha=0.9,
                linewidth=3.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return handles, labels


def plot_cluster_size_distribution(casos, df_dist, df):
    dfs = _prepare_cluster_data(casos, df_dist, df)
    velocities = sorted(set(v for df in dfs for v in df["v"].unique()))

    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, axes = plt.subplots(
            len(velocities),
            3,
            figsize=(15, 5 * len(velocities)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for i, v in enumerate(velocities):
            for j, (n, alfa) in enumerate(casos):
                ax = axes[i, j]
                df_v = dfs[j][dfs[j]["v"] == v].copy()
                if df_v.empty:
                    ax.set_visible(False)
                    continue

                df_v["Prob_local"] = np.where(
                    df_v["M_t"] > 0,
                    df_v["F(n)_j"] / df_v["M_t"],
                    0.0,
                )

                _ = _plot_distribution_panel(df_v, "Prob_local", ax, n, alfa)

                if i == 0:
                    label = COL_LABELS[j]
                    if n == 1 and alfa == 0:
                        ax.set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
                    elif n == 3 and alfa == 0:
                        ax.set_title(
                            rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale)
                        )
                    else:
                        ax.set_title(
                            rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale)
                        )
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")

        axes[0, 0].set_xlim(0.95, 12)
        axes[0, 0].set_ylim(1e-3, 1.1)

        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.07),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.08)
        fig.savefig("figuras/plot_PDF_por_vel_cluster.png", bbox_inches="tight")
        plt.close(fig)


def plot_cluster_size_distribution_sin_relleno(casos, df_dist):
    dfs = _prepare_cluster_data_sin_relleno(casos, df_dist)
    velocities = sorted(set(v for df in dfs for v in df["v"].unique()))

    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, axes = plt.subplots(
            len(velocities),
            3,
            figsize=(15, 5 * len(velocities)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for i, v in enumerate(velocities):
            for j, (n, alfa) in enumerate(casos):
                ax = axes[i, j]
                df_v = dfs[j][dfs[j]["v"] == v].copy()
                if df_v.empty:
                    ax.set_visible(False)
                    continue

                df_v["Prob_local"] = np.where(
                    df_v["M_t"] > 0,
                    df_v["F(n)_j"] / df_v["M_t"],
                    0.0,
                )

                _ = _plot_distribution_panel(df_v, "Prob_local", ax, n, alfa)

                if i == 0:
                    label = COL_LABELS[j]
                    if n == 1 and alfa == 0:
                        ax.set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
                    elif n == 3 and alfa == 0:
                        ax.set_title(
                            rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale)
                        )
                    else:
                        ax.set_title(
                            rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale)
                        )
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")

        axes[0, 0].set_xlim(0.95, 12)
        axes[0, 0].set_ylim(1e-3, 1.1)

        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.11)
        fig.savefig("figuras/plot_PDF_por_vel_cluster_sin_relleno.png", bbox_inches="tight")
        plt.close(fig)


def plot_max_depth_proyectil_distribution(casos, df_dist):
    scale = 1.30
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        palette = sns.color_palette("tab10")
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[
                (df["proyectil_erosion"] == False)
                & (df["i"] == 80)
                & (df["z"] < 100e-6)
            ]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"], as_index=False)[
                "depth"
            ].max()
            m_kw = _marker_kwargs(n, alfa)
            sns.lineplot(
                data=df,
                x="vel0",
                y="depth",
                errorbar="se",
                palette="tab10",
                hue=r"$\phi$",
                markersize=MK_SIZE,
                linestyle="",
                linewidth=3.8,
                err_style="bars",
                err_kws={"capsize": 5},
                ax=ax[i],
                **m_kw,
            )
            _make_hollow_markers(ax[i])
            ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            label = COL_LABELS[i]
            if n == 1 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            elif n == 3 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale))
            else:
                ax[i].set_title(rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"$d_{max}$ [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
            ff_vals = sorted(df[r"$\phi$"].unique())
            for idx, ff_val in enumerate(ff_vals):
                ff_data = df[df[r"$\phi$"] == ff_val]
                x_fit = ff_data["vel0"].values
                y_fit = ff_data["depth"].values
                mask_fit = (y_fit > 0) & (x_fit > 0)
                x_fit, y_fit = x_fit[mask_fit], y_fit[mask_fit]
                if len(x_fit) < 2:
                    continue
                try:
                    popt, _ = curve_fit(
                        power_law,
                        x_fit,
                        y_fit,
                        p0=[1, 0, 1],
                        maxfev=5000,
                        bounds=([0, 0, 0], [np.inf, 0.00000001, 4]),
                    )
                    _, _, c = popt
                    x_sm = np.logspace(
                        np.log10(x_fit.min()), np.log10(x_fit.max()), 200
                    )
                    ax[i].plot(
                        x_sm,
                        power_law(x_sm, *popt),
                        ":",
                        color=palette[idx],
                        alpha=0.9,
                        linewidth=3.8,
                    )
                    ax[i].text(
                        0.97,
                        0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                        rf"$c={c:.2f}$",
                        transform=ax[i].transAxes,
                        color=palette[idx],
                        fontsize=FONT_ANNOTATION,
                        fontweight="bold",
                        ha="right",
                        va="bottom",
                    )
                except (RuntimeError, ValueError):
                    pass
        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        "figuras/plot_max_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_mean_depth_proyectil_distribution(casos, df_dist):
    scale = 1.30
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        palette = sns.color_palette("tab10")
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[
                (df["proyectil_erosion"] == False)
                & (df["i"] == 80)
                & (df["z"] < 100e-6)
            ]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"], as_index=False)[
                "depth"
            ].mean()
            m_kw = _marker_kwargs(n, alfa)
            sns.lineplot(
                data=df,
                x="vel0",
                y="depth",
                errorbar="se",
                palette="tab10",
                hue=r"$\phi$",
                markersize=MK_SIZE,
                linestyle="",
                linewidth=3.8,
                err_style="bars",
                err_kws={"capsize": 5},
                ax=ax[i],
                **m_kw,
            )
            _make_hollow_markers(ax[i])

            ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            label = COL_LABELS[i]
            if n == 1 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            elif n == 3 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale))
            else:
                ax[i].set_title(rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"$d_{mean}$ [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
            ff_vals = sorted(df[r"$\phi$"].unique())
            for idx, ff_val in enumerate(ff_vals):
                ff_data = df[df[r"$\phi$"] == ff_val]
                x_fit = ff_data["vel0"].values
                y_fit = ff_data["depth"].values
                mask_fit = (y_fit > 0) & (x_fit > 0)
                x_fit, y_fit = x_fit[mask_fit], y_fit[mask_fit]
                if len(x_fit) < 2:
                    continue
                try:
                    popt, _ = curve_fit(
                        power_law,
                        x_fit,
                        y_fit,
                        p0=[1, 0, 1],
                        maxfev=5000,
                        bounds=([0, 0, 0], [np.inf, 0.00000001, 4]),
                    )
                    _, _, c = popt
                    x_sm = np.logspace(
                        np.log10(x_fit.min()), np.log10(x_fit.max()), 200
                    )
                    ax[i].plot(
                        x_sm,
                        power_law(x_sm, *popt),
                        ":",
                        color=palette[idx],
                        alpha=0.9,
                        linewidth=3.8,
                    )
                    ax[i].text(
                        0.97,
                        0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                        rf"$c={c:.2f}$",
                        transform=ax[i].transAxes,
                        color=palette[idx],
                        fontsize=FONT_ANNOTATION,
                        fontweight="bold",
                        ha="right",
                        va="bottom",
                    )
                except (RuntimeError, ValueError):
                    pass
        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        "figuras/plot_mean_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_max_depth_lecho_distribution(casos, df_dist):
    scale = 1.30
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        palette = sns.color_palette("tab10")
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[(df["type"] == 1) & (df["i"] == 0) & (df["z"] < 100e-6)]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"], as_index=False)[
                "depth"
            ].max()
            m_kw = _marker_kwargs(n, alfa)
            sns.lineplot(
                data=df,
                x="vel0",
                y="depth",
                errorbar="se",
                palette="tab10",
                hue=r"$\phi$",
                markersize=MK_SIZE,
                linestyle="",
                linewidth=3.8,
                err_style="bars",
                err_kws={"capsize": 5},
                ax=ax[i],
                **m_kw,
            )
            _make_hollow_markers(ax[i])
            # ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            label = COL_LABELS[i]
            if n == 1 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            elif n == 3 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale))
            else:
                ax[i].set_title(rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"$D_{max}$ [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
            ff_vals = sorted(df[r"$\phi$"].unique())
            for idx, ff_val in enumerate(ff_vals):
                ff_data = df[(df[r"$\phi$"] == ff_val) & (df["vel0"] > 5)]
                mask_fit = ff_data["depth"] > 0
                x_fit = ff_data.loc[mask_fit, "vel0"].values
                y_fit = ff_data.loc[mask_fit, "depth"].values
                if len(x_fit) < 2:
                    continue
                try:
                    popt, _ = curve_fit(
                        power_law,
                        x_fit,
                        y_fit,
                        p0=[1, 0, 1],
                        maxfev=5000,
                        bounds=([-np.inf, 0, 0], [np.inf, 0.00001, np.inf]),
                    )
                    _, _, c = popt
                    x_sm = np.logspace(np.log10(5), np.log10(x_fit.max()), 200)
                    ax[i].plot(
                        x_sm,
                        power_law(x_sm, *popt),
                        ":",
                        color=palette[idx],
                        alpha=0.9,
                        linewidth=3.8,
                    )
                    ax[i].text(
                        0.97,
                        0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                        rf"$c={c:.2f}$",
                        transform=ax[i].transAxes,
                        color=palette[idx],
                        fontsize=FONT_ANNOTATION,
                        fontweight="bold",
                        ha="right",
                        va="bottom",
                    )
                    ax[i].set_ylim(bottom=1e-1, top=1.1e1)
                except (RuntimeError, ValueError):
                    pass
        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        "figuras/plot_max_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_mean_depth_lecho_distribution(casos, df_dist):
    scale = 1.30
    with plt.rc_context(
        {
            "axes.labelsize": round(FONT_AXES_LABEL * scale),
            "xtick.labelsize": round(FONT_TICK_LABEL * scale),
            "ytick.labelsize": round(FONT_TICK_LABEL * scale),
            "legend.fontsize": round(FONT_LEGEND * scale),
        }
    ):
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        palette = sns.color_palette("tab10")
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[(df["type"] == 1) & (df["i"] == 0) & (df["z"] < 100e-6)]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"], as_index=False)[
                "depth"
            ].mean()
            m_kw = _marker_kwargs(n, alfa)
            sns.lineplot(
                data=df,
                x="vel0",
                y="depth",
                errorbar="se",
                palette="tab10",
                hue=r"$\phi$",
                markersize=MK_SIZE,
                linestyle="",
                linewidth=3.8,
                err_style="bars",
                err_kws={"capsize": 5},
                ax=ax[i],
                **m_kw,
            )
            _make_hollow_markers(ax[i])

            # ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            label = COL_LABELS[i]
            if n == 1 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            elif n == 3 and alfa == 0:
                ax[i].set_title(rf"{label} $N_p$: {n} H", fontsize=round(FONT_TITLE * scale))
            else:
                ax[i].set_title(rf"{label} $N_p$: {n} V", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"$D_{mean}$ [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
            ff_vals = sorted(df[r"$\phi$"].unique())
            for idx, ff_val in enumerate(ff_vals):
                ff_data = df[(df[r"$\phi$"] == ff_val) & (df["vel0"] > 5)]
                mask_fit = ff_data["depth"] > 0
                x_fit = ff_data.loc[mask_fit, "vel0"].values
                y_fit = ff_data.loc[mask_fit, "depth"].values
                if len(x_fit) < 2:
                    continue
                try:
                    popt, _ = curve_fit(
                        power_law,
                        x_fit,
                        y_fit,
                        p0=[1, 0, 1],
                        maxfev=5000,
                        bounds=([-np.inf, 0, -np.inf], [np.inf, 0.00001, np.inf]),
                    )
                    _, _, c = popt
                    x_sm = np.logspace(np.log10(5), np.log10(x_fit.max()), 200)
                    ax[i].plot(
                        x_sm,
                        power_law(x_sm, *popt),
                        ":",
                        color=palette[idx],
                        alpha=0.9,
                        linewidth=3.8,
                    )
                    ax[i].text(
                        0.97,
                        0.03 + (len(ff_vals) - 1 - idx) * 0.055,
                        rf"$c={c:.2f}$",
                        transform=ax[i].transAxes,
                        color=palette[idx],
                        fontsize=FONT_ANNOTATION,
                        fontweight="bold",
                        ha="right",
                        va="bottom",
                    )
                    ax[i].set_ylim(bottom=1e-1, top=1.1e1)
                except (RuntimeError, ValueError):
                    pass
        handles, labels = _ff_marker_handles()
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=4,
        )
        phi_palette = sns.color_palette("bright")
        for i in range(len(PHI_VALS)):
            leg.get_texts()[i].set_color(phi_palette[i])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        "figuras/plot_mean_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
        bbox_inches="tight",
    )
    plt.close()


def linear_model(x, m, b):
    return m * x + b


def power_law(x, a, b, c):
    scalar_input = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    b = b if abs(b) > 1e-6 else 0.0
    z = x - b
    mask = z > 0
    result = np.zeros_like(z)
    result[mask] = a * (z[mask] / 0.17) ** c
    return result.item() if scalar_input else result


def ajuste_modelos(subset, n, x_col="vel0", y_col="yield"):
    raw_results = []
    formatted_results = []

    grouped = subset.groupby(x_col)[y_col]
    means = grouped.mean()
    sems = grouped.sem(ddof=1)
    mask = (sems >= 0) & (~np.isnan(sems))
    x_fit = means.index[mask].values
    y_fit = means.values[mask]

    n_params = 3
    dof = len(x_fit) - n_params
    t_val = t.ppf(0.975, max(dof, 1)) if dof > 0 else 1.0

    popt, pcov = curve_fit(
        power_law,
        x_fit,
        y_fit,
        p0=[1, 2, 1],
        maxfev=5000,
        bounds=([0, 0, 0], [np.inf, 5.1, 4]),
    )

    err = np.sqrt(np.diag(pcov)) * t_val
    raw_results.append(popt)
    formatted_results.append([f"{v}" for v in unp.uarray(popt, err)])

    return raw_results, formatted_results


def calcular_metricas(df, modelo, params, x_col="vel0", y_col="yield"):
    y_obs = df[y_col].values
    y_pred = modelo(df[x_col].values, *params)
    n_params = len(params)

    # 1. RMSE sobre los datos crudos
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))

    # 2. Chi-cuadrado sobre las medias de los grupos
    grouped = df.groupby(x_col)[y_col]
    means = grouped.mean()
    sems = grouped.sem(ddof=1)

    # Identificar únicamente los grupos válidos (SEM > 0 y no NaN)
    grupos_validos = [x for x in means.index if sems[x] > 0 and not np.isnan(sems[x])]
    n_grupos_validos = len(grupos_validos)

    chi2 = sum(((means[x] - modelo(x, *params)) / sems[x]) ** 2 for x in grupos_validos)

    # Usar n_grupos_validos para los grados de libertad
    chi2_red = (
        chi2 / (n_grupos_validos - n_params) if n_grupos_validos > n_params else np.nan
    )

    return {"rmse": rmse, "chi2_red": chi2_red}


def grilla(np, df):
    data = df[df["np"] == np].copy()
    # 1. Grilla Global Vectorizada
    step = data[data["posx"] > 0]["posx"].min()
    data["grid_x"] = (data["posx"] / step).round().astype(int)
    data["grid_y"] = (data["posy"] / step).round().astype(int)

    x_min = int(data["grid_x"].min())
    y_min = int(data["grid_y"].min())
    x_max = int(data["grid_x"].max())
    y_max = int(data["grid_y"].max())

    reales = (
        data[["grid_x", "grid_y"]]
        .drop_duplicates()
        .assign(existe=True)
        .set_index(["grid_x", "grid_y"])
    )
    x_r, y_r = range(x_min, x_max + 1), range(y_min, y_max + 1)
    base_grid = (
        pd.MultiIndex.from_product([x_r, y_r], names=["grid_x", "grid_y"])
        .to_frame(index=False)
        .set_index(["grid_x", "grid_y"])
    )
    base_grid = base_grid.join(reales).fillna(False).infer_objects(copy=False)
    return base_grid, data


def extract_cluster_data(df, np_val, angle, ff, v):
    i_max = df["i"].max()
    mask = (df["i"] == i_max) & ((df["type"] == 1) | df["proyectil_erosion"])
    df_filt = df[mask]
    if df_filt.empty:
        return pd.DataFrame()

    df_cluster = (
        df_filt.groupby(["posx", "posy", "cluster"])
        .agg(
            count=("cluster", "size"),
            x_cm=("x", "mean"),
            y_cm=("y", "mean"),
            z_cm=("z", "mean"),
            v_cm_x=("vx", "mean"),
            v_cm_y=("vy", "mean"),
            v_cm_z=("vz", "mean"),
            var_x=("x", lambda x: x.var(ddof=0)),
            var_y=("y", lambda x: x.var(ddof=0)),
            var_z=("z", lambda x: x.var(ddof=0)),
        )
        .reset_index()
    )

    v_cm = df_cluster[["v_cm_x", "v_cm_y", "v_cm_z"]].values
    km = 0.5 * np.sum(v_cm**2, axis=1)
    rg = np.sqrt(df_cluster["var_x"] + df_cluster["var_y"] + df_cluster["var_z"])
    v_mag = np.sqrt(2 * km)
    angulo_vel = np.where(
        v_mag > 0,
        np.degrees(np.arccos(np.clip(df_cluster["v_cm_z"] / v_mag, -1.0, 1.0))),
        0.0,
    )

    return df_cluster.assign(
        km=km, rg=rg, angulo_vel=angulo_vel, np=np_val, angulo=angle, ff=ff, v=v
    ).drop(columns=["var_x", "var_y", "var_z"])


def extract_main_data(df, casos):
    datos_distribucion = []
    datos_cluster = []
    datos_proyectil = []
    for n, alfa in casos:
        base_grid, df_grid = grilla(n, df)
        for ff in [0.15, 0.35]:
            vels = sorted(df_grid.loc[df_grid["ff"] == ff, "vel0"].abs().unique())
            for v in vels:
                sim = df_grid[
                    (df_grid["ff"] == ff)
                    & (df_grid["vel0"] == -1 * v)
                    & (df_grid["np"] == n)
                    & (df_grid["angulo"] == alfa)
                ]
                if not sim.empty:
                    datos_distribucion.append(
                        plot_heatmap(sim, ff, v, n, alfa, base_grid)
                    )
                    datos_cluster.append(extract_cluster_data(sim, n, alfa, ff, v))
                    datos_proyectil.append(sim)
                    plt.close()
    df_dist = pd.concat(datos_distribucion)
    df_cluster = pd.concat(datos_cluster)
    df_proyectil = pd.concat(datos_proyectil)
    return df_dist, df_cluster, df_proyectil


def tabla_rebote_proyectil(df):
    i_max = df["i"].max()
    proj = df[(df["type"] == 2) & (df["i"] == i_max)].copy()
    proj["vel0"] = proj["vel0"].abs()

    prob_impacto = (
        (
            proj.groupby(["np", "angulo", "ff", "vel0", "posx", "posy"])[
                "proyectil_erosion"
            ]
        )
        .mean()
        .reset_index(name="p_rebote")
    )

    resumen = prob_impacto.groupby(["np", "angulo", "ff", "vel0"], as_index=False)[
        "p_rebote"
    ].agg(probabilidad="mean", se="sem")

    pivot = resumen.pivot_table(
        index=["np", "angulo"],
        columns=["vel0", "ff"],
        values="probabilidad",
        aggfunc="first",
    )
    pivot.columns.names = [r"$v$", r"$\phi$"]
    pivot.index.names = [r"$N_p$", "Orientación"]

    idx_frame = pivot.index.to_frame()
    # Np 1 no tiene ni H ni V
    idx_frame["Orientación"] = idx_frame["Orientación"].astype(object)
    idx_frame.loc[idx_frame[r"$N_p$"] == 1, "Orientación"] = ""

    idx_frame["Orientación"] = (
        idx_frame["Orientación"].map({0: "V", 90: "H"}).fillna(idx_frame["Orientación"])
    )

    pivot.index = pd.MultiIndex.from_frame(idx_frame)

    return pivot, resumen


def graficar_heatmap(pivot, cbar_label="Probabilidad de Rebote", fname=None):
    pivot_numerico = pivot.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        pivot_numerico,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        annot_kws={"size": 13},
        cbar_kws={"label": cbar_label},
        vmin=min(pivot_numerico.values.flatten()),
        vmax=max(pivot_numerico.values.flatten()),
        ax=ax,
    )
    sns.set(font_scale=1.3)
    ff_labels = pivot_numerico.columns.get_level_values(1)

    ax.set_xticks(np.arange(len(ff_labels)) + 0.5)
    ax.set_xticklabels(ff_labels)
    ax.set_xlabel(r"$\phi$")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    v_labels = pivot_numerico.columns.get_level_values(0)
    v_unicos = v_labels.unique()

    posiciones_top = []
    etiquetas_top = []

    for v_val in v_unicos:
        indices = np.where(v_labels == v_val)[0]
        centro = indices.mean() + 0.5
        posiciones_top.append(centro)
        etiquetas_top.append(v_val)

    ax_top.set_xticks(posiciones_top)
    ax_top.set_xticklabels(etiquetas_top)
    ax_top.set_xlabel(r"$v$ [$m/s$]")

    ax_top.tick_params(axis="x", length=0)

    ax.set_ylabel("$N_p$")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight")

def tabla_porcentaje_proyectil(df):
    i_max = df["i"].max()

    # 1. Grilla base para mantener la estructura dimensional de todos los casos
    df_grid = df[["np", "angulo", "ff", "vel0", "posx", "posy"]].drop_duplicates().copy()
    df_grid["vel0"] = df_grid["vel0"].abs()
    df_grid = df_grid.drop_duplicates()

    # 2. Filtrar proyectiles que erosionaron y partículas del lecho erosionadas
    proj = df[(df["type"] == 2) & (df["i"] == i_max) & (df["proyectil_erosion"] == True)].copy()
    eros = df[(df["type"] == 1) & (df["i"] == i_max)].copy()

    proj["vel0"] = proj["vel0"].abs()
    eros["vel0"] = eros["vel0"].abs()

    columnas_grupo = ["np", "angulo", "ff", "vel0", "posx", "posy"]

    # 3. Contar ocurrencias por celda
    total_proyectiles = proj.groupby(columnas_grupo).size().reset_index(name="count_proj")
    total_erosion = eros.groupby(columnas_grupo).size().reset_index(name="count_eros")

    # 4. Unir con la grilla base
    df_merged = df_grid.merge(total_proyectiles, on=columnas_grupo, how="left")
    df_merged = df_merged.merge(total_erosion, on=columnas_grupo, how="left").fillna(0)

    # 5. REGLA CLAVE: Si no hubo erosión (total_eventos == 0), asignamos np.nan
    # Esto le dice a Pandas: "NO cuentes esta celda en el promedio ni en el error estándar"
    df_merged["total_eventos"] = df_merged["count_eros"] + df_merged["count_proj"]
    df_merged["porcentaje_erosion"] = np.where(
        df_merged["total_eventos"] > 0,
        (100 * df_merged["count_proj"]) / df_merged["total_eventos"],
        np.nan
    )

    # 6. Agrupar y resumir (Pandas omitirá automáticamente los np.nan al calcular mean y sem)
    resumen = df_merged.groupby(["np", "angulo", "ff", "vel0"], as_index=False)[
        "porcentaje_erosion"
    ].agg(mean="mean", se="sem")

    # 7. Pivotear para la tabla LaTeX
    pivot = resumen.pivot_table(
        index=["np", "angulo"],
        columns=["vel0", "ff"],
        values="mean",
        aggfunc="first",
    )
    pivot.columns.names = [r"$v$", r"$\phi$"]
    pivot.index.names = [r"$N_p$", "Orientación"]

    idx_frame = pivot.index.to_frame()
    idx_frame["Orientación"] = (
        idx_frame["Orientación"].map({0: "V", 90: "H"}).fillna(idx_frame["Orientación"])
    )
    pivot.index = pd.MultiIndex.from_frame(idx_frame)

    return pivot, resumen

def tabla_erosion_cruda(df):
    i_max = df["i"].max()

    proj = df[
        (df["type"] == 2) & (df["i"] == i_max) & (df["proyectil_erosion"] == True)
    ].copy()
    eros = df[(df["type"] == 1) & (df["i"] == i_max)].copy()

    proj["vel0"] = proj["vel0"].abs()
    eros["vel0"] = eros["vel0"].abs()

    columnas_grupo = ["np", "angulo", "ff", "vel0", "posx", "posy"]

    total_proyectiles = (
        proj.groupby(columnas_grupo).size().reset_index(name="count_proj")
    )
    total_erosion = eros.groupby(columnas_grupo).size().reset_index(name="count_eros")

    df_merged = pd.merge(
        total_proyectiles, total_erosion, on=columnas_grupo, how="outer"
    ).fillna(0)

    df_merged["total_eventos"] = df_merged["count_eros"] + df_merged["count_proj"]
    df_merged["porcentaje_erosion"] = np.where(
        df_merged["total_eventos"] > 0,
        (100 * df_merged["count_proj"]) / df_merged["total_eventos"],
        0,
    )

    return df_merged


def _jacobiano_numerico(modelo, x, params, delta=1e-7):
    n = len(x)
    p = len(params)
    J = np.empty((n, p))
    for j in range(p):
        params_up = params.copy()
        params_dn = params.copy()
        params_up[j] += delta
        params_dn[j] -= delta
        J[:, j] = (modelo(x, *params_up) - modelo(x, *params_dn)) / (2 * delta)
    return J


def _ajustar_con_ci(modelo, x, y, p0, bounds):
    # NOTA DE CORRECCIÓN: El cálculo de los intervalos de confianza fue modificado
    # para realizarse sobre la nube de puntos empleando el factor de la distribución t de Student al 95%[cite: 1].
    popt, _ = curve_fit(modelo, x, y, p0=p0, maxfev=5000, bounds=bounds)
    J = _jacobiano_numerico(modelo, x, popt)
    residuos = y - modelo(x, *popt)
    n_pts = len(x)
    n_params = len(popt)
    dof = max(n_pts - n_params, 1)
    sigma2 = np.dot(residuos, residuos) / dof
    cov_theta = sigma2 * np.linalg.pinv(J.T @ J)

    # Cuantil t de Student para un nivel de confianza del 95% sobre los grados de libertad[cite: 1]
    t_val = t.ppf(0.975, dof) if dof > 0 else 1.0
    ci = np.sqrt(np.diag(cov_theta)) * t_val
    return popt, ci


def _depth_stats(sub_proj_v, agg_name):
    depth = sub_proj_v.groupby(["posx", "posy"])["depth"].agg(agg_name)
    return depth.mean(), depth.sem(ddof=1)


def tabla_ajustes_profundidad(casos, df_proyectil):
    variantes = [
        {
            "nombre": "d_max (proyectil)",
            "filtro": lambda d: d[
                (d["proyectil_erosion"] == False) & (d["i"] == 80) & (d["z"] < 100e-6)
            ],
            "agg": "max",
            "vel0_min": 0,
            "bounds": ([0, 0, 0], [np.inf, 0.00000001, 4]),
            "sem_filter": False,
        },
        {
            "nombre": "d_mean (proyectil)",
            "filtro": lambda d: d[
                (d["proyectil_erosion"] == False) & (d["i"] == 80) & (d["z"] < 100e-6)
            ],
            "agg": "mean",
            "vel0_min": 0,
            "bounds": ([0, 0, 0], [np.inf, 0.00000001, 4]),
            "sem_filter": False,
        },
        {
            "nombre": "D_max (lecho)",
            "filtro": lambda d: d[(d["type"] == 1) & (d["i"] == 0) & (d["z"] < 100e-6)],
            "agg": "max",
            "vel0_min": 5,
            "bounds": ([-np.inf, 0, 0], [np.inf, 0.00001, np.inf]),
            "sem_filter": False,
        },
        {
            "nombre": "D_mean (lecho)",
            "filtro": lambda d: d[(d["type"] == 1) & (d["i"] == 0) & (d["z"] < 100e-6)],
            "agg": "mean",
            "vel0_min": 5,
            "bounds": ([-np.inf, 0, -np.inf], [np.inf, 0.00001, np.inf]),
            "sem_filter": True,
        },
    ]

    rows = []
    for var in variantes:
        for n, alfa in casos:
            sub = df_proyectil[
                (df_proyectil["np"] == n) & (df_proyectil["angulo"] == alfa)
            ].copy()
            sub = var["filtro"](sub)
            sub["z"] = sub["z"] * 1e6
            sub["depth"] = 100 - sub["z"]
            sub["vel0"] = sub["vel0"].abs()
            sub = sub.groupby(["vel0", "posx", "posy", "ff"], as_index=False)[
                "depth"
            ].agg(var["agg"])

            for ff in PHI_VALS:
                ff_data = sub[sub["ff"] == ff]

                if var["vel0_min"] > 0:
                    ff_data = ff_data[ff_data["vel0"] > var["vel0_min"]]
                    mask = ff_data["depth"] > 0
                    x_fit = ff_data.loc[mask, "vel0"].values
                    y_fit = ff_data.loc[mask, "depth"].values
                else:
                    x_fit = ff_data["vel0"].values
                    y_fit = ff_data["depth"].values
                    mask = (y_fit > 0) & (x_fit > 0)
                    x_fit, y_fit = x_fit[mask], y_fit[mask]

                if len(x_fit) < 2:
                    continue

                try:
                    popt, se = _ajustar_con_ci(
                        power_law, x_fit, y_fit, p0=[1, 0, 1], bounds=var["bounds"]
                    )
                    vals = unp.uarray(popt, se)
                    rows.append(
                        {
                            "profundidad": var["nombre"],
                            "n": n,
                            "angulo": alfa,
                            "ff": ff,
                            "a": f"{vals[0]}",
                            "b": f"{vals[1]}",
                            "c": f"{vals[2]}",
                        }
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    pass

    return pd.DataFrame(rows)


def tabla_resumen(df_dist, df_proyectil, df, casos):
    ff, v = 0.15, 50.0
    rows = []

    rows.append(("Variable", *[rf"${n},{a}$" for n, a in casos]))

    # Np row
    rows.append((r"$N_p$", *[str(n) for n, _ in casos]))
    rows.append((r"$v$", *[f"${v:.0f}\\,m/s$"] * len(casos)))
    rows.append((r"$\phi$", *[str(ff)] * len(casos)))

    # Repeticiones
    reps = []
    for n, alfa in casos:
        sub = df_dist[
            (df_dist["np"] == n)
            & (df_dist["angulo"] == alfa)
            & (df_dist["ff"] == ff)
            & (df_dist["vel0"] == v)
        ]
        reps.append(str(sub["yield"].count()))
    rows.append(("Repeticiones", *reps))

    rows.append(("",) * (len(casos) + 1))

    # Yield
    y_vals = []
    for n, alfa in casos:
        sub = df_dist[
            (df_dist["np"] == n)
            & (df_dist["angulo"] == alfa)
            & (df_dist["ff"] == ff)
            & (df_dist["vel0"] == v)
        ]
        y = sub["yield"]
        y_vals.append(f"${y.mean():.3f}\\pm{y.sem(ddof=1):.3f}$")
    rows.append((r"$Y$", *y_vals))

    # Depth metrics
    for label, tipo, agg_name in [
        (r"$d_{\max}$ (proyectil)", "proyectil", "max"),
        (r"$d_{mean}$ (proyectil)", "proyectil", "mean"),
        (r"$d_{\max}$ (lecho)", "lecho", "max"),
        (r"$d_{mean}$ (lecho)", "lecho", "mean"),
    ]:
        vals = []
        for n, alfa in casos:
            proj = df_proyectil[
                (df_proyectil["np"] == n)
                & (df_proyectil["angulo"] == alfa)
                & (df_proyectil["ff"] == ff)
            ].copy()
            df_v = proj[proj["vel0"].abs() == v].copy()
            if tipo == "proyectil":
                filtro = (
                    (df_v["proyectil_erosion"] == False)
                    & (df_v["i"] == 80)
                    & (df_v["z"] < 100e-6)
                )
            else:
                filtro = (df_v["type"] == 1) & (df_v["i"] == 0) & (df_v["z"] < 100e-6)
            df_filt = df_v[filtro].copy()
            if df_filt.empty:
                vals.append("---")
            else:
                df_filt["depth"] = 100 - df_filt["z"] * 1e6
                m, s = _depth_stats(df_filt, agg_name)
                vals.append(f"${m:.2f}\\pm{s:.2f}\\,\\mu m$")
        rows.append((label, *vals))

    # Proporciones
    _, df_rebote = tabla_rebote_proyectil(df)
    rb_vals = []
    for n, alfa in casos:
        sub_r = df_rebote[
            (df_rebote["np"] == n)
            & (df_rebote["angulo"] == alfa)
            & (df_rebote["ff"] == ff)
            & (df_rebote["vel0"] == v)
        ]
        if sub_r.empty:
            rb_vals.append("---")
        else:
            rm, rs = sub_r["probabilidad"].iloc[0], sub_r["se"].iloc[0]
            rb_vals.append(f"${rm * 100:.1f}\\pm{rs * 100:.1f}\\,\\%$")
    rows.append((r"Proporci\'on de masa", *rb_vals))

    _, df_erosion = tabla_porcentaje_proyectil(df)
    er_vals = []
    for n, alfa in casos:
        sub_e = df_erosion[
            (df_erosion["np"] == n)
            & (df_erosion["angulo"] == alfa)
            & (df_erosion["ff"] == ff)
            & (df_erosion["vel0"] == v)
        ]
        if sub_e.empty:
            er_vals.append("---")
        else:
            em, es = sub_e["mean"].iloc[0], sub_e["se"].iloc[0]
            er_vals.append(f"${em:.1f}\\pm{es:.1f}\\,\\%$")
    rows.append((r"Proporci\'on de proyectil", *er_vals))

    lines = [r"\begin{tabular}{l" + "c" * len(casos) + "}"]
    for i, row in enumerate(rows):
        if i == 5:
            lines.append(r"\hline")
        if row[0] == "":
            lines.append(r"\hline")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

def generar_tabla_latex_profundidad(df, var_nombre, caption, label):
    sub = df[df["profundidad"] == var_nombre].copy()
    if sub.empty:
        return f"% No hay datos disponibles para {var_nombre}"

    # Mapeo de (n, angulo) a las etiquetas LaTeX deseadas
    config_names = {
        (1, 0): r"\(N_p:1\)",
        (3, 90): r"\(N_p:3\text{V}\)",
        (3, 0): r"\(N_p:3\text{H}\)",
    }

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        \textbf{Configuración} & \(\phi = 0.15\) & \(\phi = 0.35\) \\",
        r"        \midrule"
    ]

    for (n, angulo), config_label in config_names.items():
        row_vals = [f"        {config_label}"]
        for ff in [0.15, 0.35]:
            val_series = sub[(sub["n"] == n) & (sub["angulo"] == angulo) & (sub["ff"] == ff)]["c"]
            if not val_series.empty:
                # Reemplazar +/- por \pm y envolver en \( ... \)
                val_str = str(val_series.iloc[0]).replace("+/-", r" \pm ")
                row_vals.append(f"\\({val_str}\\)")
            else:
                row_vals.append("---")
        lines.append(" & ".join(row_vals) + r" \\")

    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)

def main():
    df = pd.read_csv(CSV_PATH)
    # 2. Análisis iterativo
    # Grfico 1x3 con np, angulo
    casos = [(1, 0), (3, 90), (3, 0)]
    combinaciones = [
        (1, 0, 0.15),
        (1, 0, 0.35),
        (3, 90, 0.15),
        (3, 90, 0.35),
        (3, 0, 0.15),
        (3, 0, 0.35),
    ]
    df_dist, df_cluster, df_proyectil = extract_main_data(df, casos)

    #########################################
    ######## Distribución de Erosión ########
    #########################################
    # Ajustes por combinación (n, alfa, ff)
    idx = pd.MultiIndex.from_tuples(combinaciones, names=["n", "angulo", "ff"])
    ajustes_formatted = pd.DataFrame(
        columns=["power_a", "power_b", "power_c"], index=idx
    )
    ajustes_raw = pd.DataFrame(columns=["power_a", "power_b", "power_c"], index=idx)

    for n, alfa, ff in combinaciones:
        subset = df_dist[
            (df_dist["np"] == n) & (df_dist["angulo"] == alfa) & (df_dist["ff"] == ff)
        ]
        if subset.empty:
            continue
        raw, formatted = ajuste_modelos(subset, n)
        ajustes_raw.loc[(n, alfa, ff)] = [*raw[0]]
        ajustes_formatted.loc[(n, alfa, ff)] = [r for sub in formatted for r in sub]
    # 1x3: línea de puntos con ambos ff por subplot (existente)

    plot_erosion_distribution(casos, df_dist)
    plot_erosion_loglog(casos, df_dist)
    plot_erosion_violin(casos, df_dist)

    ######################################################
    ######## Distribución de Tamaño de Cluster ###########
    ######################################################

    plot_cluster_size_distribution(casos, df_cluster, df)
    plot_cluster_size_distribution_sin_relleno(casos, df_cluster)

    #######################################################
    ######## Distribución de Profundidad de Proyectil #####
    #######################################################

    plot_mean_depth_proyectil_distribution(casos, df_proyectil)
    plot_max_depth_proyectil_distribution(casos, df_proyectil)

    #######################################################
    ######## Distribución de Profundidad de Lecho #########
    #######################################################

    plot_mean_depth_lecho_distribution(casos, df_proyectil)
    plot_max_depth_lecho_distribution(casos, df_proyectil)

    #############################################################
    ######## Ajustes de profundidad (CSV) #######################
    #############################################################

    df_ajustes_prof = tabla_ajustes_profundidad(casos, df_proyectil)
    df_ajustes_prof.to_csv("datos/ajustes_profundidad.csv", index=False)

    print("\n% --- NOTA DE CORRECCIÓN ---")
    print("% Los intervalos de confianza para los ajustes de profundidad del lecho")
    print("% fueron corregidos para calcularse sobre la nube de puntos empleando la distribución t de Student[cite: 1].")

    #############################################################
    ######## Tabla resumen ######################################
    #############################################################

    latex_resumen = tabla_resumen(df_dist, df_proyectil, df, casos)
    with open("datos/resumen_graficos.tex", "w") as f:
        f.write(latex_resumen)
    print("\n% --- TABLA RESUMEN ---")
    print(latex_resumen)

    #######################################################
    ######## Métricas de los ajustes ######################
    #######################################################
    modelos = {"power_law": power_law}
    cols_modelo = {
        "power_law": ["power_a", "power_b", "power_c"],
    }
    metricas = []
    for n, alfa, ff in combinaciones:
        subset = df_dist[
            (df_dist["np"] == n) & (df_dist["angulo"] == alfa) & (df_dist["ff"] == ff)
        ]
        if subset.empty:
            continue
        for nombre_modelo, modelo in modelos.items():
            params = [
                ajustes_raw.loc[(n, alfa, ff), c] for c in cols_modelo[nombre_modelo]
            ]
            m = calcular_metricas(subset, modelo, params)
            metricas.append(
                {"n": n, "angulo": alfa, "ff": ff, "modelo": nombre_modelo, **m}
            )
    pd.DataFrame(metricas).to_csv("datos/metricas.csv", index=False)

    #################################################################
    ######## Probabilidad de Rebote del Proyectil ###################
    #################################################################

    pivot_rebote, df_rebote = tabla_rebote_proyectil(df)

    graficar_heatmap(
        pivot_rebote * 100, "Proporción de masa %", "figuras/heatmap_rebote.png"
    )
    df_rebote.to_csv("datos/rebote_proyectil.csv", index=False)
    latex_rebote = pivot_rebote.to_latex(multirow=True, float_format="%.2f")

    #################################################################
    # Probabilidad de Erosión del Proyectil
    #################################################################

    pivot_erosion, df_erosion = tabla_porcentaje_proyectil(df)
    graficar_heatmap(
        pivot_erosion, "Proporción de Erosión %", "figuras/heatmap_erosion.png"
    )
    df_erosion.to_csv("datos/erosion_proyectil.csv", index=False)
    latex_erosion = pivot_erosion.to_latex(multirow=True, float_format="%.3f")

    ##################################################################
    # Datos crudos de erosión por posición (posx, posy)
    ##################################################################

    df_erosion_cruda = tabla_erosion_cruda(df)
    df_erosion_cruda.to_csv("datos/erosion_cruda.csv", index=False)
    print("\n% --- DATOS CRUDOS DE EROSIÓN (por posx, posy) ---")
    print(
        df_erosion_cruda.groupby(["np", "angulo", "ff", "vel0"], as_index=False)[
            "porcentaje_erosion"
        ]
        .agg(
            conteo_posiciones="count",
            media="mean",
            std="std",
            min="min",
            max="max",
        )
        .to_string(index=False)
    )

    # ==========================================
    # 2. TABLA DE AJUSTES (solo exponente c)
    # ==========================================
    records_ajustes = []
    for n, alfa, ff in combinaciones:
        fmt = ajustes_formatted.loc[(n, alfa, ff)]
        if pd.isna(fmt["power_a"]):
            continue
        records_ajustes.append(
            {
                "n": n,
                "angulo": alfa,
                "ff": ff,
                r"$\alpha$": fmt["power_c"],
            }
        )
    df_ajustes = pd.DataFrame(records_ajustes)
    df_ajustes.to_csv("datos/ajustes.csv", index=False)

    # ==========================================
    # 3. GENERAR LaTeX
    # ==========================================
    # Reemplazar '+/-' por ' \pm ' y envolver en modo matemático
    for col in [r"$\alpha$"]:
        df_ajustes[col] = df_ajustes[col].apply(
            lambda x: f"${str(x).replace('+/-', r' \pm ')}$" if pd.notnull(x) else x
        )

    df_metricas = pd.read_csv("datos/metricas.csv")
    df_metricas["rmse"] = df_metricas["rmse"].round(2)
    df_metricas["chi2_red"] = df_metricas["chi2_red"].round(2)

    latex_metricas = df_metricas.set_index(["n", "angulo", "ff", "modelo"]).to_latex(
        multirow=True
    )
    latex_ajustes = df_ajustes.set_index(["n", "angulo", "ff"]).to_latex(
        multirow=True, escape=False
    )

    print("% --- TABLA DE REBOTE DE PROYECTIL ---")
    print(latex_rebote)
    print("\n% --- TABLA DE MÉTRICAS ---")
    print(latex_metricas)
    print("\n% --- TABLA DE AJUSTES ---")
    print(latex_ajustes)
    print("\n% --- TABLA DE EROSIÓN DE PROYECTIL ---")
    print(latex_erosion)
    #############################################################
    ######## Ajustes de profundidad (CSV y LaTeX) ###############
    #############################################################

    df_ajustes_prof = tabla_ajustes_profundidad(casos, df_proyectil)
    df_ajustes_prof.to_csv("datos/ajustes_profundidad.csv", index=False)

    # --- Generación de tablas LaTeX para D_max y D_mean (Lecho) ---
    latex_dmax = generar_tabla_latex_profundidad(
        df_ajustes_prof,
        var_nombre="D_max (lecho)",
        caption="Exponente \(c\) del ajuste de la profundidad máxima de excavación del lecho para cada configuración.",
        label="tab:exponentes_dmax"
    )

    latex_dmean = generar_tabla_latex_profundidad(
        df_ajustes_prof,
        var_nombre="D_mean (lecho)",
        caption="Exponente \(c\) del ajuste de la profundidad media de excavación del lecho para cada configuración.",
        label="tab:exponentes_dmean"
    )

    # Guardar en archivos .tex independientes
    with open("datos/tabla_dmax_lecho.tex", "w") as f:
        f.write(latex_dmax)
    with open("datos/tabla_dmean_lecho.tex", "w") as f:
        f.write(latex_dmean)

    print("\n% --- TABLA D_MAX (LECHO) ---")
    print(latex_dmax)
    print("\n% --- TABLA D_MEAN (LECHO) ---")
    print(latex_dmean)

if __name__ == "__main__":
    main()
    print("exito")
