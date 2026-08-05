"""Análisis comparativo de erosión para el set de 49 repeticiones.

Reutiliza las funciones de ``erosion`` (set de 9) para comparar resultados
estadísticos, generar tablas LaTeX y animaciones de la evolución temporal.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from uncertainties import unumpy as unp

import erosion as erosion_9

warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings(
    "ignore", message="Covariance of the parameters could not be estimated"
)

MK_SIZE = 13
ROT_COLS = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]

FONT_AXES_LABEL = 15
FONT_TICK_LABEL = 13
FONT_LEGEND = 15
FONT_TITLE = 16
FONT_SUBPLOT_LABEL = 14
FONT_ANNOTATION = 20
FONT_HEATMAP_CELL = 8
FONT_HEATMAP_ANNOT = 13
FONT_COMP_LEGEND = 12

CSV_PATH = Path("datos/resultados_49_mov/merge_resultados.csv").expanduser()

# Mapping de (np, ángulo) a marcadores
MARKER_STYLES = {
    (1, 0): "o",
}


def _marker_kwargs(n, angle):
    kwargs = {"marker": MARKER_STYLES[(n, angle)]}
    return kwargs


def _make_hollow_markers(ax):
    for line in ax.lines:
        line.set_markerfacecolor("none")
        line.set_markeredgecolor(line.get_color())
        line.set_markeredgewidth(1.5)
    for collection in ax.collections:
        edge = collection.get_edgecolors()
        color = collection.get_facecolors()
        collection.set_facecolor("none")
        collection.set_edgecolor(
            color if len(edge) == 0 or (len(edge) > 0 and edge[0, 3] == 0) else edge
        )
        collection.set_linewidth(1.5)


def _mpl_scatter_with_se(ax, data, x, y, hue, marker, markersize=MK_SIZE, capsize=5):
    palette = [plt.cm.tab10(i) for i in range(10)]
    hues = sorted(data[hue].unique())
    for idx, h in enumerate(hues):
        sub = data[data[hue] == h]
        grouped = sub.groupby(x)[y]
        means = grouped.mean()
        ses = grouped.sem(ddof=1)
        color = palette[idx]
        ax.errorbar(
            means.index.values,
            means.values,
            yerr=ses.values,
            fmt=marker,
            markersize=markersize,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.5,
            color=color,
            linestyle="",
            capsize=capsize,
            capthick=1.5,
            elinewidth=1.5,
        )


def _ff_marker_handles():
    palette = [plt.cm.tab10(i) for i in range(10)]
    markers = ["o"]
    case_labels = [r"$N_p: 1$"]
    phi_vals = [0.15]
    handles, labels = [], []
    for phi_val in phi_vals:
        handles.append(Line2D([0], [0], color="none", linewidth=0))
        labels.append(rf"$\phi={phi_val}$")
    for marker, lbl in zip(markers, case_labels):
        for phi_idx, phi_val in enumerate(phi_vals):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=palette[phi_idx],
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    markersize=MK_SIZE,
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


def plot_heatmap(sim, ff, v, np, angle, base_grid, rot=None):
    i_max = sim["i"].max()
    erosion_mask = (sim["i"] == i_max) & (sim["proyectil_erosion"] != False)
    erosion_count = sim[erosion_mask].groupby(["grid_x", "grid_y"])["id"].nunique()
    yields = erosion_count.rename("yield")
    res = base_grid.join(yields)
    res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0

    valores_reales = res.loc[res["existe"], "yield"].reset_index(drop=True)

    # Generar Heatmap
    matrix = res["yield"].unstack(level="grid_x").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", origin="upper")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{int(x)}" for x in matrix.columns], rotation=0)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{int(y)}" for y in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=FONT_HEATMAP_CELL)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Yield")
    rot_suffix = (
        f"_rot{rot['rot_angulo']}_{rot['rot_eje_x']}{rot['rot_eje_y']}{rot['rot_eje_z']}"
        if rot
        else ""
    )
    ax.set_title(f"Erosión (ff={ff}, v={v}, np={np}, angle={angle}{rot_suffix})")
    ax.set_xlabel(r"posx ($\sigma /2$)")
    ax.set_ylabel(r"posy ($\sigma /2$)")
    plt.savefig(
        f"figuras/heatmap_ff_{ff}_v{v}_np{np}_angle{angle}{rot_suffix}.png",
        bbox_inches="tight",
    )
    plt.close(fig)
    df_out = pd.DataFrame(
        {
            "vel0": abs(v),
            "yield": valores_reales,
            "ff": ff,
            "np": np,
            "angulo": angle,
        }
    )
    if rot is not None:
        for k, val in rot.items():
            df_out[k] = val
    return df_out


def plot_erosion_loglog(casos, df_dist):
    palette = [plt.cm.tab10(i) for i in range(10)]

    fig, axes = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
    axes = np.atleast_1d(axes)

    for i, (n, alfa) in enumerate(casos):
        ax = axes[i]
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})
        df_plot[r"$v$"] = df_plot["vel0"].abs()

        m_kw = _marker_kwargs(n, alfa)
        _mpl_scatter_with_se(ax, df_plot, r"$v$", "yield", r"$\phi$", m_kw["marker"])

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
                    bounds=([0, 0, 0], [np.inf, 10, 4]),
                )
                _, _, c = popt
                x_sm = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
                ax.plot(
                    x_sm,
                    power_law(x_sm, *popt),
                    ":",
                    color=palette[idx],
                    alpha=0.7,
                    linewidth=1.8,
                )
                ax.text(
                    0.97,
                    0.03 + idx * 0.055,
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

        _style_subplot(ax, n, alfa, i == 0)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylim(bottom=1e-10)
        ax.set_xlabel(r"$v$ [$m/s$]", fontsize=FONT_TITLE)
        ax.set_ylabel(r"$Y$", fontsize=FONT_TITLE)
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig("figuras/plot_erosion_loglog.png", bbox_inches="tight")
    plt.close(fig)


def plot_erosion_distribution(casos, df_dist):
    palette = [plt.cm.tab10(i) for i in range(10)]
    ff_vals = sorted(df_dist["ff"].unique())

    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
    ax1 = np.atleast_1d(ax1)
    filas = []

    for i, (n, alfa) in enumerate(casos):
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})
        df_plot[r"$v$"] = df_plot["vel0"].abs()
        m_kw = _marker_kwargs(n, alfa)
        _mpl_scatter_with_se(ax1[i], df_plot, r"$v$", "yield", r"$\phi$", m_kw["marker"])

        if ax1[i].get_legend() is not None:
            ax1[i].get_legend().remove()

        c_vals = {}
        for idx, ff in enumerate(ff_vals):
            ff_data = df_plot[df_plot[r"$\phi$"] == ff]
            x_fit = ff_data[r"$v$"].values
            y_fit = ff_data["yield"].values

            try:
                popt, pcov = curve_fit(
                    power_law,
                    x_fit,
                    y_fit,
                    p0=[1, 2, 1],
                    maxfev=5000,
                    bounds=([0, 0, 0], [np.inf, 10, 4]),
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
                    alpha=0.7,
                    linewidth=1.8,
                )
                ax1[i].text(
                    0.03,
                    0.97 - idx * 0.055,
                    rf"$c={c:.2f}$",
                    transform=ax1[i].transAxes,
                    color=palette[idx],
                    fontsize=FONT_ANNOTATION,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )
            except (RuntimeError, ValueError):
                popt = [np.nan, np.nan, np.nan]
                perr = [np.nan, np.nan, np.nan]
                a, b, c = np.nan, np.nan, np.nan
                c_vals[ff] = np.nan

            ang_s = "H" if alfa == 0 else "V"
            filas.append((n, ang_s, ff, *popt, *perr))

        _style_subplot(ax1[i], n, alfa, i == 0)
        ax1[i].set_ylim(-2, 36.5)
        ax1[i].set_xlabel(r"$v$ [$m/s$]", fontsize=FONT_TITLE)
        ax1[i].set_ylabel(r"$Y$", fontsize=FONT_TITLE)
        ax1[i].tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles()
    fig1.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
    )

    plt.tight_layout()
    fig1.subplots_adjust(bottom=0.14)
    fig1.savefig("figuras/plot_power_law_todos.png", bbox_inches="tight")
    plt.close(fig1)


def _style_subplot(ax, n, alfa, show_ylabel):
    ax.set_title(f"$N_p$: {n}", fontsize=FONT_TITLE)
    ax.set_xlabel(r"v [$m/s$]", fontsize=FONT_SUBPLOT_LABEL)
    ax.set_ylabel(r"$Y$", fontsize=FONT_SUBPLOT_LABEL) if show_ylabel else ax.set_ylabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()


def plot_erosion_violin(casos, df_dist):
    palette = [plt.cm.tab10(i) for i in range(10)]
    fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (n, alfa) in enumerate(casos):
        ax = axes[i]
        subset = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
        subset = subset.rename(columns={"ff": r"$\phi$"})
        subset["vel_str"] = subset["vel0"].astype(int).astype(str)

        vel_cats = sorted(subset["vel_str"].unique())
        hue_vals = sorted(subset[r"$\phi$"].unique())
        n_vels = len(vel_cats)
        n_hues = len(hue_vals)
        width = 0.8 / n_hues

        for vi, vel in enumerate(vel_cats):
            for hi, phi in enumerate(hue_vals):
                data = subset[(subset["vel_str"] == vel) & (subset[r"$\phi$"] == phi)]["yield"].values
                if len(data) < 2:
                    continue
                parts = ax.violinplot(
                    [data],
                    positions=[vi],
                    widths=width,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(palette[hi])
                    pc.set_alpha(0.6)
                    pc.set_edgecolor("black")
                    pc.set_linewidth(0.8)
                q25, median, q75 = np.percentile(data, [25, 50, 75])
                offset = (hi - (n_hues - 1) / 2) * width
                ax.plot(
                    [vi + offset, vi + offset],
                    [q25, q75],
                    color="black",
                    linewidth=1.5,
                )
                ax.plot(
                    vi + offset,
                    median,
                    color="white",
                    markersize=4,
                    marker="D",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )

        ax.set_xticks(range(n_vels))
        ax.set_xticklabels(vel_cats)
        ax.set_title(rf"$N_p$: {n}", fontsize=FONT_TITLE)
        ax.set_xlabel(r"$v$ [m/s]")
        if i == 0:
            ax.set_ylabel(r"$Y$")
        if ax.get_legend() is not None:
            if i != 2:
                ax.get_legend().remove()
    global_min = df_dist["yield"].min() - 1
    global_max = df_dist["yield"].max() + 1
    for ax in np.atleast_1d(axes):
        ax.set_ylim(global_min, global_max)
    plt.tight_layout()
    plt.savefig("figuras/plot_violin_erosion.png", bbox_inches="tight")
    plt.close()


def plot_model_comparison(casos, df_cluster, df):
    dfs = _prepare_cluster_data(casos, df_cluster, df)

    modelos_plot = {
        "Power law": power_law_cluster,
        "Aström": astrom_model,
    }
    colors = {"Power law": "C0", "Aström": "C1"}
    linestyles = {"Power law": "-", "Aström": "--"}
    p0 = {
        "Power law": [1.8],
        "Aström": [1, 0.3, 2, 15, 10, 2],
    }
    bounds = {
        "Power law": ([0.8], [2]),
        "Aström": ([0, 0, 0, 1e-6, 1e-6, 1], [np.inf, 1, 5, 100, 100, 10]),
    }
    palette = [plt.cm.tab10(i) for i in range(10)]

    velocities = sorted(set(v for dfv in dfs for v in dfv["v"].unique()))
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
            len(casos),
            figsize=(5 * len(casos), 5 * len(velocities)),
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

                m_kw = _marker_kwargs(n, alfa)
                m_kw["markerfacecolor"] = "none"
                m_kw["markeredgewidth"] = 3

                stats = (
                    df_v.groupby(["count", r"$\phi$"])["F(n)_j"]
                    .sum()
                    .reset_index(name="N_k")
                )
                phi_sums = stats.groupby(r"$\phi$")["N_k"].transform("sum")
                stats["mean"] = np.where(phi_sums > 0, stats["N_k"] / phi_sums, 0.0)
                stats["mean_plot"] = stats["mean"].replace(0, np.nan)

                phi_vals = sorted(df_v[r"$\phi$"].unique())
                for idx, phi_val in enumerate(phi_vals):
                    subset = stats[stats[r"$\phi$"] == phi_val].sort_values("count")
                    valid = subset[subset["mean_plot"].notna() & (subset["mean_plot"] > 0)]

                    ax.plot(
                        valid["count"].values,
                        valid["mean_plot"].values,
                        color=palette[idx % len(palette)],
                        markersize=MK_SIZE,
                        linestyle="",
                        **m_kw,
                    )

                    x_fit = valid["count"].values
                    y_fit = valid["mean_plot"].values
                    if len(x_fit) < 3:
                        continue

                    for nombre_modelo, modelo in modelos_plot.items():
                        try:
                            popt, _ = curve_fit(
                                modelo,
                                x_fit,
                                y_fit,
                                p0=p0[nombre_modelo],
                                maxfev=10000,
                                bounds=bounds[nombre_modelo],
                            )
                            x_sm = np.logspace(
                                np.log10(x_fit.min()), np.log10(x_fit.max()), 200
                            )
                            ax.plot(
                                x_sm,
                                modelo(x_sm, *popt),
                                linestyle=linestyles[nombre_modelo],
                                color=colors[nombre_modelo],
                                alpha=0.7,
                                linewidth=1.8,
                            )
                        except (RuntimeError, ValueError):
                            pass

                _style_subplot(ax, n, alfa, j == 0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left=0.95, right=41)
                ax.set_ylim(2.1e-4, 1.01)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.grid(axis="x", linestyle="--", alpha=0.7)
                ax.grid(which="minor", linestyle=":", alpha=0.3)
                ax.minorticks_on()

                if i == 0:
                    ax.set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")

        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=colors[name], linestyle=linestyles[name], label=name)
            for name in modelos_plot
        ]
        handles_ff, labels_ff = _ff_marker_handles()
        handles += handles_ff
        labels = list(modelos_plot.keys()) + labels_ff
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.savefig("figuras/plot_PDF_alternativos.png", bbox_inches="tight")
        plt.close(fig)


def survival_cumsum(modelo, k, *params, k_max, n_grid=1000):
    k = np.asarray(k, dtype=float)
    scalar = np.ndim(k) == 0
    k_arr = np.atleast_1d(k)

    k_grid = np.linspace(1, k_max, n_grid)
    dk = k_grid[1] - k_grid[0]
    pdf_vals = modelo(k_grid, *params)
    pdf_vals = np.maximum(pdf_vals, 0)
    cdf = np.cumsum(pdf_vals) * dk
    cdf_norm = cdf / cdf[-1]
    surv = np.where(k_arr <= k_max, 1.0 - np.interp(k_arr, k_grid, cdf_norm), 0.0)

    return surv.item() if scalar else surv


def survival_power_law(k, K_max):
    k = np.asarray(k, dtype=float)
    num = K_max ** (1 - 2.8) - k ** (1 - 2.8)
    den = K_max ** (1 - 2.8) - 1
    return np.where(k <= K_max, np.maximum(num / den, 0.0), 0.0)


def plot_model_survival_comparison(casos, df_cluster, df):
    dfs = _prepare_cluster_data(casos, df_cluster, df)

    def _make_survival_model(modelo):
        return lambda k, *p: survival_cumsum(modelo, k, *p[:-1], k_max=p[-1])

    survival_models = {
        "Power law": survival_power_law,
        "Aström": _make_survival_model(astrom_model),
    }

    colors = {"Power law": "C0", "Aström": "C1"}
    linestyles = {"Power law": "-", "Aström": "--"}
    p0 = {
        "Power law": [300],
        "Aström": [1, 0.3, 2, 15, 10, 2, 100],
    }
    bounds = {
        "Power law": ([300], [310]),
        "Aström": ([0, 0, 0, 1e-6, 1e-6, 1, 41], [np.inf, 1, 5, 100, 100, 10, 600]),
    }
    palette = [plt.cm.tab10(i) for i in range(10)]

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
            1,
            len(casos),
            figsize=(5 * len(casos), 5),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        for j, (n, alfa) in enumerate(casos):
            ax = axes[j]
            df_v = dfs[j].copy()
            if df_v.empty:
                ax.set_visible(False)
                continue

            m_kw = _marker_kwargs(n, alfa)
            m_kw["markerfacecolor"] = "none"
            m_kw["markeredgewidth"] = 3

            df_v["P_exact"] = df_v["F(n)_j"] / df_v["M_t"]
            group_cols = ["posx", "posy", r"$\phi$", "v"] + ROT_COLS
            df_v = df_v.sort_values(["count"] + group_cols)
            df_v["P_surv"] = df_v.groupby(group_cols)["P_exact"].transform(
                lambda x: 1 - x.cumsum().shift(1, fill_value=0)
            )

            stats_surv = (
                df_v.groupby(["count", r"$\phi$"])["P_surv"]
                .agg(mean="mean", se="sem")
                .reset_index()
            )

            phi_vals = sorted(df_v[r"$\phi$"].unique())
            for idx, phi_val in enumerate(phi_vals):
                subset = stats_surv[stats_surv[r"$\phi$"] == phi_val].sort_values(
                    "count"
                )
                valid = subset[subset["mean"] > 0]

                ax.plot(
                    valid["count"].values,
                    valid["mean"].values,
                    color=palette[idx % len(palette)],
                    markersize=MK_SIZE,
                    linestyle="",
                    **m_kw,
                )

                # Fit CCDF models directly to survival data
                x_surv = valid["count"].values
                y_surv = valid["mean"].values
                if len(x_surv) < 3:
                    continue

                for nombre_modelo, surv_model in survival_models.items():
                    try:
                        popt, _ = curve_fit(
                            surv_model,
                            x_surv,
                            y_surv,
                            p0=p0[nombre_modelo],
                            maxfev=10000,
                            bounds=bounds[nombre_modelo],
                        )

                        k_smooth = np.linspace(1, 41, 200)
                        model_surv = surv_model(k_smooth, *popt)

                        ax.plot(
                            k_smooth,
                            model_surv,
                            linestyle=linestyles[nombre_modelo],
                            color=colors[nombre_modelo],
                            alpha=0.7,
                            linewidth=1.8,
                        )
                    except (RuntimeError, ValueError):
                        pass

            ax.set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            if j == 0:
                ax.set_ylabel(r"$P(K \geq k)$")
            ax.set_xlabel("k")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=0.95, right=41)
            ax.set_ylim(bottom=2.1e-4, top=1.01)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.grid(which="minor", linestyle=":", alpha=0.3)
            ax.minorticks_on()

        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=colors[name], linestyle=linestyles[name], label=name)
            for name in survival_models
        ]
        handles_ff, labels_ff = _ff_marker_handles()
        handles += handles_ff
        labels = list(survival_models.keys()) + labels_ff
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(
            "figuras/plot_model_survival.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _prepare_cluster_data(casos, df_dist, df):
    dfs = []
    for n, alfa in casos:
        subset = df_dist[(df_dist["angulo"] == alfa) & (df_dist["np"] == n)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})

        group_cols = [
            "posx",
            "posy",
            "np",
            "angulo",
            r"$\phi$",
            "count",
            "v",
        ] + ROT_COLS
        df_por_impacto = df_plot.groupby(group_cols).size().reset_index(name="F(n)_j")
        df_tamaños_unicos = sorted(df_plot["count"].unique())
        mt_group_cols = ["posx", "posy", r"$\phi$", "v"] + ROT_COLS
        df_por_impacto["M_t"] = df_por_impacto.groupby(mt_group_cols)[
            "F(n)_j"
        ].transform("sum")

        base_grid, df_grid = grilla(n, df)
        grid_existe = df_grid[["posx", "posy"]].drop_duplicates()

        pos_phi = grid_existe[["posx", "posy"]].merge(
            df_plot[[r"$\phi$"]].drop_duplicates(), how="cross"
        )
        all_sizes = pd.DataFrame({"count": df_tamaños_unicos})
        all_combos = pos_phi.merge(all_sizes, how="cross")
        all_combos = all_combos.merge(df_plot[["v"]].drop_duplicates(), how="cross")
        rot_combos = df_plot[ROT_COLS].drop_duplicates()
        all_combos = all_combos.merge(rot_combos, how="cross")
        merge_cols = ["posx", "posy", r"$\phi$", "count", "v"] + ROT_COLS
        df_por_impacto = all_combos.merge(df_por_impacto, how="left", on=merge_cols)
        all_cols = ["posx", "posy", "v", r"$\phi$", "count"] + ROT_COLS
        df_por_impacto = df_por_impacto.sort_values(all_cols)

        df_por_impacto["F(n)_j"] = df_por_impacto["F(n)_j"].fillna(0)
        df_por_impacto["M_t"] = df_por_impacto.groupby(mt_group_cols)[
            "F(n)_j"
        ].transform("sum")
        dfs.append(df_por_impacto)
    return dfs


def _plot_distribution_panel(df_por_impacto, y_col, ax, n, alfa):
    palette = [plt.cm.tab10(i) for i in range(10)]
    m_kw = _marker_kwargs(n, alfa)
    m_kw["markerfacecolor"] = "none"
    m_kw["markeredgewidth"] = 3

    stats = (
        df_por_impacto.groupby(["count", r"$\phi$"])["F(n)_j"]
        .sum()
        .reset_index(name="N_k")
    )
    phi_sums = stats.groupby(r"$\phi$")["N_k"].transform("sum")
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
            x_trend = np.logspace(0, np.log10(41), 100)
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
    ax.set_xlim(left=0.95, right=41)
    ax.set_yscale("log")
    ax.set_ylim(2.1e-4, 1.01)
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
            len(casos),
            figsize=(5 * len(casos), 5 * len(velocities)),
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
                    ax.set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")

        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=4,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.11)
        fig.savefig("figuras/plot_PDF_por_vel_cluster.png", bbox_inches="tight")
        plt.close(fig)


def plot_cluster_survival(casos, df_dist, df):
    dfs = _prepare_cluster_data(casos, df_dist, df)

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
            1,
            len(casos),
            figsize=(5 * len(casos), 5),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        for j, (n, alfa) in enumerate(casos):
            ax = axes[j]
            df_v = dfs[j].copy()
            if df_v.empty:
                ax.set_visible(False)
                continue

            palette = [plt.cm.tab10(i) for i in range(10)]
            m_kw = _marker_kwargs(n, alfa)
            m_kw["markerfacecolor"] = "none"
            m_kw["markeredgewidth"] = 3

            df_v["P_exact"] = df_v["F(n)_j"] / df_v["M_t"]
            group_cols = ["posx", "posy", r"$\phi$", "v"] + ROT_COLS
            df_v = df_v.sort_values(["count"] + group_cols)
            df_v["P_surv"] = df_v.groupby(group_cols)["P_exact"].transform(
                lambda x: 1 - x.cumsum().shift(1, fill_value=0)
            )

            stats = (
                df_v.groupby(["count", r"$\phi$"])["P_surv"]
                .agg(mean="mean", se="sem")
                .reset_index()
            )
            stats["mean_plot"] = stats["mean"].replace(0, np.nan)

            phi_vals = sorted(df_v[r"$\phi$"].unique())
            for idx, phi_val in enumerate(phi_vals):
                subset = stats[stats[r"$\phi$"] == phi_val].sort_values("count")
                valid = subset[subset["mean"] > 0]
                if valid.empty:
                    continue
                ax.plot(
                    valid["count"].values,
                    valid["mean"].values,
                    label=phi_val,
                    color=palette[idx % len(palette)],
                    markersize=MK_SIZE,
                    linestyle="",
                    **m_kw,
                )
                x_trend = np.logspace(0, np.log10(41), 100)
                y_trend = x_trend ** (-1.8)
                ax.plot(
                    x_trend,
                    y_trend,
                    "--",
                    color=palette[idx],
                    alpha=0.5,
                    linewidth=1.5,
                )

            ax.set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            if j == 0:
                ax.set_ylabel(r"$P(K > k)$")
            ax.set_xlabel("k")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=0.95, right=41)
            ax.set_ylim(bottom=2.1e-4, top=1.01)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.grid(which="minor", linestyle=":", alpha=0.3)
            ax.minorticks_on()

        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=4,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig("figuras/plot_cluster_survival.png", bbox_inches="tight")
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
        ax = np.atleast_1d(ax)
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
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"] + ROT_COLS, as_index=False)[
                "depth"
            ].max()
            m_kw = _marker_kwargs(n, alfa)
            _mpl_scatter_with_se(ax[i], df, "vel0", "depth", r"$\phi$", m_kw["marker"])
            ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"d [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
        )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
        ax = np.atleast_1d(ax)
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
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"] + ROT_COLS, as_index=False)[
                "depth"
            ].mean()
            m_kw = _marker_kwargs(n, alfa)
            _mpl_scatter_with_se(ax[i], df, "vel0", "depth", r"$\phi$", m_kw["marker"])

            ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"d [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
        )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
        ax = np.atleast_1d(ax)
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[(df["type"] == 1) & (df["i"] == 0) & (df["z"] < 100e-6)]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"] + ROT_COLS, as_index=False)[
                "depth"
            ].max()
            m_kw = _marker_kwargs(n, alfa)
            _mpl_scatter_with_se(ax[i], df, "vel0", "depth", r"$\phi$", m_kw["marker"])
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"d [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
        )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharey=True)
        ax = np.atleast_1d(ax)
        for i, (n, alfa) in enumerate(casos):
            df = df_dist[(df_dist["np"] == n) & (df_dist["angulo"] == alfa)].copy()
            df = df.rename(columns={"ff": r"$\phi$"})
            df = df[(df["type"] == 1) & (df["i"] == 0) & (df["z"] < 100e-6)]
            df["z"] = df["z"] * 1e6
            df["depth"] = 100 - df["z"]
            df["vel0"] = df["vel0"].abs()
            df = df.groupby(["vel0", "posx", "posy", r"$\phi$"] + ROT_COLS, as_index=False)[
                "depth"
            ].mean()
            m_kw = _marker_kwargs(n, alfa)
            _mpl_scatter_with_se(ax[i], df, "vel0", "depth", r"$\phi$", m_kw["marker"])

            # ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(rf"$N_p$: {n}", fontsize=round(FONT_TITLE * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(r"d [$\mu m$]")
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=0.7)
            ax[i].grid(axis="x", linestyle="--", alpha=0.7)
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()
            if ax[i].get_legend() is not None:
                ax[i].get_legend().remove()
        handles, labels = _ff_marker_handles()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
        )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(
        "figuras/plot_mean_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
        bbox_inches="tight",
    )
    plt.close()


def astrom_model(v, A, beta_r, alpha, lambda_param, v_uncorr, D):
    v = np.maximum(v, 1e-12)

    cutoff_term = ((2.0 / lambda_param) ** D) * v
    term1 = (1.0 - beta_r) * (v**-alpha) * np.exp(-cutoff_term)

    poisson_term = (1.0 / v_uncorr) * ((v ** (1.0 / D)) + lambda_param) ** D
    term2 = beta_r * np.exp(-poisson_term)

    return A * (term1 + term2)


def power_law(x, a, b, c):
    b = b if abs(b) > 1e-6 else 0.0
    z = x - b * 0
    mask = z > 0
    result = np.zeros_like(z)
    result[mask] = a * (z[mask] / 0.17) ** c
    return result


def power_law_cluster(k, a):
    return a * k ** (-2.8)


def ajuste_modelos(subset, n, x_col="vel0", y_col="yield"):
    raw_results = []
    formatted_results = []

    grouped = subset.groupby(x_col)[y_col]
    means = grouped.mean()
    sems = grouped.sem(ddof=1)
    mask = (sems >= 0) & (~np.isnan(sems))
    x_fit = means.index[mask].values
    y_fit = means.values[mask]

    # --- Power law ---
    n_params = 3
    dof = len(x_fit) - n_params
    t_val = t.ppf(0.975, max(dof, 1)) if dof > 0 else 1.0

    try:
        popt, pcov = curve_fit(
            power_law,
            x_fit,
            y_fit,
            p0=[1, 2, 1],
            maxfev=5000,
            bounds=([0, 0, 0], [np.inf, 10, 4]),
        )
        err = np.sqrt(np.diag(pcov)) * t_val
    except (RuntimeError, ValueError):
        popt = np.full(3, np.nan)
        err = np.full(3, np.nan)
    raw_results.append(popt)
    formatted_results.append([f"{v}" for v in unp.uarray(popt, err)])

    # --- Astrom model ---
    n_params = 6
    dof = len(x_fit) - n_params
    t_val = t.ppf(0.975, max(dof, 1)) if dof > 0 else 1.0

    try:
        popt, pcov = curve_fit(
            astrom_model,
            x_fit,
            y_fit,
            p0=[1, 0.5, 1, 10, 10, 2],
            maxfev=10000,
            bounds=([0, 0, 0, 1e-6, 1e-6, 1], [np.inf, 1, 5, 100, 100, 10]),
        )
        err = np.sqrt(np.diag(pcov)) * t_val
    except (RuntimeError, ValueError):
        popt = np.full(6, np.nan)
        err = np.full(6, np.nan)
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


def extract_cluster_data(df, np_val, angle, ff, v, rot=None):
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

    out = df_cluster.assign(
        km=km, rg=rg, angulo_vel=angulo_vel, np=np_val, angulo=angle, ff=ff, v=v
    ).drop(columns=["var_x", "var_y", "var_z"])
    if rot is not None:
        for k, val in rot.items():
            out[k] = val
    return out


def extract_main_data(df, casos):
    datos_distribucion = []
    datos_cluster = []
    datos_proyectil = []
    for n, alfa in casos:
        base_grid, df_grid = grilla(n, df)
        for ff in [0.15, 0.35]:
            vels = sorted(df_grid.loc[df_grid["ff"] == ff, "vel0"].abs().unique())
            for v in vels:
                mask_base = (
                    (df_grid["ff"] == ff)
                    & (df_grid["vel0"] == -1 * v)
                    & (df_grid["np"] == n)
                    & (df_grid["angulo"] == alfa)
                )
                rotaciones = df_grid.loc[mask_base, ROT_COLS].drop_duplicates()
                for _, rot_row in rotaciones.iterrows():
                    rot = rot_row.to_dict()
                    mask = mask_base & (
                        (df_grid["rot_angulo"] == rot["rot_angulo"])
                        & (df_grid["rot_eje_x"] == rot["rot_eje_x"])
                        & (df_grid["rot_eje_y"] == rot["rot_eje_y"])
                        & (df_grid["rot_eje_z"] == rot["rot_eje_z"])
                    )
                    sim = df_grid[mask]
                    if not sim.empty:
                        datos_distribucion.append(
                            plot_heatmap(sim, ff, v, n, alfa, base_grid, rot)
                        )
                        datos_cluster.append(
                            extract_cluster_data(sim, n, alfa, ff, v, rot)
                        )
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
        proj.groupby(["np", "angulo", "ff", "vel0", "posx", "posy"] + ROT_COLS)[
            "proyectil_erosion"
        ]
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

    data = pivot_numerico.values
    vmin = data.min()
    vmax = data.max()
    cmap = plt.cm.viridis
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cmap_with_bad = cmap.copy()
    cmap_with_bad.set_bad(color="white")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=FONT_HEATMAP_ANNOT, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.rcParams.update({"font.size": 1.3 * 10})
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

    # 1. Filtrar DataFrames
    proj = df[
        (df["type"] == 2) & (df["i"] == i_max) & (df["proyectil_erosion"] == True)
    ].copy()
    eros = df[(df["type"] == 1) & (df["i"] == i_max)].copy()

    # 2. Homologar velocidades (aplicar a ambos si son clave de agrupación)
    proj["vel0"] = proj["vel0"].abs()
    eros["vel0"] = eros["vel0"].abs()

    columnas_grupo = ["np", "angulo", "ff", "vel0", "posx", "posy"] + ROT_COLS

    # 3. Contar ocurrencias
    total_proyectiles = (
        proj.groupby(columnas_grupo).size().reset_index(name="count_proj")
    )
    total_erosion = eros.groupby(columnas_grupo).size().reset_index(name="count_eros")

    # 4. Unir (Merge) de forma segura para alinear las filas correctamente
    # Usamos 'outer' para mantener combinaciones que estén en uno pero no en el otro
    df_merged = pd.merge(
        total_proyectiles, total_erosion, on=columnas_grupo, how="outer"
    ).fillna(0)  # Los valores que no crucen tendrán 0 en su conteo

    # 5. Calcular porcentaje evitando divisiones por cero
    df_merged["total_eventos"] = df_merged["count_eros"] + df_merged["count_proj"]
    df_merged["porcentaje_erosion"] = np.where(
        df_merged["total_eventos"] > 0,
        (100 * df_merged["count_proj"]) / df_merged["total_eventos"],
        0,
    )

    # 6. Generar resumen
    resumen = df_merged.groupby(["np", "angulo", "ff", "vel0"], as_index=False)[
        "porcentaje_erosion"
    ].agg(mean="mean", se="sem")

    # 7. Pivotear
    pivot = resumen.pivot_table(
        index=["np", "angulo"],
        columns=["vel0", "ff"],
        values="mean",
        aggfunc="first",
    )

    # Renombrar ejes
    pivot.columns.names = [r"$v$", r"$\phi$"]
    pivot.index.names = [r"$N_p$", "Orientación"]

    # 8. Mapear V y H de forma segura, sin importar si falta alguno
    # Reconstruimos el índice mapeando los valores explícitamente
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

    columnas_grupo = ["np", "angulo", "ff", "vel0", "posx", "posy"] + ROT_COLS

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


def _depth_stats(sub_proj_v, agg_name):
    depth = sub_proj_v.groupby(["posx", "posy"] + ROT_COLS)["depth"].agg(agg_name)
    return depth.mean(), depth.sem(ddof=1)


def tabla_resumen(df_dist, df_proyectil, df, casos):
    n, alfa = casos[0]
    ff, v = 0.15, 50.0
    rows = []

    sub = df_dist[
        (df_dist["np"] == n)
        & (df_dist["angulo"] == alfa)
        & (df_dist["ff"] == ff)
        & (df_dist["vel0"] == v)
    ]
    n_reps = sub["yield"].count()

    rows.append((r"$N_p$", f"{n}"))
    rows.append((r"$v$", f"${v:.0f}\\,m/s$"))
    rows.append((r"$\phi$", f"{ff}"))
    rows.append(("Repeticiones", f"{n_reps}"))

    # ---- Yield ----
    y = sub["yield"]
    rows.append((r"$Y$", f"${y.mean():.3f}\\pm{y.sem(ddof=1):.3f}$"))

    # ---- Depth ----
    proj = df_proyectil[
        (df_proyectil["np"] == n)
        & (df_proyectil["angulo"] == alfa)
        & (df_proyectil["ff"] == ff)
    ].copy()
    for label, tipo, agg_name in [
        (r"$d_{\max}$", "proyectil", "max"),
        (r"$d_{mean}$", "proyectil", "mean"),
        (r"$D_{\max}$", "lecho", "max"),
        (r"$D_{mean}$", "lecho", "mean"),
    ]:
        df_v = proj[proj["vel0"].abs() == v].copy()
        if tipo == "proyectil":
            filtro = (
                (df_v["proyectil_erosion"] != 1)
                & (df_v["i"] == 80)
                & (df_v["z"] < 100e-6)
            )
        else:
            filtro = (df_v["type"] == 1) & (df_v["i"] == 0) & (df_v["z"] < 100e-6)
        df_filt = df_v[filtro].copy()
        if df_filt.empty:
            rows.append((label, r"---"))
        else:
            df_filt["depth"] = 100 - df_filt["z"] * 1e6
            m, s = _depth_stats(df_filt, agg_name)
            rows.append((label, f"${m:.2f}\\pm{s:.2f}\\,\\mu m$"))

    # ---- Proporciones ----
    _, df_rebote = tabla_rebote_proyectil(df)
    sub_r = df_rebote[
        (df_rebote["np"] == n)
        & (df_rebote["angulo"] == alfa)
        & (df_rebote["ff"] == ff)
        & (df_rebote["vel0"] == v)
    ]
    rm, rs = sub_r["probabilidad"].iloc[0], sub_r["se"].iloc[0]
    rows.append((r"Proporci\'on de masa", f"${rm * 100:.1f}\\pm{rs * 100:.1f}\\,\\%$"))

    _, df_erosion = tabla_porcentaje_proyectil(df)
    sub_e = df_erosion[
        (df_erosion["np"] == n)
        & (df_erosion["angulo"] == alfa)
        & (df_erosion["ff"] == ff)
        & (df_erosion["vel0"] == v)
    ]
    em, es = sub_e["mean"].iloc[0], sub_e["se"].iloc[0]
    rows.append((r"Proporci\'on de proyectil", f"${em:.1f}\\pm{es:.1f}\\,\\%$"))

    lines = [r"\begin{tabular}{lc}"]
    for i, (c1, c2) in enumerate(rows):
        if i == 4:
            lines.append(r"\hline")
        lines.append(f"{c1} & {c2} \\\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _extract_summary_values(df_dist, df_proyectil, df, casos, depth_stats_fn):
    ff, v = 0.15, 50.0
    n, alfa = casos[0]
    values = {}

    sub = df_dist[
        (df_dist["np"] == n)
        & (df_dist["angulo"] == alfa)
        & (df_dist["ff"] == ff)
        & (df_dist["vel0"] == v)
    ]
    values["reps"] = sub["yield"].count()
    y = sub["yield"]
    values["Y"] = (y.mean(), y.sem(ddof=1))

    proj = df_proyectil[
        (df_proyectil["np"] == n)
        & (df_proyectil["angulo"] == alfa)
        & (df_proyectil["ff"] == ff)
    ].copy()
    df_v = proj[proj["vel0"].abs() == v].copy()

    for label, key, tipo, agg_name in [
        (r"$d_{mean}$ [\(\mu\text{m}\)]", "d_mean", "proyectil", "mean"),
        (r"$D_{max}$ [\(\mu\text{m}\)]", "D_max", "lecho", "max"),
        (r"$D_{mean}$ [\(\mu\text{m}\)]", "D_mean", "lecho", "mean"),
    ]:
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
            values[key] = None
        else:
            df_filt["depth"] = 100 - df_filt["z"] * 1e6
            m, s = depth_stats_fn(df_filt, agg_name)
            values[key] = (m, s)

    _, df_rebote = tabla_rebote_proyectil(df)
    sub_r = df_rebote[
        (df_rebote["np"] == n)
        & (df_rebote["angulo"] == alfa)
        & (df_rebote["ff"] == ff)
        & (df_rebote["vel0"] == v)
    ]
    rm, rs = sub_r["probabilidad"].iloc[0], sub_r["se"].iloc[0]
    values["masa"] = (rm * 100, rs * 100)

    _, df_erosion = tabla_porcentaje_proyectil(df)
    sub_e = df_erosion[
        (df_erosion["np"] == n)
        & (df_erosion["angulo"] == alfa)
        & (df_erosion["ff"] == ff)
        & (df_erosion["vel0"] == v)
    ]
    em, es = sub_e["mean"].iloc[0], sub_e["se"].iloc[0]
    values["proyectil"] = (em, es)

    return values


def tabla_comparativa_latex(df_dist_9, df_proyectil_9, df_9,
                            df_dist_245, df_proyectil_245, df_245):
    v_9 = _extract_summary_values(
        df_dist_9, df_proyectil_9, df_9, [(1, 0)], erosion_9._depth_stats
    )
    v_245 = _extract_summary_values(
        df_dist_245, df_proyectil_245, df_245, [(1, 0)], _depth_stats
    )

    def _fmt(val):
        if val is None:
            return "---"
        return f"${val[0]:.2f}\\pm{val[1]:.2f}$"

    rows = [
        ("Variable", "9 repeticiones", "245 repeticiones"),
        ("Repeticiones", str(v_9["reps"]), str(v_245["reps"])),
        ("", "", ""),
        (r"$Y$", _fmt(v_9["Y"]), _fmt(v_245["Y"])),
        (r"$d_{mean}$ [\(\mu\text{m}\)]", _fmt(v_9["d_mean"]), _fmt(v_245["d_mean"])),
        (r"$D_{max}$ [\(\mu\text{m}\)]", _fmt(v_9["D_max"]), _fmt(v_245["D_max"])),
        (r"$D_{mean}$ [\(\mu\text{m}\)]", _fmt(v_9["D_mean"]), _fmt(v_245["D_mean"])),
        (r"Proporci\'on de masa [\%]", _fmt(v_9["masa"]), _fmt(v_245["masa"])),
        (r"Proporci\'on de proyectil [\%]", _fmt(v_9["proyectil"]), _fmt(v_245["proyectil"])),
    ]

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Comparación de métricas para 9 y 245 repeticiones en la configuración \(N_p:1\), \(v=50\,\text{m/s}\), \(\phi=0.15\).}",
        r"    \label{tab:estadistica_comparativa}",
        r"    \begin{tabular}{lcc}",
        r"    \hline",
    ]
    for i, row in enumerate(rows):
        if i == 2:
            lines.append(r"    \hline")
        if row[0] == "":
            lines.append(r"    \hline")
        else:
            lines.append(f"    {' & '.join(row)} \\\\")
    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _compute_pdf_stats(df_por_impacto, phi_target):
    stats = (
        df_por_impacto.groupby(["count", r"$\phi$"])["F(n)_j"]
        .sum()
        .reset_index(name="N_k")
    )
    phi_sums = stats.groupby(r"$\phi$")["N_k"].transform("sum")
    stats["mean"] = np.where(phi_sums > 0, stats["N_k"] / phi_sums, 0.0)
    stats["mean_plot"] = stats["mean"].replace(0, np.nan)
    subset = stats[stats[r"$\phi$"] == phi_target].sort_values("count")
    return subset[subset["mean_plot"].notna() & (subset["mean_plot"] > 0)]


def plot_cluster_pdf_comparativo(dfs_9, dfs_49, v_target=50, ff_target=0.15):
    palette = ["#DC143C", "#00A86B"]
    fig, ax = plt.subplots(figsize=(7, 5))

    datasets = [
        (dfs_9, "9 repeticiones", "o", palette[0]),
        (dfs_49, "245 repeticiones", "s", palette[1]),
    ]

    norm = None
    for dfs, label, marker, color in datasets:
        for df_full in dfs:
            df_v = df_full[df_full["v"] == v_target].copy()
            if df_v.empty:
                continue
            valid = _compute_pdf_stats(df_v, ff_target)
            ax.plot(
                valid["count"].values,
                valid["mean_plot"].values,
                marker=marker,
                markersize=MK_SIZE,
                linestyle="",
                color=color,
                markerfacecolor="none",
                markeredgewidth=3,
                label=label,
            )
            if label == "9 repeticiones":
                p_k1 = valid.loc[valid["count"] == 1, "mean_plot"]
                if not p_k1.empty:
                    norm = float(p_k1.iloc[0])

    x_trend = np.logspace(0, np.log10(41), 100)
    y_trend = x_trend ** (-2.8)
    if norm is not None:
        y_trend = y_trend * norm
    ax.plot(x_trend, y_trend, "--", color="black", alpha=1, linewidth=1.5, label=r"$k^{-2.8}$")

    ax.set_xscale("log")
    ax.set_xlim(left=0.95, right=41)
    ax.set_yscale("log")
    ax.set_ylim(2.1e-4, 1.01)
    ax.set_xlabel("k")
    ax.set_ylabel(f"v = {v_target:.0f} m/s\n$P(k)$")
    ax.legend(fontsize=FONT_COMP_LEGEND, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()

    fig.subplots_adjust(bottom=0.14)
    plt.tight_layout()
    fig.savefig("figuras/plot_PDF_comparativo.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)
    df["np"] = df["np"].replace(3, 1)
    df["angulo"] = df["angulo"].replace(90, 0)
    # 2. Análisis iterativo
    # Grfico 1x3 con np, angulo
    casos = [(1, 0)]
    combinaciones = [
        (1, 0, 0.15),
    ]
    df_dist, df_cluster, df_proyectil = extract_main_data(df, casos)

    #########################################
    ######## Distribución de Erosión ########
    #########################################
    # Ajustes por combinación (n, alfa, ff)
    idx = pd.MultiIndex.from_tuples(combinaciones, names=["n", "angulo", "ff"])
    cols_ajustes = ["power_a", "power_b", "power_c"] + [
        "astrom_A",
        "astrom_beta",
        "astrom_alpha",
        "astrom_lambda",
        "astrom_vuncorr",
        "astrom_D",
    ]
    ajustes_formatted = pd.DataFrame(columns=cols_ajustes, index=idx)
    ajustes_raw = pd.DataFrame(columns=cols_ajustes, index=idx)

    for n, alfa, ff in combinaciones:
        subset = df_dist[
            (df_dist["np"] == n) & (df_dist["angulo"] == alfa) & (df_dist["ff"] == ff)
        ]
        if subset.empty:
            continue
        raw, formatted = ajuste_modelos(subset, n)
        ajustes_raw.loc[(n, alfa, ff)] = [*raw[0], *raw[1]]
        ajustes_formatted.loc[(n, alfa, ff)] = [r for sub in formatted for r in sub]
    # 1x3: línea de puntos con ambos ff por subplot (existente)

    plot_erosion_distribution(casos, df_dist)
    plot_erosion_loglog(casos, df_dist)
    plot_erosion_violin(casos, df_dist)

    ######################################################
    ######## Distribución de Tamaño de Cluster ###########
    ######################################################

    plot_cluster_size_distribution(casos, df_cluster, df)
    plot_cluster_survival(casos, df_cluster, df)

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
    ######## Tabla resumen (excepto cluster size) ###############
    #############################################################

    latex_resumen = tabla_resumen(df_dist, df_proyectil, df, casos)
    with open("datos/resumen_graficos.tex", "w") as f:
        f.write(latex_resumen)
    print("\n% --- TABLA RESUMEN ---")
    print(latex_resumen)

    #######################################################
    ######## Métricas de los ajustes ######################
    #######################################################
    #######################################################
    ######## Métricas de los ajustes ######################
    #######################################################
    modelos = {
        "power_law": power_law,
        "astrom": astrom_model,
    }

    cols_modelo = {
        "power_law": ["power_a", "power_b", "power_c"],
        "astrom": [
            "astrom_A",
            "astrom_beta",
            "astrom_alpha",
            "astrom_lambda",
            "astrom_vuncorr",
            "astrom_D",
        ],
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
    ######## Comparación de modelos (PDF alternativos) ##############
    #################################################################

    plot_model_comparison(casos, df_cluster, df)
    plot_model_survival_comparison(casos, df_cluster, df)

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

    ##################################################################
    ######## Análisis comparativo (9 vs 245 repeticiones) ###########
    ##################################################################

    df_9 = pd.read_csv(erosion_9.CSV_PATH)
    casos_9 = [(1, 0)]
    df_dist_9, df_cluster_9, df_proyectil_9 = erosion_9.extract_main_data(df_9, casos_9)

    latex_comparativa = tabla_comparativa_latex(
        df_dist_9, df_proyectil_9, df_9,
        df_dist, df_proyectil, df,
    )
    with open("datos/tabla_comparativa.tex", "w") as f:
        f.write(latex_comparativa)
    print("\n% --- TABLA COMPARATIVA (9 vs 245 repeticiones) ---")
    print(latex_comparativa)

    dfs_9 = erosion_9._prepare_cluster_data(casos_9, df_cluster_9, df_9)
    dfs_49 = _prepare_cluster_data(casos, df_cluster, df)
    plot_cluster_pdf_comparativo(dfs_9, dfs_49, v_target=50, ff_target=0.15)


if __name__ == "__main__":
    main()
    print("exito")
