"""Módulo de visualizaciones cinemáticas.

Genera gráficos de ángulo de velocidad, energía cinética y penetración.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from kinematics_calc import (
    compute_kinetic_energy,
    compute_velocity_angle,
    compute_penetration_depth,
)
from constants import I_VALUE, COLORS


def plot_velocity_angle(df: pd.DataFrame, ff: float, vel0: float) -> Figure:
    """Genera gráfico de distribución del ángulo de velocidad del CM.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Figura matplotlib.
    """
    data = compute_velocity_angle(df, ff, vel0)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
    labels = [f"{int(b)}° a {int(b) + 10}°" for b in data["bin"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, data["frecuencia"], color=COLORS["primary"])

    ax.set_xlabel("Ángulo (°)", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(f"Ángulo velocidad CM (ff={ff}, vel0={vel0}, i={I_VALUE})", fontsize=fs * 0.8)
    plt.xticks(rotation=45, ha="right", fontsize=fs * 0.7)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_kinetic_energy(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de K/m promedio del CM de clusters salientes.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = compute_kinetic_energy(df, ff)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = data.set_index("vel0").loc[unique_vel0, "avg"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(unique_vel0)), avg_values, color=COLORS["primary"])

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel(r"$K$/$m$ cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_penetration(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de profundidad de penetración vs vel0.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = compute_penetration_depth(df, ff)
    if data.empty:
        return plt.figure()

    fs = 1.8 * 12
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = [data[data["vel0"] == v]["avg"].values[0] for v in unique_vel0]

    ax.bar(range(len(unique_vel0)), avg_values, color=COLORS["primary"])

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel("Profundidad de penetración (µm)", fontsize=fs)
    ax.set_title(f"Penetración (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig