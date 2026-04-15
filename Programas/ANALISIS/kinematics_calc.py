"""Módulo de cálculos cinemáticos para visualizaciones de erosión.

Contiene funciones para el grid de posiciones y heatmap.
"""

import numpy as np
import pandas as pd


def get_grid_positions(df: pd.DataFrame) -> list[float]:
    """Obtiene las posiciones únicas de x y y del dataframe completo.

    Args:
        df: DataFrame con todos los datos.

    Returns:
        Lista de posiciones únicas de x y y.
    """
    return list(set(df["posx"].unique()))


def get_grid_pairs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Obtiene los pares únicos de posx/posy del dataframe completo.

    Args:
        df: DataFrame con todos los datos.

    Returns:
        Lista de tuplas (posx, posy) únicas.
    """
    x = df["posx"].unique()
    y = df["posy"].unique()
    pos = []
    not_found = []
    for x_val in x:
        for y_val in y:
            if df[(df["posx"] == x_val) & (df["posy"] == y_val)].empty:
                not_found.append((x_val, y_val))
                continue
            else:
                pos.append((x_val, y_val))
    return pos, not_found


def compute_heatmap_data(
    df: pd.DataFrame,
    ff: float,
    vel0: float,
) -> pd.DataFrame:
    """Calcula partículas erosionadas por posición usando el último time step.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame pivot con IDs por posición, índices y columnas normalizados por 1.52.
    """
    df_filt = df[(df["ff"] == ff) & (df["vel0"] == vel0)]
    max_i = df_filt["i"].max()
    df_filt = df_filt[df_filt["i"] == max_i]

    df_validas = df_filt[
        (df_filt["type"] == 1) | ((df_filt["type"] == 2) & (df_filt["z"] > 1e-4))
    ]

    use_pairs, not_found = get_grid_pairs(df_validas)
    grid_positions = get_grid_positions(df_validas)

    grouped = df_validas.groupby(["posx", "posy"]).size().reset_index(name="id")
    grouped = grouped.set_index(["posx", "posy"])["id"]

    result = {}
    for posx, posy in use_pairs:
        result[(posx, posy)] = grouped.get((posx, posy), 0)

    result.update({k: np.nan for k in not_found})

    pivot = pd.DataFrame(
        result.values(),
        index=pd.MultiIndex.from_tuples(result.keys(), names=["posy", "posx"]),
    ).unstack(level="posx")

    pivot = pivot.droplevel(0, axis=1)
    pivot = pivot.reindex(index=grid_positions, columns=grid_positions)

    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot
