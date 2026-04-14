"""Módulo de cálculos cinemáticos para visualizaciones de erosión.

Contiene funciones para el grid de posiciones y heatmap.
"""

import pandas as pd
import numpy as np

from data_loader import (
    filter_by_ff,
    filter_by_vel0,
    filter_by_i_value,
)


GRID_POSITIONS = [-1.52, 0.0, 1.52, 3.04]

GRID_PAIRS = [(x, y) for x in GRID_POSITIONS for y in GRID_POSITIONS]


def get_grid_pairs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Obtiene los pares únicos de posx/posy del dataframe completo.

    Args:
        df: DataFrame con todos los datos.

    Returns:
        Lista de tuplas (posx, posy) únicas.
    """
    return list(set(zip(df["posx"], df["posy"])))


def compute_heatmap_data(
    df: pd.DataFrame, ff: float, vel0: float, grid_pairs: list[tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Calcula partículas erosionadas por posición usando el último time step.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.
        grid_pairs: Lista de pares (posx, posy) válidos. Si es None, usa GRID_PAIRS.

    Returns:
        DataFrame pivot con IDs por posición, índices y columnas normalizados por 1.52.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    max_i = df_filt["i"].max()
    df_filt = filter_by_i_value(df_filt, max_i)

    df_validas = df_filt[(df_filt["type"] == 1) | 
                         ((df_filt["type"] == 2) & (df_filt["z"] > 1e-4))]

    use_pairs = grid_pairs if grid_pairs is not None else GRID_PAIRS

    grouped = df_validas.groupby(["posx", "posy"]).size().reset_index(name="id")
    grouped = grouped.set_index(["posx", "posy"])["id"]

    result = {}
    for posx, posy in use_pairs:
        result[(posx, posy)] = grouped.get((posx, posy), 0)

    result.update({k: np.nan for k in GRID_PAIRS if k not in use_pairs})

    pivot = pd.DataFrame(
        result.values(),
        index=pd.MultiIndex.from_tuples(result.keys(), names=["posy", "posx"])
    ).unstack(level="posx")

    pivot = pivot.droplevel(0, axis=1)
    pivot = pivot.reindex(index=GRID_POSITIONS, columns=GRID_POSITIONS)

    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot