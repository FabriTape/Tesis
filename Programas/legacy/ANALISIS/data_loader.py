"""Módulo de carga de datos.

Maneja la lectura del CSV de resultados de simulaciones LAMMPS.
"""

import pandas as pd


def filter_exclude_erosion(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra exclusión de erosión: partículas tipo 2 con z < threshold o vz < 0.

    Args:
        df: DataFrame de datos.

    Returns:
        DataFrame sin partículas erosionadas.
    """
    return df[~((df["type"] == 2) & ((df["z"] < 1e-4) | (df["vz"] < 0)))]


def filter_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra partículas penetradas: tipo 2 con z < threshold o vz < 0.

    Args:
        df: DataFrame de datos.

    Returns:
        DataFrame con solo partículas penetradas.
    """
    return df[(df["type"] == 2) & ((df["z"] < 1e-4) | (df["vz"] < 0))]


def filter_is_erosion(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra partículas erosionadas: type 2 con z > threshold.

    Args:
        df: DataFrame de datos.

    Returns:
        DataFrame con solo partículas erosionadas.
    """
    return df[(df["type"] == 2) & (df["z"] > 1e-4)]
