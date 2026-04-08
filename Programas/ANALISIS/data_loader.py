"""Módulo de carga de datos.

Maneja la lectura del CSV de resultados de simulaciones LAMMPS.
"""

import pandas as pd

from constants import CSV_PATH


def load_data() -> pd.DataFrame:
    """Carga el CSV de resultados de simulaciones LAMMPS.

    Returns:
        DataFrame con los datos de merge_resultados.csv.
    """
    return pd.read_csv(CSV_PATH)


def filter_by_ff(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Filtra datos por factor de fricción.

    Args:
        df: DataFrame de datos.
        ff: Factor de fricción.

    Returns:
        DataFrame filtrado.
    """
    return df[df["ff"] == ff]


def filter_by_vel0(df: pd.DataFrame, vel0: float) -> pd.DataFrame:
    """Filtra datos por velocidad inicial.

    Args:
        df: DataFrame de datos.
        vel0: Velocidad inicial.

    Returns:
        DataFrame filtrado.
    """
    return df[df["vel0"] == vel0]


def filter_by_i_value(df: pd.DataFrame, i_value: int) -> pd.DataFrame:
    """Filtra datos por paso temporal.

    Args:
        df: DataFrame de datos.
        i_value: Paso temporal.

    Returns:
        DataFrame filtrado.
    """
    return df[df["i"] == i_value]


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
    """Filtra partículas erosionadas usando columna proyectil_erosion.

    Args:
        df: DataFrame de datos.

    Returns:
        DataFrame con solo partículas erosionadas.
    """
    return df[df["proyectil_erosion"].eq(True)]