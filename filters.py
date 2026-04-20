"""
Filtros cinemáticos para detección de erosión.

Física: Los filtros aplican criterios cinéticos y espaciales para identificar
partículas eyectadas de la cama de sílice durante el impacto.
"""

from __future__ import annotations

import polars as pl


def filter_exclude_erosion(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra exclusión de erosión: partículas tipo 2 con z < threshold o vz < 0.

    Args:
        df: DataFrame con columnas: type, z, vz.

    Returns:
        DataFrame sin partículas erosionadas.
    """
    return df.filter(~((pl.col("type") == 2) & ((pl.col("z") < 1e-4) | (pl.col("vz") < 0))))


def filter_penetration(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra partículas penetradas: tipo 2 con z < threshold o vz < 0.

    Args:
        df: DataFrame con columnas: type, z, vz.

    Returns:
        DataFrame con solo partículas penetradas.
    """
    return df.filter((pl.col("type") == 2) & ((pl.col("z") < 1e-4) | (pl.col("vz") < 0)))


def filter_is_erosion(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra partículas erosionadas: type 2 con z > threshold.

    Args:
        df: DataFrame con columnas: type, z.

    Returns:
        DataFrame con solo partículas erosionadas.
    """
    return df.filter((pl.col("type") == 2) & (pl.col("z") > 1e-4))


def filter_fast_particles(
    lf: pl.LazyFrame,
    min_energy: float,
    min_vz: float = 0.0,
) -> pl.LazyFrame:
    """Filtra partículas con energía cinética superior al umbral.

    Args:
        lf: LazyFrame con columnas: vx, vy, vz.
        min_energy: Energía cinética mínima [m²/s²].
        min_vz: Velocidad vertical mínima [m/s].

    Returns:
        LazyFrame filtrado.
    """
    return lf.with_columns(
        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v2")
    ).filter(
        (pl.col("v2") >= min_energy) & (pl.col("vz") >= min_vz)
    ).drop("v2")


def filter_by_height(
    lf: pl.LazyFrame,
    min_height: float,
    max_height: float | None = None,
    exclude_type: int | None = None,
) -> pl.LazyFrame:
    """Filtra partículas por rango de altura.

    Args:
        lf: LazyFrame con columna 'z'.
        min_height: Altura mínima [m].
        max_height: Altura máxima [m]. Si None, sin límite superior.
        exclude_type: Excluir partículas de este tipo del filtro de altura.

    Returns:
        LazyFrame filtrado.
    """
    mask = pl.col("z") > min_height
    if max_height is not None:
        mask = mask & (pl.col("z") < max_height)

    if exclude_type is not None:
        return lf.filter(mask | (pl.col("type") == exclude_type))

    return lf.filter(mask)


def filter_by_box_bounds(
    lf: pl.LazyFrame,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float] | None = None,
    exclude_type: int | None = None,
) -> pl.LazyFrame:
    """Filtra partículas dentro de los límites de la caja.

    Args:
        lf: LazyFrame con columnas: x, y, z.
        x_bounds: Tuple (min, max) para X [m].
        y_bounds: Tuple (min, max) para Y [m].
        z_bounds: Tuple (min, max) para Z [m]. Si None, Sin filtro en Z.
        exclude_type: Excluir partículas de este tipo del filtro espacial.

    Returns:
        LazyFrame filtrado.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    mask = (pl.col("x") >= x_min) & (pl.col("x") <= x_max) & (
        pl.col("y") >= y_min
    ) & (pl.col("y") <= y_max)

    if z_bounds is not None:
        z_min, z_max = z_bounds
        mask = mask & (pl.col("z") >= z_min) & (pl.col("z") <= z_max)

    if exclude_type is not None:
        return lf.filter(mask | (pl.col("type") == exclude_type))

    return lf.filter(mask)