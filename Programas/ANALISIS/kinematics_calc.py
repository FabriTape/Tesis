import itertools

import pandas as pd


def get_grid_pairs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Obtiene la grilla global (producto cartesiano de todos los posx y posy únicos)."""
    x_vals = df["posx"].unique()
    y_vals = df["posy"].unique()

    # Producto cartesiano para asegurar una grilla completa global
    return list(itertools.product(x_vals, y_vals))


def get_global_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Obtiene DataFrame con la grilla global union de todas las posiciones.

    Returns:
        DataFrame con columnas [posx, posy] de todas las posiciones únicas.
    """
    return df[["posx", "posy"]].drop_duplicates().reset_index(drop=True)


def get_max_data_count(df: pd.DataFrame, ff: float) -> int:
    """Calcula el máximo de datos reales entre todas las simulaciones para un ff dado.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Máximo conteo de datos reales por simulación.
    """
    df_filt = df[(df["ff"] == ff) & (df["i"] == df["i"].max())]
    df_filt = df_filt[
        ((df_filt["type"] == 1) | ((df_filt["type"] == 2) & (df_filt["z"] > 1e-4)))
    ]
    counts = df_filt.groupby(["vel0", "angulo", "posx", "posy"]).size()
    return counts.max() if len(counts) > 0 else 0


def get_sim_positions(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Obtiene las posiciones únicas de una simulación específica.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con columnas [posx, posy] únicas de la sim.
    """
    return (
        df[(df["ff"] == ff)][["posx", "posy"]].drop_duplicates().reset_index(drop=True)
    )


def get_max_positions(df: pd.DataFrame, ff: float) -> int:
    """Calcula el máximo de posiciones únicas entre todas las simulaciones para un ff.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Máximo número de posiciones únicas.
    """
    max_pos = 0
    for vel0 in df[df["ff"] == ff]["vel0"].unique():
        positions = get_sim_positions(df, ff, vel0)
        max_pos = max(max_pos, len(positions))
    return max_pos


def get_global_positions(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Obtiene las posiciones globales para un ff específico.

    Une las posiciones de todas las simulaciones de ese ff.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con columnas [posx, posy] únicas para ese ff.
    """
    all_positions = []
    for vel0 in df[df["ff"] == ff]["vel0"].unique():
        sim_pos = get_sim_positions(df, ff, vel0)
        all_positions.append(sim_pos)
    result = pd.concat(all_positions, ignore_index=True).drop_duplicates()
    return result.reset_index(drop=True)


def compute_heatmap_data(
    df: pd.DataFrame,
    ff: float,
    vel0: float,
) -> pd.DataFrame:
    """Calcula partículas erosionadas por posición usando la grilla global."""

    # 1. Obtener los ejes globales de toda la simulación (antes de filtrar)
    global_x = sorted(df["posx"].unique())
    global_y = sorted(df["posy"].unique())

    # 2. Filtrar por la simulación actual
    df_filt = df[(df["ff"] == ff) & (df["vel0"] == vel0)]

    # Si no hay datos, devolvemos la grilla global llena de ceros directamente
    if df_filt.empty:
        pivot = pd.DataFrame(0, index=global_y, columns=global_x)
    else:
        max_i = df_filt["i"].max()
        df_validas = df_filt[
            (df_filt["i"] == max_i)
            & (
                (df_filt["type"] == 1)
                | ((df_filt["type"] == 2) & (df_filt["z"] > 1e-4))
            )
        ]

        # Agrupar y crear la matriz 2D (pone NaN donde falta intersección interna)
        pivot = df_validas.groupby(["posy", "posx"]).size().unstack(fill_value=0)

    # 3. EL PASO CLAVE: Forzar la grilla global.
    # Esto agrega las filas (posy) y columnas (posx) que faltan y les pone 0
    pivot = pivot.reindex(index=global_y, columns=global_x, fill_value=0)

    # 4. Normalizar índices y columnas
    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot
