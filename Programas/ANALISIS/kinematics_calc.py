import pandas as pd


def get_grid_pairs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Obtiene los pares únicos de posx/posy del dataframe completo."""
    # Extrae pares únicos de forma instantánea usando un set
    return list(set(zip(df["posx"], df["posy"])))


def compute_heatmap_data(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula partículas erosionadas por posición usando el último time step."""
    df_filt = df[(df["ff"] == ff) & (df["vel0"] == vel0)]
    if df_filt.empty:
        return pd.DataFrame()

    max_i = df_filt["i"].max()
    df_validas = df_filt[
        (df_filt["i"] == max_i)
        & ((df_filt["type"] == 1) | ((df_filt["type"] == 2) & (df_filt["z"] > 1e-4)))
    ]

    # Agrupa por coordenadas, cuenta el tamaño y unstack crea el heatmap 2D automáticamente
    pivot = df_validas.groupby(["posy", "posx"]).size().unstack()

    # Normaliza índices y columnas
    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot
