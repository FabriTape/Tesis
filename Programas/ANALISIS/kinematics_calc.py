import itertools

import pandas as pd
import numpy as np

def get_grid_pairs(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Obtiene la grilla global (producto cartesiano de todos los posx y posy únicos)."""
    x_vals = df["posx"].unique()
    y_vals = df["posy"].unique()

    # Producto cartesiano para asegurar una grilla completa global
    return list(itertools.product(x_vals, y_vals))


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
        pivot = df_validas.groupby(["posy", "posx"]).size().unstack(fill_value=np.nan)

    # 3. EL PASO CLAVE: Forzar la grilla global.
    # Esto agrega las filas (posy) y columnas (posx) que faltan y les pone 0
    pivot = pivot.reindex(index=global_y, columns=global_x, fill_value=0)

    # 4. Normalizar índices y columnas
    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot
