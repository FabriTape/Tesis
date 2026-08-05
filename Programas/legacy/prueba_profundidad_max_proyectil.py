"""Prueba: profundidad máxima de los cráteres de proyectil por celda.

A partir del archivo de resultados de 49 movimientos, selecciona las
partículas de proyectil (type == 2) que quedaron incrustadas al final de la
simulación y calcula la profundidad máxima del cráter para cada combinación
de posición de impacto (posx, posy) y orientación de impacto (rot_*).

Salida por consola:
    - Resumen general (cantidad de partículas, rango de z y de depth).
    - Tabla de profundidad máxima por celda (posx, posy, rotación).
    - Estadísticas globales sobre las celdas.
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path("datos/resultados_49_mov/merge_resultados.csv")

df = pd.read_csv(CSV_PATH)

# Profundidad de los cráteres de proyectil: partículas tipo 2 en el último
# paso de tiempo (i == 80) que no se erosionaron por completo
# (proyectil_erosion != 1) y quedaron bajo la superficie (z < 100e-6).
# La profundidad se mide respecto de la superficie de referencia en
# z = 100e-6 m (= 100 µm): depth = 100 - z*1e6, en µm.
mask = (
    (df["proyectil_erosion"] != 1)
    & (df["i"] == 80)
    & (df["z"] < 100e-6)
)
proj = df.loc[mask, ["posx", "posy", "z", "rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]].copy()
proj["depth"] = 100 - proj["z"] * 1e6

print(f"{'Partículas:':>12} {len(proj)}")
print(f"{'Rango z:':>12} {proj['z'].min()*1e6:.2f} – {proj['z'].max()*1e6:.2f} µm")
print(f"{'Rango depth:':>12} {proj['depth'].min():.2f} – {proj['depth'].max():.2f} µm")
print()

# Profundidad máxima por celda: el cráter de una celda queda definido por la
# partícula de proyectil que alcanzó la mayor profundidad en esa celda.
rot_cols = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]
max_cell = proj.groupby(["posx", "posy", *rot_cols])["depth"].max()
print("=== Profundidad máxima por celda (posx, posy, rot) ===")
print(f"{'posx':>6} {'posy':>6} {'rot_ang':>7} {'rot_ex':>6} {'rot_ey':>6} {'rot_ez':>6} {'depth_max (µm)':>14}")
print("-" * 60)
for (px, py, ra, rx, ry, rz), val in max_cell.sort_index().items():
    print(f"{px:>6.2f} {py:>6.2f} {ra:>7.0f} {rx:>6.0f} {ry:>6.0f} {rz:>6.0f} {val:>14.2f}")

print()
# Estadísticas globales sobre las celdas de la grilla
print(f"{'Máximo global:':>20} {max_cell.max():.2f} µm")
print(f"{'Media celdas:':>20} {max_cell.mean():.2f} µm")
print(f"{'Mínimo celda:':>20} {max_cell.min():.2f} µm")
print(f"{'Std celdas:':>20} {max_cell.std():.2f} µm")
print(f"{'Celdas totales:':>20} {len(max_cell)}")
