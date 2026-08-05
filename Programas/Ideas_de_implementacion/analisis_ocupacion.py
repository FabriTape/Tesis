#!/usr/bin/env python3
"""
Calcula el porcentaje del área XY (100x100 um²) ocupada por los granos cuyo
centro cumple z > z_threshold.

El cálculo se realiza proyectando cada partícula como un disco sobre el plano
XY y rasterizando sobre una grilla booleana.

Autor: ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Configuración
# ======================================================================

archivo = "box_fill_0.35_SI_100um.data"

# Caja (metros)
Lx = 100e-6
Ly = 100e-6

# Resolución de la grilla
NPIX = 1200

# Paso en altura (µm)
dz = 0.1

# ======================================================================
# Lectura del archivo
# ======================================================================

print("Leyendo archivo...")

x = []
y = []
z = []
r = []

with open(archivo) as f:

    atoms = False

    for line in f:

        if line.strip() == "Atoms":
            atoms = True
            next(f)   # línea en blanco
            continue

        if not atoms:
            continue

        if not line.strip():
            continue

        cols = line.split()

        if len(cols) < 7:
            continue

        diam = float(cols[2])

        x.append(float(cols[4]))
        y.append(float(cols[5]))
        z.append(float(cols[6]))
        r.append(diam/2)

x = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)
r = np.asarray(r)

print(f"{len(x)} partículas leídas")

# ======================================================================
# Ordenar por altura
# ======================================================================

orden = np.argsort(-z)

x = x[orden]
y = y[orden]
z = z[orden]
r = r[orden]

# ======================================================================
# Grilla
# ======================================================================

dx = Lx / NPIX

mask = np.zeros((NPIX, NPIX), dtype=bool)

pix_ocupados = 0

# Alturas
z_levels = np.arange(100.0, -dz, -dz) * 1e-6

coverage = []

indice = 0

print("Calculando cobertura...")

for zlim in z_levels:

    while indice < len(z) and z[indice] > zlim:

        xc = x[indice]
        yc = y[indice]
        rc = r[indice]

        ix0 = max(0, int((xc - rc) / dx))
        ix1 = min(NPIX - 1, int((xc + rc) / dx))

        iy0 = max(0, int((yc - rc) / dx))
        iy1 = min(NPIX - 1, int((yc + rc) / dx))

        xx = (np.arange(ix0, ix1 + 1) + 0.5) * dx
        yy = (np.arange(iy0, iy1 + 1) + 0.5) * dx

        XX, YY = np.meshgrid(xx, yy)

        disco = (XX - xc) ** 2 + (YY - yc) ** 2 <= rc ** 2

        sub = mask[iy0:iy1 + 1, ix0:ix1 + 1]

        nuevos = disco & (~sub)

        pix_ocupados += np.count_nonzero(nuevos)

        sub |= disco

        mask[iy0:iy1 + 1, ix0:ix1 + 1] = sub

        indice += 1

    coverage.append(100 * pix_ocupados / mask.size)

coverage = np.asarray(coverage)

# ======================================================================
# Guardar CSV
# ======================================================================

np.savetxt(
    "coverage_vs_height.csv",
    np.column_stack((z_levels * 1e6, coverage)),
    delimiter=",",
    header="z_um,coverage_percent",
    comments=""
)

print("CSV guardado.")
# ======================================================================
# Derivada
# ======================================================================
from scipy.signal import savgol_filter
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ======================================================================
# Variables del sistema
# ======================================================================
# Profundidad z medida desde la cresta más alta hacia abajo (0 a 100 µm)
z_profundidad = 100.0 - (z_levels * 1e6)
y_data = coverage / 100.0  # Convertir porcentaje a fracción (0 a 1)

# ======================================================================
# 1. Ajuste del Modelo Exponencial: 1 - exp(-lambda * z)
# ======================================================================
def modelo_exp(z, lamb):
    return 1 - np.exp(-lamb * z)

# Ejecutar el ajuste no lineal (estimación inicial p0 = 0.5 µm^-1)
popt, pcov = curve_fit(modelo_exp, z_profundidad, y_data, p0=[0.35],bounds=(0.1,0.36))
lambda_opt = popt[0]
error_lambda = np.sqrt(np.diag(pcov))[0]

# Cálculo del R^2 (coeficiente de determinación) para evaluar el ajuste
residuos = y_data - modelo_exp(z_profundidad, lambda_opt)
ss_res = np.sum(residuos**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print(" RESULTADOS DEL AJUSTE")
print("="*50)
print(f"Parámetro lambda       : {lambda_opt:.4f} ± {error_lambda:.4f} µm⁻¹")
print(f"Profundidad media (1/λ)  : {1.0/lambda_opt:.4f} µm")
print(f"Calidad del ajuste (R²)  : {r2:.6f}")
print("="*50 + "\n")

# ======================================================================
# 2. Suavizado y Derivada (Con parámetros estables)
# ======================================================================
window_length = 51     
polyorder = 3          

y_smooth = savgol_filter(y_data, window_length=window_length, polyorder=polyorder)
# Al avanzar el índice, z_profundidad aumenta en +dz, por lo que la derivada es positiva
gradient_data = savgol_filter(y_data, window_length=window_length, polyorder=polyorder, deriv=1, delta=dz)

# Derivada analítica teórica del modelo: dA/dz = lambda * exp(-lambda * z)
gradient_teorico = lambda_opt * np.exp(-lambda_opt * z_profundidad)

# ======================================================================
# 3. Figura Comparativa
# ======================================================================
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Gráfico superior: Cobertura vs Profundidad
ax[0].plot(z_profundidad, y_data * 100, label="Datos crudos (Grilla)", alpha=0.4, lw=1.5, color="gray")
ax[0].plot(z_profundidad, y_smooth * 100, label="Suavizado Savitzky-Golay", lw=1.5, color="blue", alpha=0.7)
ax[0].plot(z_profundidad, modelo_exp(z_profundidad, lambda_opt) * 100, 
           label=f"Ajuste: $1 - e^{{-{lambda_opt:.3f} z}}$", color="red", lw=2, ls="--")
ax[0].set_ylabel("Área ocupada (%)")
ax[0].legend(loc="lower right")
ax[0].grid(alpha=0.3)

# Gráfico inferior: Derivada (Tasa de cambio del área)
ax[1].plot(z_profundidad, gradient_data, label="Derivada Savitzky-Golay", lw=2, color="blue")
ax[1].plot(z_profundidad, gradient_teorico, 
           label=r"Derivada Teórica: $\lambda e^{-\lambda z}$", color="red", lw=2, ls="--")
ax[1].set_xlabel("Profundidad desde la cresta $z$ (µm)")
ax[1].set_ylabel(r"$dA/dz$ ($\mu$m$^{-1}$)")
ax[1].legend(loc="upper right")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()