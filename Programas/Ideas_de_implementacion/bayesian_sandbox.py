from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import erf
from scipy.stats import ks_2samp, norm


# =============================================================================
# 1. EXTRACCIÓN DE DATOS
# =============================================================================
ROT_COLS = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]


def get_raw_data(df, n, alfa, ff):
    print(rf"Extrayendo datos puros para np={n}, angulo={alfa}, ff={ff}...")
    data = df[(df["np"] == n)].copy()

    if data.empty:
        return np.array([]), np.array([])

    step = data[data["posx"] > 0]["posx"].min()
    data["grid_x"] = (data["posx"] / step).round().astype(int)
    data["grid_y"] = (data["posy"] / step).round().astype(int)

    x_min, y_min = int(data["grid_x"].min()), int(data["grid_y"].min())
    x_max, y_max = int(data["grid_x"].max()), int(data["grid_y"].max())

    reales = (
        data[["grid_x", "grid_y"]]
        .drop_duplicates()
        .assign(existe=True)
        .set_index(["grid_x", "grid_y"])
    )
    x_r, y_r = range(x_min, x_max + 1), range(y_min, y_max + 1)
    base_grid = (
        pd.MultiIndex.from_product([x_r, y_r], names=["grid_x", "grid_y"])
        .to_frame(index=False)
        .set_index(["grid_x", "grid_y"])
    )
    base_grid = base_grid.join(reales).fillna(False).infer_objects(copy=False)

    vels = sorted(
        data.abs().loc[(data["ff"] == ff) & (data["angulo"] == alfa), "vel0"].unique()
    )
    x_out, y_out = [], []

    for v in vels:
        mask_base = (
            (data["ff"] == ff)
            & (data["vel0"] == -1 * v)
            & (data["angulo"] == alfa)
        )
        rotaciones = data.loc[mask_base, ROT_COLS].drop_duplicates()
        for _, rot_row in rotaciones.iterrows():
            rot = rot_row.to_dict()
            mask = mask_base & (
                (data["rot_angulo"] == rot["rot_angulo"])
                & (data["rot_eje_x"] == rot["rot_eje_x"])
                & (data["rot_eje_y"] == rot["rot_eje_y"])
                & (data["rot_eje_z"] == rot["rot_eje_z"])
            )
            sim = data[mask]
            if sim.empty:
                continue

            i_max = sim["i"].max()
            erosion_mask = (sim["i"] == i_max) & (sim["type"] == 1)
            erosion_count = sim[erosion_mask].groupby(
                ["grid_x", "grid_y"]
            )["id"].nunique()
            yields = erosion_count.rename("yield")

            res = base_grid.join(yields)
            res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0
            valores = res.loc[res["existe"], "yield"].values

            x_out.extend([v] * len(valores))
            y_out.extend(valores)

    return np.array(x_out), np.array(y_out)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================
ff_test = 0.15
n_fisico = 1
v_validacion = 50

csv_9 = Path("datos/resultados_9_mov/merge_resultados.csv").expanduser()
csv_49 = Path("datos/resultados_49_mov/merge_resultados.csv").expanduser()


# =============================================================================
# 3. MODELO
# =============================================================================
def fit_tobit_powerlaw(x_data, y_data, n_test):
    x_max_prior = float(x_data.min()) * 0.95

    with pm.Model():
        a = pm.HalfNormal("a", sigma=10)
        c = pm.HalfNormal("c", sigma=5.0)
        d = pm.HalfNormal("d", sigma=2)
        x_0 = pm.Uniform("x_0", lower=0.0, upper=x_max_prior)

        x_shifted = x_data - x_0
        mu = pm.Deterministic("mu", a * x_shifted - n_test)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(c * x_shifted**d))

        dist_normal = pm.Normal.dist(mu=mu, sigma=sigma)
        pm.Censored("y_obs", dist=dist_normal, lower=0, upper=None, observed=y_data)

        print("\nIniciando muestreo MCMC...")
        trace = pm.sample(
            draws=2000, tune=1000, return_inferencedata=True, target_accept=0.95
        )

    return trace


def predict_at_v(trace, v, n_test):
    """Predice mu y sigma en una velocidad v usando el posterior."""
    posterior = trace.posterior
    a_s = posterior["a"].values.flatten()
    c_s = posterior["c"].values.flatten()
    d_s = posterior["d"].values.flatten()
    x0_s = posterior["x_0"].values.flatten()

    x_shifted = v - x0_s
    mu = a_s * x_shifted - n_test
    sigma = np.sqrt(c_s * x_shifted**d_s)
    return mu, sigma


# =============================================================================
# 4. EJECUCIÓN
# =============================================================================
# --- Ajustar modelo con datos de 9 mov (4 velocidades) ---
df_9 = pd.read_csv(csv_9)
x_9, y_9 = get_raw_data(df_9, 1, 0, ff_test)
print(f"\nDatos 9 mov: {len(x_9)} puntos, v={sorted(np.unique(x_9))}")

trace_9 = fit_tobit_powerlaw(x_9, y_9, n_fisico)
resumen_9 = az.summary(trace_9, var_names=["a", "c", "d", "x_0"])
print("\n--- Parámetros ajustados (9 mov) ---")
print(resumen_9)

# --- Extraer distribución empírica en v=50 del dataset 49 mov ---
df_49 = pd.read_csv(csv_49)
x_49, y_49 = get_raw_data(df_49, 3, 90, ff_test)
y_49_v50 = y_49[x_49 == v_validacion]
print(f"\nDatos 49 mov en v={v_validacion}: {len(y_49_v50)} puntos")
print(f"  Yield media: {y_49_v50.mean():.2f}, std: {y_49_v50.std():.2f}")
print(f"  Ceros: {(y_49_v50 == 0).sum()} ({(y_49_v50 == 0).mean()*100:.1f}%)")

# --- Predicción del modelo en v=50 ---
mu_post, sigma_post = predict_at_v(trace_9, v_validacion, n_fisico)

# Distribución predictiva censorada (promedio sobre el posterior)
n_samples = 5000
y_rep = np.zeros((len(mu_post), n_samples))
for i in range(len(mu_post)):
    y_rep[i] = np.maximum(0, np.random.normal(mu_post[i], sigma_post[i]))

y_rep_flat = y_rep.flatten()

# --- Figura de comparación ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Ajuste del modelo 9 mov
ax = axes[0]
df_temp = pd.DataFrame({"v": x_9, "y": y_9})
medias = df_temp.groupby("v")["y"].mean()
ax.plot(medias.index, medias.values, "ko", markersize=8, label="Medias observadas")

x_plot = np.linspace(0.01, max(x_9) + 5, 200)
a_est = float(resumen_9.loc["a", "mean"])
c_est = float(resumen_9.loc["c", "mean"])
d_est = float(resumen_9.loc["d", "mean"])
x0_est = float(resumen_9.loc["x_0", "mean"])
x_sh = x_plot - x0_est
f_x = a_est * x_sh - n_fisico
sigma_x = np.sqrt(c_est * x_sh**d_est)
z = f_x / sigma_x
Phi_z = 0.5 * (1 + erf(z / np.sqrt(2)))
phi_z = norm.pdf(z)
g_x = f_x * Phi_z + sigma_x * phi_z

ax.plot(x_plot, g_x, "-", color="C0", linewidth=2.5, label="Modelo Tobit $g(x)$")
ax.axvline(v_validacion, color="red", linestyle="--", alpha=0.5, label=f"v={v_validacion} (validación)")
ax.set_xlabel("Velocidad de impacto (v)")
ax.set_ylabel("Yield (Y)")
ax.set_title("Ajuste con datos 9 mov")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Comparación de distribuciones en v=50
ax = axes[1]
bins = np.arange(-0.5, y_49_v50.max() + 1.5, 1)
ax.hist(
    y_49_v50, bins=bins, density=True, alpha=0.6,
    color="C1", edgecolor="white", label=f"49 mov (n={len(y_49_v50)})",
)
ax.hist(
    y_rep_flat, bins=bins, density=True, alpha=0.5,
    color="C0", edgecolor="white", label=f"Predicción 9 mov (n={len(y_rep_flat)})",
)
ax.set_xlabel("Yield (Y)")
ax.set_ylabel("Densidad")
ax.set_title(f"Distribución en v={v_validacion}")
ax.legend()
ax.grid(True, alpha=0.3)

ks_stat, ks_pval = ks_2samp(y_49_v50, y_rep_flat)
ax.text(
    0.95, 0.95,
    f"KS = {ks_stat:.3f}\np = {ks_pval:.3f}",
    transform=ax.transAxes, ha="right", va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.tight_layout()
plt.savefig(
    Path(f"validacion_v{v_validacion}_np{n_fisico}_ff{ff_test}.png"),
    dpi=150, bbox_inches="tight",
)
plt.close()
print(f"\nFigura guardada: validacion_v{v_validacion}_np{n_fisico}_ff{ff_test}.png")
print(f"KS = {ks_stat:.3f}, p-valor = {ks_pval:.3f}")
