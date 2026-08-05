import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# =============================================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS (Para probar el script)
# =============================================================================
np.random.seed(42)
n_muestras = 500

# Tu intervalo (0, b]
x_data = np.random.uniform(0.1, 10.0, size=n_muestras)

# Parámetros reales ocultos que queremos recuperar
m_true = 0.6
c_true = -3.0
k_true = 0.118

# Proceso físico original V ~ N(f(x), sigma(x))
mu_real = m_true * x_data + c_true
sigma_real = np.sqrt(k_true * x_data)
v_escondido = np.random.normal(mu_real, sigma_real)

# Transformación de censura: V' = max(0, V)
y_data = np.maximum(0, v_escondido)


# =============================================================================
# 2. DEFINICIÓN DEL MODELO BAYESIANO EN PyMC
# =============================================================================
with pm.Model() as modelo_tobit:
    
    # --- Priors (Distribuciones a priori para tus parámetros) ---
    # Usamos priors normales amplios para la recta
    m = pm.Normal("m", mu=0, sigma=10)
    c = pm.Normal("c", mu=0, sigma=10)
    
    # Para 'k', como debe ser estrictamente positivo, usamos una HalfNormal
    k = pm.HalfNormal("k", sigma=1.0)
    
    # --- Relaciones Deterministas (Tus ecuaciones físicas) ---
    mu = pm.Deterministic("mu", m * x_data + c)
    
    # Añadimos un eps muy chico (1e-6) solo para evitar problemas numéricos si x -> 0
    sigma = pm.Deterministic("sigma", pm.math.sqrt(k * x_data + 1e-6))
    
    # --- Definición de la Verosimilitud Censurada ---
    # 1. Definimos la distribución normal latente (subyacente)
    dist_normal = pm.Normal.dist(mu=mu, sigma=sigma)
    
    # 2. Aplicamos la censura por la izquierda en 0 (Equivale a la Ec. del PDF)
    y_obs = pm.Censored("y_obs", 
                        dist=dist_normal, 
                        lower=0, 
                        upper=None, 
                        observed=y_data)
    
    # --- Muestreo MCMC (Hamiltonian Monte Carlo / NUTS) ---
    print("Iniciando el muestreo bayesiano...")
    trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True, target_accept=0.95)
    
    # --- Posterior Predictive Check (PPC) para Validación ---
    print("Generando predicciones de la posterior para validación...")
    ppc = pm.sample_posterior_predictive(trace)

# =============================================================================
# 3. ANÁLISIS DE RESULTADOS Y DIAGNÓSTICOS
# =============================================================================
# Resumen estadístico de los parámetros inferidos
print("\n--- Resumen de la Posterior ---")
resumen = az.summary(trace, var_names=["m", "c", "k"])
print(resumen)

# Graficar las cadenas para verificar convergencia (deben parecer "orugas")
az.plot_trace(trace, var_names=["m", "c", "k"])
plt.show()

# =============================================================================
# 4. EXTRACCIÓN DE PARÁMETROS Y GRAFICACIÓN DE LAS CURVAS TEÓRICAS
# =============================================================================
# Tomamos la media de la posterior de cada parámetro como nuestro valor óptimo
m_est = resumen.loc["m", "mean"]
c_est = resumen.loc["c", "mean"]
k_est = resumen.loc["k", "mean"]

# Reconstrucción de las curvas g(x) y s(x) usando las funciones de error
from scipy.special import erf
from scipy.stats import norm

x_plot = np.linspace(0.01, 10.0, 200)
f_x = float(m_est) * x_plot +float(c_est)
sigma_x = np.sqrt(float(k_est) * x_plot)
z = f_x / sigma_x

# Componentes analíticas del PDF
Phi_z = 0.5 * (1 + erf(z / np.sqrt(2)))
phi_z = norm.pdf(z)

# Curva media esperada g(x)
g_x = f_x * Phi_z + sigma_x * phi_z

# Desviación estándar efectiva para las bandas s+ y s-
segundo_momento = (f_x**2 + sigma_x**2) * Phi_z + f_x * sigma_x * phi_z
varianza_efectiva = segundo_momento - g_x**2
std_efectiva = np.sqrt(np.maximum(0, varianza_efectiva)) # evitar flotantes negativos mínimos

s_positiva = g_x + std_efectiva
s_negativa = np.maximum(0, g_x - std_efectiva)

# Gráfica Final
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.3, label="Datos observados (Censurados)", color="gray")
plt.plot(x_plot, f_x, '--', label="f(x) lineal subyacente estimada", color="green")
plt.plot(x_plot, g_x, '-', label="g(x) curva promedio teórica", color="orange", linewidth=2)
plt.fill_between(x_plot, s_negativa, s_positiva, color="orange", alpha=0.2, label="Banda de dispersión efectiva (s±)")
plt.xlabel("Variable X")
plt.ylabel("Variable Y (Mediciones)")
plt.title("Ajuste Tobit Bayesiano con Varianza Heterocedástica")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

az.plot_ppc_dist(ppc)
plt.title("Chequeo Predictivo Posterior (PPC)")
plt.show()
y_pred = ppc.posterior_predictive["y_obs"].values.flatten()
frac_real = np.mean(y_data == 0)
frac_pred = np.mean(y_pred == 0)

print(frac_real, frac_pred)
