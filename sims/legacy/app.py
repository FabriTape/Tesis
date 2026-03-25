from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Dashboard CARBOXI", layout="wide")

# Ruta donde CARBOXI guarda los parquet
RESULTS_DIR = Path("resultados")


@st.cache_data
def load_data():
    """Lee y pre-procesa todos los archivos parquet en memoria de forma optimizada."""
    if not RESULTS_DIR.exists():
        return None

    files = list(RESULTS_DIR.glob("*_TOTAL_*.parquet"))
    if not files:
        return None

    dataframes = []
    for f in files:
        try:
            # Extraemos el ID base del nombre del archivo
            sim_id = f.name.split("_TOTAL_")[0]

            # Usamos LazyFrames para procesar solo lo que necesitamos
            lf = pl.scan_parquet(f)

            # Agrupamos por paso 'i' y los metadatos
            # Asumimos que las partículas eyectadas de la cama tienen proyectil_erosion == null o False
            # Ajustá esta lógica si proyectil_erosion usa otra convención en tu pipeline
            agg_lf = lf.group_by(["i", "ff", "np", "vel0", "angulo"]).agg(
                pl.col("proyectil_erosion").is_null().sum().alias("cantidad_eyectados")
            )

            # Materializamos este pequeño resumen
            df_resumen = agg_lf.collect()

            # Añadimos el ID de simulación para agrupar las curvas en el gráfico
            df_resumen = df_resumen.with_columns(pl.lit(sim_id).alias("sim_id"))

            dataframes.append(df_resumen)

        except Exception as e:
            st.warning(f"Error procesando {f.name}: {e}")

    if not dataframes:
        return None

    return pl.concat(dataframes, how="vertical_relaxed")


# --- INTERFAZ ---
st.title("CARBOXI - Análisis Topológico de Erosión")

df = load_data()

if df is None:
    st.error(
        f"No se encontraron datos. Verificá que existan archivos *_TOTAL_*.parquet en '{RESULTS_DIR.absolute()}'"
    )
    st.stop()

# --- BARRA LATERAL (Filtros) ---
st.sidebar.header("Filtros de Simulación")

# Extraemos valores únicos para los filtros (convertimos a pandas/listas para los widgets)
opciones_ff = df["ff"].unique().to_list()
opciones_np = df["np"].unique().to_list()
opciones_vel = df["vel0"].unique().to_list()
opciones_ang = df["angulo"].unique().to_list()

selected_ff = st.sidebar.multiselect(
    "Fracción de fricción (ff)", opciones_ff, default=opciones_ff
)
selected_np = st.sidebar.multiselect(
    "Número de proyectiles (np)", opciones_np, default=opciones_np
)
selected_vel = st.sidebar.multiselect(
    "Velocidad inicial (vel0)", opciones_vel, default=opciones_vel
)
selected_ang = st.sidebar.multiselect(
    "Ángulo de impacto", opciones_ang, default=opciones_ang
)

# --- FILTRADO ---
df_filtered = df.filter(
    pl.col("ff").is_in(selected_ff)
    & pl.col("np").is_in(selected_np)
    & pl.col("vel0").is_in(selected_vel)
    & pl.col("angulo").is_in(selected_ang)
)

# --- GRÁFICO PRINCIPAL ---
if df_filtered.height == 0:
    st.info("No hay simulaciones que coincidan con los filtros actuales.")
else:
    # Ordenamos cronológicamente para que las líneas se tracen bien
    df_plot = df_filtered.sort(["sim_id", "i"])

    # Plotly a veces es más amigable con Pandas para mapear colores automáticos
    fig = px.line(
        df_plot.to_pandas(),
        x="i",
        y="cantidad_eyectados",
        color="sim_id",
        markers=True,
        hover_data=["ff", "np", "vel0", "angulo"],
        title="Evolución de partículas eyectadas por frame",
        labels={
            "i": "Frame de volcado (LAMMPS)",
            "cantidad_eyectados": "Nº Partículas Eyectadas",
            "sim_id": "Simulación",
        },
    )

    # Ajustes estéticos para que ocupe bien la pantalla
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)
