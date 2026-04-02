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
            # Separamos parámetros de la fecha usando _TOTAL_ como pivote
            partes = f.name.split("_TOTAL_")
            sim_id = partes[0]
            timestamp = partes[1].replace(".parquet", "")

            # Usamos LazyFrames para procesar solo lo que necesitamos
            lf = pl.scan_parquet(f)

            # Agrupamos por paso 'i' y los metadatos
            agg_lf = lf.group_by(["i", "ff", "np", "vel0", "angulo"]).agg(
                pl.col("proyectil_erosion").is_null().sum().alias("cantidad_eyectados")
            )

            # Materializamos el resumen
            df_resumen = agg_lf.collect()

            # Añadimos parámetros, fecha y un ID único para la curva
            df_resumen = df_resumen.with_columns(
                [
                    pl.lit(sim_id).alias("parametros"),
                    pl.lit(timestamp).alias("fecha"),
                    pl.lit(f"{sim_id}  📅 {timestamp}").alias("curva_id"),
                ]
            )

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

# Extraemos valores únicos para los filtros
opciones_ff = df["ff"].unique().to_list()
opciones_np = df["np"].unique().to_list()
opciones_vel = df["vel0"].unique().to_list()
opciones_ang = df["angulo"].unique().to_list()
opciones_fechas = df["fecha"].unique().to_list()

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

# Agregamos un filtro de fechas por si querés aislar una corrida específica
st.sidebar.divider()
selected_fechas = st.sidebar.multiselect(
    "Filtrar por Fecha de corrida", opciones_fechas, default=opciones_fechas
)

# --- FILTRADO ---
df_filtered = df.filter(
    pl.col("ff").is_in(selected_ff)
    & pl.col("np").is_in(selected_np)
    & pl.col("vel0").is_in(selected_vel)
    & pl.col("angulo").is_in(selected_ang)
    & pl.col("fecha").is_in(selected_fechas)
)

# --- GRÁFICO PRINCIPAL ---
if df_filtered.height == 0:
    st.info("No hay simulaciones que coincidan con los filtros actuales.")
else:
    # Ordenamos cronológicamente
    df_plot = df_filtered.sort(["curva_id", "i"])

    fig = px.line(
        df_plot.to_pandas(),
        x="i",
        y="cantidad_eyectados",
        color="curva_id",  # Distintas fechas = distintas curvas
        markers=True,
        hover_data=["parametros", "fecha"],
        title="Evolución de partículas eyectadas por frame",
        labels={
            "i": "Frame de volcado (LAMMPS)",
            "cantidad_eyectados": "Nº Partículas Eyectadas",
            "curva_id": "Parámetros y Fecha",
        },
    )

    # Configuramos la leyenda para que sea vertical a la derecha (scrolleable por defecto)
    fig.update_layout(
        height=750,  # Aumentamos un poco el alto para ver mejor los datos
        legend=dict(
            title="Simulaciones (Scroll para ver más)",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(
            r=300
        ),  # Damos margen derecho para que la leyenda no se pise con el gráfico
    )

    st.plotly_chart(fig, width="stretch")
