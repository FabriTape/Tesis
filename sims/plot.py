"""
Módulo de visualización para el análisis de simulaciones LAMMPS.

Proporciona funciones para generar gráficos de la evolución de partículas
eyectadas utilizando Matplotlib y Polars. Los gráficos incluyen metadatos
de configuración para garantizar la trazabilidad y reproducibilidad.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def plot_erosion(
    df: pl.DataFrame,
    ff: float,
    np_: float,
    vel: float,
    angle: float,
    move1: float,
    move2: float,
    move3: float,
    dir_resultados: Path | None = None,
    config: dict | None = None,
) -> None:
    """
    Genera y guarda un gráfico de la evolución de partículas eyectadas.

    Args:
        df (pl.DataFrame): DataFrame con los datos de la simulación, incluyendo
                           las columnas 'i' (iteración) y 'proyectil_erosion'.
        ff (float): Parámetro de fricción de la simulación.
        np_ (float): Número de partículas de la simulación.
        vel (float): Velocidad inicial del proyectil.
        angle (float): Ángulo de impacto.
        move1 (float): Posición inicial en X del proyectil.
        move2 (float): Posición inicial en Y del proyectil.
        move3 (float): Posición inicial en Z del proyectil.
        dir_resultados (Path | None): Directorio donde se guardarán los resultados.
                                      Por defecto es './resultados'.
        config (dict | None): Configuración cargada desde config.yaml para
                              incluir metadatos en el gráfico.

    Returns:
        None
    """
    if dir_resultados is None:
        dir_resultados = Path("./resultados")

    # Asegurar que el directorio de imágenes existe
    dir_imagenes = dir_resultados / "Imagenes"
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    # 3. Armamos el plan de filtrado y conteo
    df_plot = (
        df
        # Agrupamos por el paso de tiempo 'i'
        .group_by("i")
        # Sumamos como 'cantidad_eyectados' solo aquellas partículas que cumplen
        # la condición de erosión
        .agg(
            (pl.col("proyectil_erosion").is_null() | pl.col("proyectil_erosion"))
            .sum()
            .alias("cantidad_eyectados")
        )
        # Ordenamos cronológicamente para que el gráfico no salga cruzado
        .sort("i")
    )

    # 4. Graficamos los resultados
    plt.figure(figsize=(10, 6))

    # Polars permite extraer las columnas directamente como si fueran listas
    plt.plot(
        df_plot["i"],
        df_plot["cantidad_eyectados"],
        marker="o",
        linestyle="-",
        color="b",
    )

    # Construir título con metadatos completos
    titulo = "Evolución de partículas eyectadas\n"
    titulo += f"(v₀={vel} m/s, ff={ff}, N={int(np_)}, θ={angle}°)"

    # Agregar información de configuración si está disponible
    if config:
        material = config.get("material", {}).get("nombre", "Material")
        version = config.get("metadatos", {}).get("version", "N/A")
        titulo += f"\n{material} | Config v{version}"

    # Ponemos linda la gráfica
    plt.title(titulo, fontsize=12)
    plt.xlabel("Iteración (i)", fontsize=12)
    plt.ylabel("Cantidad total de partículas eyectadas", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Agregar leyenda con parámetros de simulación
    texto_parametros = (
        f"Parámetros:\n"
        f"  • Fricción: ff = {ff}\n"
        f"  • Partículas: N = {int(np_)}\n"
        f"  • Velocidad: v₀ = {vel} m/s\n"
        f"  • Ángulo: θ = {angle}°\n"
        f"  • Posición: ({move1}, {move2}, {move3})"
    )

    # Agregar texto con parámetros en la esquina superior derecha
    plt.text(
        0.98,
        0.98,
        texto_parametros,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Mostrar el gráfico
    plt.tight_layout()

    # Guardo la imagen con todos los parametros de la simulación
    nombre_archivo = (
        f"plot_erosion_bed_SI_ff_{ff}_np_{np_}_vel_{vel}_"
        f"angle_{angle}_move_{move1}_{move2}_{move3}.svg"
    )
    ruta_guardado = dir_imagenes / nombre_archivo
    plt.savefig(ruta_guardado)
    print(f"Gráfico guardado en: {ruta_guardado}")

    # Cerrar la figura para liberar memoria
    plt.close()


def plot_erosion_desde_parquet(
    ruta_parquet: str | Path,
    dir_resultados: Path | None = None,
    config: dict | None = None,
) -> None:
    """
    Genera un gráfico de erosión leyendo datos directamente desde un archivo Parquet.

    Args:
        ruta_parquet (str | Path): Ruta al archivo Parquet con los datos.
        dir_resultados (Path | None): Directorio donde se guardarán los resultados.
        config (dict | None): Configuración cargada desde config.yaml.

    Returns:
        None
    """
    # Leer el archivo Parquet
    df = pl.read_parquet(ruta_parquet)

    # Extraer metadatos del DataFrame
    try:
        ff = float(df["ff"].item(0))
        np_ = float(df["np"].item(0))
        vel = float(df["vel0"].item(0))
        angle = float(df["angulo"].item(0))
        move1 = float(df["posx"].item(0))
        move2 = float(df["posy"].item(0))
        move3 = float(df["posz"].item(0))

        plot_erosion(
            df,
            ff,
            np_,
            vel,
            angle,
            move1,
            move2,
            move3,
            dir_resultados,
            config,
        )
    except (KeyError, ValueError) as e:
        print(f"No se pudieron extraer los metadatos para el gráfico: {e}")


def plot_erosion_interactivo() -> None:
    """
    Función interactiva para generar gráficos de erosión.

    Solicita al usuario los parámetros de filtrado y genera el gráfico
    correspondiente a partir de los archivos parquet en el directorio
    de resultados.
    """
    # 1. Definimos los parámetros que queremos filtrar
    parametros: dict[str, float | None] = {
        "ff": None,
        "np": None,
        "vel": None,
        "angle": None,
        "move1": None,
        "move2": None,
        "move3": None,
    }

    for key in parametros:
        while True:
            valor = input(f"Ingresa el valor para {key}: ")
            try:
                parametros[key] = float(valor)
                break
            except ValueError:
                print("Valor inválido, intenta de nuevo.")

    # 2. Conectamos con los datos de forma perezosa
    ruta_archivos = "./resultados/*_TOTAL_*.parquet"
    lf = pl.scan_parquet(ruta_archivos)

    # 3. Armamos el plan de filtrado y conteo
    df_plot = (
        lf
        # Filtramos por las condiciones físicas de la simulación
        .filter(
            (pl.col("vel0").cast(pl.Float64) == parametros["vel"])
            & (pl.col("ff").cast(pl.Float64) == parametros["ff"])
        )
        # Agrupamos por el paso de tiempo 'i'
        .group_by("i")
        # Sumamos como 'cantidad_eyectados' solo aquellas partículas que
        # cumplen la condición de erosión
        .agg(
            (pl.col("proyectil_erosion").is_null() | pl.col("proyectil_erosion"))
            .sum()
            .alias("cantidad_eyectados")
        )
        # Ordenamos cronológicamente para que el gráfico no salga cruzado
        .sort("i")
        # ¡Ahora sí, ejecutamos todo y lo traemos a la RAM!
        .collect()
    )

    # 4. Graficamos los resultados
    plt.figure(figsize=(10, 6))

    # Polars permite extraer las columnas directamente como si fueran listas
    plt.plot(
        df_plot["i"],
        df_plot["cantidad_eyectados"],
        marker="o",
        linestyle="-",
        color="b",
    )

    # Ponemos linda la gráfica
    plt.title(
        f"Evolución de partículas eyectadas (v0={parametros['vel']}, "
        f"ff={parametros['ff']})",
        fontsize=14,
    )
    plt.xlabel("Iteración (i)", fontsize=12)
    plt.ylabel("Cantidad total de partículas", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Mostrar el gráfico
    plt.tight_layout()

    # Guardo la imagen con todos los parametros de la simulación
    plt.savefig(
        f"./resultados/Imagenes/plot_erosion_bed_SI_ff_{parametros['ff']}_"
        f"np_{parametros['np']}_vel_{parametros['vel']}_angle_{parametros['angle']}_"
        f"move_{parametros['move1']}_{parametros['move2']}_{parametros['move3']}.svg"
    )


# Punto de entrada para ejecución independiente
if __name__ == "__main__":
    plot_erosion_interactivo()
