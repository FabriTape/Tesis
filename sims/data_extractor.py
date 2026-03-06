#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
import polars as pl
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

def limpiar_pantalla():
    """
    Limpia la salida de la terminal dependiendo del sistema operativo.

    Soporta:
        - Windows ('nt') -> cls
        - Linux/macOS ('posix') -> clear
    Args:
        None
    Returns:
        None
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def buscar_carpeta():
    """
    Interfaz de navegación interactiva por terminal para seleccionar un directorio.

    Permite al usuario moverse entre carpetas, subir de nivel o seleccionar
    la carpeta actual. Utiliza pathlib para garantizar rutas seguras.
    Args:
        None
    Returns:
        str: Ruta absoluta de la carpeta seleccionada por el usuario.
    """
    mensaje = """\nIngrese el numero de la carpeta a la que desea acceder
[-1] Seleccionar carpeta actual
[-2] Volver atrás : """

    # Iniciamos en el directorio actual, resuelto a ruta absoluta
    ruta = Path(".").resolve()

    while True:
        # Listamos solo directorios, ordenados y sin ocultos
        try:
            items = sorted([p for p in ruta.iterdir() if p.is_dir()
                            and not p.name.startswith(".")])
        except PermissionError:
            print(f"No tienes permiso para acceder a {ruta}")
            ruta = ruta.parent
            continue

        # Buscamos archivos output para referencia
        datos_gz = sorted(list(ruta.glob("output.*.gz")))

        print(f"\n--- Ubicación actual: {ruta} ---")
        print("Carpetas disponibles:")
        for i, carpeta in enumerate(items, start=1):
            print(f"[{i}] {carpeta.name}")

        if datos_gz:
            N = len(datos_gz)
            ejemplo = datos_gz[0].name
            print(f"\nSe encontraron {N} archivos .gz (ej. {ejemplo})")

        try:
            entrada = input(mensaje)
            eleccion = int(entrada)

            if eleccion == -1:
                limpiar_pantalla()
                return str(ruta) # Retornamos string para compatibilidad

            elif eleccion == -2:
                ruta = ruta.parent
                limpiar_pantalla()

            elif 1 <= eleccion <= len(items):
                ruta = items[eleccion-1]
                limpiar_pantalla()
            else:
                limpiar_pantalla()
                print("Número inválido.")

        except ValueError:
            limpiar_pantalla()
            print("Entrada inválida.")

def generar_carpeta():
    """
    Gestiona la selección de la carpeta de origen y crea la carpeta de salida.

    La carpeta de salida tendrá un nombre con marca de tiempo (timestamp)
    para evitar sobrescribir análisis previos.
    Args:
        None
    Returns:
        tuple[str, str]: Un par conteniendo (ruta_origen, ruta_destino).
    """
    ruta_str = buscar_carpeta()
    ruta = Path(ruta_str)

    # Creamos nombre de carpeta con fecha
    timestamp = (datetime.now(UTC) - timedelta(hours=3)
                 ).strftime('_%Y-%m-%d_%H:%M')

    base = ruta / f"datos_analisis{timestamp}"

    base.mkdir(exist_ok=True)
    return str(ruta), str(base)

def Lector(archivo_gz: str, datos: list):
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    df_archivo = pl.scan_csv(
        archivo_gz,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True, # Maneja espacios extra al final
        new_columns=nombres_columnas,
        separator=" ",  # Cambiamos tab por espacio
        truncate_ragged_lines=True,  # Ignora espacios extra al final de la línea
        infer_schema_length=10000,
        schema_overrides={"type": pl.UInt8, "id": pl.UInt64},
    )

    return df_archivo.select(datos)


def limpiar_pantalla():
    """Limpia la terminal según el sistema operativo (Windows o Linux/macOS).
    Args:
        None
    Returns:
        None
    """
    # 'nt' es Windows, 'posix' es Linux/macOS
    os.system("cls" if os.name == "nt" else "clear")


def rreplace(s, old, new="", ocurrence=1):
    """Reemplaza una subcadena buscando desde el final del string (derecha).
    Args:
        s (str) : Cadena de texto original.
        old (str) : Subcadena que se desea reemplazar.
        new (str) : Subcadena nueva que reemplazará a la antigua.
        ocurrence (int) : Cantidad de veces que se realizará el reemplazo.
    Returns:
        (str) : Cadena de texto con los reemplazos realizados.
    """
    li = s.rsplit(old, ocurrence)
    return new.join(li)


# Funcion obtenida de
# https://www.geeksforgeeks.org/python/delete-a-directory-or-file-using-python/
"""Esta funcion esta en desuso y será removida en suturas versiones"""


def borrar_dir(ruta):
    """Elimina todos los archivos de un directorio y luego la carpeta misma.
    Args:
        ruta (str) : Trayectoria de la carpeta que se desea eliminar.
    Returns:
        None
    """
    for filename in os.listdir(ruta):
        file_path = os.path.join(ruta, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the file
    os.rmdir(ruta)


def buscar_carpeta():
    """Permite la navegación interactiva usando objetos de ruta (Path)."""

    mensaje = """\nIngrese el numero de la carpeta a la que desea acceder
ingresar [-1] para elegir los datos de la carpeta actual
ingresar [-2] para volver atras : """

    # Iniciamos con la ruta actual absoluta
    ruta = Path(".").resolve()

    while True:
        # Listar carpetas (filtrando ocultas)
        carpetas = sorted(
            [d for d in ruta.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

        # Listar archivos de simulación (solo para visualización)
        datos_gz = sorted([f.name for f in ruta.glob("output.*.gz")])

        print(f"\n--- Ubicación actual: {ruta} ---")
        print("Carpetas disponibles:")
        for i, carpeta in enumerate(carpetas, start=1):
            print(f"[{i}] {carpeta.name}")

        if datos_gz:
            print("\nArchivos de datos encontrados (muestra):")
            for dato in datos_gz[:3]:
                print(f"  {dato}")

        try:
            eleccion = int(input(mensaje))

            if 1 <= eleccion <= len(carpetas):
                # Ir hacia adelante
                ruta = carpetas[eleccion - 1]
                limpiar_pantalla()

            elif eleccion == -1:
                # Seleccionar carpeta actual
                limpiar_pantalla()
                return str(
                    ruta
                )  # Retornamos como string para que el resto del código no cambie

            elif eleccion == -2:
                # Ir hacia atrás de forma segura
                ruta = ruta.parent
                limpiar_pantalla()

            else:
                limpiar_pantalla()
                print("Número inválido, intente de nuevo.")

        except ValueError:
            print("Entrada inválida, por favor ingrese un número.")


def generar_carpeta():
    """Crea una carpeta para los resultados gestionando la sobreescritura.
    Args:
        None
    Returns:
        ruta (str) : Ruta de la carpeta de origen de los datos.
        base (str) : Ruta de la carpeta de destino para el análisis.
    """
    ruta = buscar_carpeta()
    base = os.path.join(ruta, "datos_analisis")

    base += (datetime.now(UTC) + timedelta(hours=-3)).strftime("_%Y-%m-%d_%H:%M")
    os.mkdir(base)
    return ruta, base


def progress(i, N):
    """Imprime una barra de progreso visual en la terminal.
    Args:
        i (int): Iteración actual.
        N (int): Total de iteraciones.
    Returns:
        None
    """
    print(
        "\r",
        "{:03d}".format(int((i * 100 / N) // 1)),
        "/100%[",
        int((i * 100 / N) // 5) * "_" + "@ï",
        sep="",
        end="",
    )


def detector_de_ondas(lf, R, altura):
    """Filtra partículas de la onda de choque y calcula métricas de interés.
    Args:
        lf (pl.LazyFrame): Datos de entrada.
        R (float): Radio de la partícula.
        altura (float): Altura de referencia de la cama de partículas.

    Returns:
        lf_onda (pl.LazyFrame) : Partículas filtradas que componen la onda.
        df_res_onda (pl.DataFrame) : Conteo y velocidad máxima de la onda.
    """
    lf_onda = lf.with_columns(
        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v^2")
    ).filter(
        (pl.col("z") < altura - 3 * R)
        & (pl.col("v^2") > 4e-4)
        & (
            (pl.col("x") < 10 * R)
            | (pl.col("x").max() - 10 * R < pl.col("x"))
            | (pl.col("y") < 10 * R)
            | (pl.col("y").max() - 10 * R < pl.col("y"))
        )
    )
    df_res_onda = lf_onda.select(
        [pl.len().alias("cantidad"), pl.col("v^2").max().alias("v^2_max")]
    ).collect()
    return lf_onda, df_res_onda


def analisis_ovito(lf: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    """Integra OVITO para calcular coordinación y clusters en partículas.
    Args:
        lf (pl.LazyFrame): Datos de las partículas eyectadas.
        cutoff (float): Distancia de corte para definir vecinos.

    Returns:
        (pl.LazyFrame) : Datos originales con columnas de coordinación y cluster.
    """
    # 1. Colectar a DataFrame
    df = lf.collect()

    # 2. Extraer posiciones como NumPy array
    positions = df.select(["x", "y", "z"]).to_numpy()

    # 3. Crear colección de datos manualmente (la forma correcta para datos
    #    en memoria)
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property("Position", data=positions)

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)
    # 4. Crear pipeline con fuente estática
    pipeline = Pipeline(source=StaticSource(data=data))

    # 5. Agregar modificadores
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    # 6. Computar
    computed_data = pipeline.compute()

    # 7. Extraer resultados
    coordinacion = computed_data.particles["Coordination"][:]
    cluster = computed_data.particles["Cluster"][:]

    # 8. Agregar columnas y devolver LazyFrame
    return df.with_columns(
        [pl.Series("coordinacion", coordinacion), pl.Series("cluster", cluster)]
    ).lazy()


def data_analysis():
    """Orquestador principal que ejecuta el flujo de análisis completo.
    Args:
        None
    Returns:
        None
    """
    ruta, base = generar_carpeta()
    # Radio de los granos
    R = float(
        input("""ingrese el valor de radio de los granos en [m]
(se puede usar formato xe-y [m]): """)
    )
    limpiar_pantalla()

    archivos = [d for d in os.listdir(ruta) if ("output." in d) and (".gz" in d)]

    archivos.sort(key=lambda x: "{0:0>19}".format(x).lower())

    LenTotal = len(archivos)
    # Variables constantes fuera del bucle
    lf0 = Lector(f"{ruta}/output.0.gz", ["type", "z"])

    # Filtramos donde type == 1 y buscamos el maximo de z
    altura = 3 * R + (
        lf0.filter(pl.col("type") == 1).select(pl.col("z").max()).collect().item()
    )
    ## Estas son las columnas que se van a leer en el bucle for
    columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    # Cálculo inicial de altura (lee el archivo 0)
    print("Calculando parámetros iniciales...")
    lf0 = Lector(os.path.join(ruta, "output.0.gz"), ["type", "z"])
    max_z_type1 = (lf0
                   .filter(pl.col("type") == 1)
                   .select(pl.col("z").max())
                   .collect()
                   .item()
                   )
    if max_z_type1 is None: max_z_type1 = 0
    altura = 4 * R + max_z_type1

    print("Iniciando análisis de datos")
    print("000/100%[", 22 * "·", "]", sep="", end="")
    for i, file in enumerate(archivos):
        progress(i, LenTotal)

        # Lectura de los datos con columnas espcificas

        lf_datos = Lector(f"{ruta}/{file}", columnas)

        # Buscamos si la onda llego al borde
        lf_onda, df_res_onda = detector_de_ondas(lf_datos, R, altura)

        cantidad = df_res_onda.item(0, "cantidad")
        max_v2 = df_res_onda.item(0, "v^2_max")
        bool_v2_max = (max_v2 >= 4e-2) if isinstance(max_v2, float) else False

        if cantidad > 3000 or bool_v2_max:
            print(f"""\nSe analizaron {i} archivos y en el archivo {file}:
                  Se detectó una onda de {cantidad} partículas
                  Con una v² máxima de {max_v2}""")
            ruta_guardado = os.path.join(base, f"onda.{i}.parquet")
            lf_onda.sink_parquet(ruta_guardado)
            break

        else:
            lf_eyectados = lf_datos.filter(pl.col("z") > altura & pl.col("vz") > 0)

            lf_procesados = analisis_ovito(lf_eyectados, R)

            df_eyectados_proy = lf_procesados.filter(pl.col("type") == 2).collect()
            df_eyectados_cama = lf_procesados.filter(pl.col("type") == 1).collect()

            # Verdadero si se eyectaron
            bool_eyectados_cama = df_eyectados_cama.shape[0] != 0
            bool_eyectados_proy = df_eyectados_proy.shape[0] != 0

            # Solo escribe archivos en casos de haber uno eyectado
            if bool_eyectados_cama or bool_eyectados_proy:
                if bool_eyectados_cama:
                    ruta_guardado_cama = os.path.join(base, f"cama.{i}.parquet")
                    df_eyectados_cama.write_parquet(ruta_guardado_cama)

                if not bool_eyectados_proy:
                    ruta_guardado_proy = os.path.join(base, f"proy.{i}.parquet")
                    df_eyectados_proy.write_parquet(ruta_guardado_proy)

    progress(100, 100)
    print("\nSe terminó el análisis exitosamente")
    print(f"{LenTotal} datos guardados en: {base}")


data_analysis()
