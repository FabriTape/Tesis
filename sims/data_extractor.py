# -*- coding: utf-8 -*-
import gzip
import os
from pathlib import Path

import polars as pl
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource


def extractor_de_borde(
    ruta: str, nombre_archivo: str, columnas={"x": 5, "y": 6}
) -> dict:
    """Esta función permite extraer los datos de un archivo output.0.gz para
    encontrar los bordes de la simulacion de la caja y devolverlo en forma de
    diccionario.
    Args:
        - ruta(str): Es la ruta absoluta donde se encuentran los datos de la simulación.
        - nombre_archivo(str): Es el nombre del archivo de donde se extraerán los datos.
        - columnas(dict): Diccionario cuyas llaves son los ejes de los cuales se extraerá
        información y los valores es la linea en las que se encuentra.
    Return:
        - resultados(dict): Diccionario cuyas llaves son los nombres de los ejes a los que
        se les extrajo los límites y en los valores hay listas con los valores mínimos
        y máximos."""
    resultados = {}
    # Sea cual sea los ejes que ponga los resultados se adaptarán a esa forma
    for llave in columnas.keys():
        resultados[llave] = []
    # Abrimos el archivo
    with gzip.open(os.path.join(ruta, nombre_archivo), "rt") as f:
        # Damos vuelta llaves y valores
        indice_a_llave = {v: k for k, v in columnas.items()}
        # Vamos leyendo linea por linea
        for i, line in enumerate(f):
            # Si la linea esta entre las que se quiere leer se sca el dato
            if i in indice_a_llave:
                # Se guarda la llave que se esta guardando
                llave_actual = indice_a_llave[i]
                # Sacamos la linea, la separamos y quitamos los \n
                # luego la convertimos en una list con entradas de valores flotantes
                datos = [float(number) for number in line.replace("\n", "").split(" ")]
                # Guardamos datos
                resultados[llave_actual].extend(datos)
    return resultados


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
    os.system("cls" if os.name == "nt" else "clear")


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
            items = sorted(
                [p for p in ruta.iterdir() if p.is_dir() and not p.name.startswith(".")]
            )
        # En caso de haber una carpeta con permisos de accesos te tira qje no podes entrar
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
        # En caso de haber archivos gz te muestra uno y la cantidad que hay
        if datos_gz:
            N = len(datos_gz)
            ejemplo = datos_gz[0].name
            print(f"\nSe encontraron {N} archivos .gz (ej. {ejemplo})")

        try:
            # Pide otra entrada
            entrada = input(mensaje)
            eleccion = int(entrada)

            if eleccion == -1:
                # Elegimos la carpeta actual
                limpiar_pantalla()
                return str(ruta)

            elif eleccion == -2:
                # Nos vamos a la carpeta de arriba
                ruta = ruta.parent
                limpiar_pantalla()

            elif 1 <= eleccion <= len(items):
                # Entramos a una de las carpetas que esten adentro
                ruta = items[eleccion - 1]
                limpiar_pantalla()
            else:
                # Error de indice, vuelve al principio
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
    # Iniciamos la busqueda de la carpeta
    ruta_str = buscar_carpeta()
    # La ruta de trabajo seleccionada
    ruta = Path(ruta_str)
    base = Path("./resultados")
    # Creo la carpeta y si ya existe no pasa naranja
    base.mkdir(exist_ok=True)
    # agarro la ruta actual y la divido en carpetas
    nombre = str(ruta).split("/")
    # La ultima carpeta es la que se está analizando
    datos = nombre[-1].replace("erosion_bed_SI_", "", 1)

    return str(ruta), datos


# Ver si cambiando nombres columnas por datos funciona
def Lector(archivo_gz: str, datos: list) -> pl.LazyFrame:
    """Lector es una funcion que permite leer los archivos output.gz de las simulaciones
    de LAMMPS y extrae solo unos datos
    Args:
        archivo_gz(str): Indica el nombre y ruta del archivo a leer
        datos(list): Es una lista que permite seleccionar una subparte de todas las
                     columnas"""
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    lf_archivo = pl.scan_csv(
        archivo_gz,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,  # Maneja espacios extra al final
        new_columns=nombres_columnas,
        infer_schema_length=10000,
        schema_overrides={"type": pl.UInt8},
    )
    return lf_archivo.select(datos)


def progress(i, N):
    """Imprime una barra de progreso visual en la terminal.
    Args:
        i (int): Iteración actual.
        N (int): Total de iteraciones.
    Returns:
        None
    """
    if i == 0:
        print("Calculando parámetros iniciales...")
        print("Iniciando análisis de datos")
        print("000/100%[", 22 * "·", "]", sep="", end="")
    print(
        "\r",
        "{:03d}".format(int((i * 100 / N) // 1)),
        "/100%[",
        int((i * 100 / N) // 5) * "_" + "@ï",
        sep="",
        end="",
    )


def detector_de_ondas(
    lf: pl.LazyFrame,
    R: float,
    altura: float,
    limites: dict,
    usar_caja: bool,
    umbral_techo=3,
    umbral_pared=20,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """Filtra partículas de la onda de choque y calcula métricas de interés.
    Args:
        - lf (pl.LazyFrame): Datos de entrada.
        - R (float): Radio de la partícula.
        - altura (float): Altura de referencia de la cama de partículas.
        - limites (dict): Un diccionario con los limites de la caja.
        - usar_caja (bool): Booleano que permite elegir entre usar el tamaño
        de la caja o el máximo.
        - umbral_techo (int): Umbral de filtrado con la altura.
        - umbral_pared (int): Umbral de filtrado con las paredes de la caja.
    Returns:
        - lf_onda (pl.LazyFrame) : Partículas filtradas que componen la onda.
        - df_res_onda (pl.DataFrame) : Conteo y velocidad máxima de la onda.
    """

    lf_onda = lf.with_columns(
        # Crea una columna con la velocidad cuadrática
        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v^2")
    ).filter(
        # Filtrado por altura
        (pl.col("z") < altura - umbral_techo * R)
        # Filtrado por velocidad mínima para considerar una onda relevante
        & (pl.col("v^2") > 4e-4)
    )
    if usar_caja:
        lf_onda.filter(
            (
                # Filtrado para ver si llega como mínimo a 20 (default) radios de la pared
                # usando los limites de la caja
                (pl.col("x") < umbral_pared * R)
                | (limites["x"][1] - umbral_pared * R < pl.col("x"))
                | (pl.col("y") < umbral_pared * R)
                | (limites["y"][1] - umbral_pared * R < pl.col("y"))
            )
        )
    else:
        lf_onda.filter(
            (
                # Filtrado para ver si llega como mínimo a 20 (default) radios de la pared
                # usando máximo y mínimo de partículas
                (pl.col("x") < umbral_pared * R)
                | (pl.col("x").max() - umbral_pared * R < pl.col("x"))
                | (pl.col("y") < umbral_pared * R)
                | (pl.col("y").max() - umbral_pared * R < pl.col("y"))
            )
        )
    # Se hace una lista con la cantidad de partículas en la onda y cual es la velocidad máxima
    df_res_onda = lf_onda.select(
        [pl.len().alias("cantidad"), pl.col("v^2").max().alias("v^2_max")]
    ).collect()
    return lf_onda, df_res_onda


def analisis_ovito(lf: pl.LazyFrame, cutoff: float, altura: float) -> pl.DataFrame:
    """Integra OVITO para calcular coordinación y clusters en partículas.
    Args:
        lf (pl.LazyFrame): Datos de las partículas eyectadas.
        cutoff (float): Distancia de corte para definir vecinos.
        altura (float): altura maxima de la caja
    Returns:
        (pl.LazyFrame) : Datos originales con columnas de coordinación y cluster.
    """
    # 1. Colectar a DataFrame
    df = lf.filter((pl.col("z") > altura) & (pl.col("vz") > 0)).collect()

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

    # 8. Agregar columnas
    return df.with_columns(
        [pl.Series("coordinacion", coordinacion), pl.Series("cluster", cluster)]
    )


def max_altura(ruta: str, R: float, umbral_techo=3) -> float:
    """Calcula la altura máxima de la cama con un umbral default de 3 radios de
    partículas
    Args:
        umbral_techo (int): Cantidad de radios de umbral. Default = 3
        R (float): Radio de la partícula en [m]
    Return:
        altura (float): altura máxima de filtrado"""
    lf0 = Lector(os.path.join(ruta, "output.0.gz"), ["type", "z"])
    max_z_type1 = (
        lf0.filter(pl.col("type") == 1).select(pl.col("z").max()).collect().item()
    )

    if max_z_type1 is None:
        raise ValueError("No puede ser que la altura maxima de la cama sea None")
    altura = umbral_techo * R + max_z_type1
    return altura


def data_analysis():
    """Orquestador principal que ejecuta el flujo de análisis completo.
    Args:
        None
    Returns:
        None
    """

    ruta, datos = generar_carpeta()
    # Radio de los granos
    R = float(input("ingrese el valor de radio de los granos en [um]: ")) * 1e-6
    limpiar_pantalla()
    # Busco los archivos de la carpeta
    archivos = [d for d in os.listdir(ruta) if ("output." in d) and (".gz" in d)]

    # Ordeno los archivos gz por numero de paso
    archivos.sort(key=lambda x: "{0:0>19}".format(x).lower())

    # Busco los limites de la caja entre el archivo inicial y el final
    limites_i = extractor_de_borde(ruta, archivos[0])
    limites_f = extractor_de_borde(ruta, archivos[-1])

    # Sesteo esto por default
    usar_caja = True

    # Si los limites varian dejo la posibilidad de aun usarlos
    # o simplemente cambiar al método tradicional de buscar el máximimo
    if limites_f != limites_i:
        while True:
            seguir = input(
                f"""Los bordes de las caja cambiaron entre {archivos[0]} y {archivos[-1]}
    En {archivos[0]} son:
    {limites_i}
    En {archivos[1]} son:
    {limites_f}
    ¿Deseás aun asi usar los bordes de la caja para el cálculo? (Y/N)"""
            )
            if seguir in "Nn":
                usar_caja = False
                break
            elif seguir in "Yy":
                break
            else:
                limpiar_pantalla()
                print("(No se registró ninguna entrada válida usar Y o N)")

    # Busco cada cuantos pasos registra datos la simulación
    paso_str = archivos[1].replace("output.", "").replace(".gz", "")
    paso_int = int(paso_str)
    # Cantidad de archivos
    LenTotal = len(archivos)
    # Estas son las columnas que se van a leer en el bucle for
    columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]
    # Cálculo inicial de altura máxima de umbral
    altura = max_altura(ruta, R)

    for i, file in enumerate(archivos):
        progress(i, LenTotal)

        # Lectura de los datos con columnas espcificas
        lf_datos = Lector(f"{ruta}/{file}", columnas)

        # Buscamos si la onda llego al borde
        lf_onda, df_res_onda = detector_de_ondas(
            lf_datos, R, altura, limites_f, usar_caja
        )
        # Estaria bueno ver si haciendo el analisis de la onda de
        # hacia atras se puede hacer de manera mas rápida sin repetir en todos los pasos.

        cantidad = df_res_onda.item(0, "cantidad")
        max_v2 = df_res_onda.item(0, "v^2_max")
        bool_v2_max = (max_v2 >= 4e-2) if isinstance(max_v2, float) else False

        if cantidad > 3000 or bool_v2_max:
            print(f"""\nSe analizaron {i} archivos y en el archivo {file}:
                  Se detectó una onda de {cantidad} partículas
                  Con una v² máxima de {max_v2}""")
            ruta_guardado = f"./resultados/{datos}_ONDA_{i}.parquet"
            lf_onda.sink_parquet(ruta_guardado)
            break

        else:
            parametros = datos.split("_")
            df_procesados = analisis_ovito(lf_datos, R, altura)
            df_procesados_final = df_procesados.with_columns(
                [
                    pl.lit(parametros[1]).alias("ff"),
                    pl.lit(parametros[3]).alias("np"),
                    pl.lit(parametros[5]).alias("vel0"),
                    pl.lit(parametros[-5]).alias("angulo"),
                    pl.lit(parametros[-3]).alias("posx"),
                    pl.lit(parametros[-2]).alias("posy"),
                    pl.lit(parametros[-1]).alias("posz"),
                    pl.lit(paso_int).alias("paso"),
                    pl.lit(i).alias("i"),
                ]
            )
            if i == 0:
                df_temp = df_procesados_final
            else:
                df_temp = pl.concat(
                    [df_temp, df_procesados_final], how="vertical_relaxed"
                )

    if df_temp.shape[0] == 0:
        with open(f"./resultados/{datos}_NADA.txt", "w") as archivo:
            archivo.write("No se obtuvo erosión en esta simulación")
    else:
        df_temp.write_parquet(f"./resultados/{datos}_TOTAL.parquet")
    progress(100, 100)
    print("\nSe terminó el análisis exitosamente")
    print("Datos guardados en: ./resultados/")


data_analysis()
