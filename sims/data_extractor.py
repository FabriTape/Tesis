# -*- coding: utf-8 -*-
import gzip
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from numpy import empty as npempty
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

# ==========================================
# PARÁMETROS FÍSICOS Y UMBRALES DE ANÁLISIS
# ==========================================
UMBRAL_V2_MIN = 4e-4  # Energía cinética mínima para considerar una partícula activa
UMBRAL_V2_MAX = 4e-2  # Energía cinética máxima para considerar un impacto crítico
PARTICULAS_CRITICAS = (
    3000  # Cantidad de partículas en movimiento para detectar onda macroscópica
)


def extractor_de_borde(
    ruta: str, nombre_archivo: str, columnas={"x": 5, "y": 6}
) -> dict:
    """
    Extrae los límites espaciales de la caja de simulación a partir de un archivo de salida comprimido.

    Args:
        ruta (str): Ruta absoluta o relativa al directorio que contiene los datos de la simulación.
        nombre_archivo (str): Nombre del archivo comprimido (ej. 'output.0.gz').
        columnas (dict): Diccionario que mapea los nombres de los ejes a la línea específica
                         del encabezado donde se encuentran sus límites.

    Returns:
        dict: Diccionario donde las llaves son los ejes analizados y los valores son listas
              conteniendo los límites mínimos y máximos de la caja en dicho eje.
    """
    resultados = {}
    # Inicializa las listas vacías adaptándose a los ejes solicitados en 'columnas'
    for llave in columnas.keys():
        resultados[llave] = []

    # Abre y lee el archivo comprimido en modo texto
    ruta_archivo = Path(ruta) / nombre_archivo
    with gzip.open(ruta_archivo, "rt") as f:
        # Inversión del diccionario para acceder rápidamente a la llave por el número de línea
        indice_a_llave = {v: k for k, v in columnas.items()}

        # Iteración línea por línea para no cargar todo el archivo en memoria
        for i, line in enumerate(f):
            # Verifica si la línea actual corresponde a uno de los límites buscados
            if i in indice_a_llave:
                llave_actual = indice_a_llave[i]

                # Limpia la línea, separa los valores por espacio y los convierte a flotantes
                datos = [float(number) for number in line.replace("\n", "").split(" ")]

                # Almacena los límites [mínimo, máximo] encontrados
                resultados[llave_actual].extend(datos)
    return resultados


def limpiar_pantalla():
    """
    Limpia la salida de la terminal de forma multiplataforma.

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
    Despliega una interfaz interactiva en la terminal para navegar y seleccionar un directorio.

    Permite al usuario moverse entre carpetas, subir un nivel jerárquico o confirmar
    la carpeta actual como directorio de trabajo.

    Args:
        None
    Returns:
        str: Ruta absoluta del directorio seleccionado.
    """
    mensaje = """\nIngrese el numero de la carpeta a la que desea acceder
[-1] Seleccionar carpeta actual
[-2] Volver atrás : """

    # Define el directorio de inicio resolviendo su ruta absoluta
    ruta = Path(".").resolve()

    while True:
        try:
            # Lista directorios ordenados alfabéticamente, excluyendo archivos y carpetas ocultas
            items = sorted(
                [p for p in ruta.iterdir() if p.is_dir() and not p.name.startswith(".")]
            )
        except PermissionError:
            # Manejo de excepciones para carpetas restringidas por el sistema operativo
            print(f"No tienes permiso para acceder a {ruta}")
            ruta = ruta.parent
            continue

        # Busca archivos de simulación como referencia visual para el usuario
        datos_gz = sorted(list(ruta.glob("output.*.gz")))

        print(f"\n--- Ubicación actual: {ruta} ---")
        print("Carpetas disponibles:")
        for i, carpeta in enumerate(items, start=1):
            print(f"[{i}] {carpeta.name}")

        # Informa sobre la presencia de archivos de datos procesables en la carpeta actual
        if datos_gz:
            N = len(datos_gz)
            ejemplo = datos_gz[0].name
            print(f"\nSe encontraron {N} archivos .gz (ej. {ejemplo})")

        try:
            entrada = input(mensaje)
            eleccion = int(entrada)

            if eleccion == -1:
                # Confirma la selección de la carpeta actual
                limpiar_pantalla()
                return str(ruta)

            elif eleccion == -2:
                # Navega al directorio padre (sube un nivel)
                ruta = ruta.parent
                limpiar_pantalla()

            elif 1 <= eleccion <= len(items):
                # Navega hacia el subdirectorio seleccionado
                ruta = items[eleccion - 1]
                limpiar_pantalla()
            else:
                limpiar_pantalla()
                print("Número inválido.")

        except ValueError:
            limpiar_pantalla()
            print("Entrada inválida. Por favor, ingrese un número.")


def generar_carpeta():
    """
    Gestiona la selección del directorio de origen y asegura la existencia del directorio de salida.

    Genera una carpeta base llamada './resultados' si no existe, y extrae los
    metadatos de la simulación a partir del nombre de la carpeta seleccionada.

    Args:
        None
    Returns:
        tuple[str, str]: Tupla que contiene la ruta de trabajo seleccionada y el
                         identificador de la simulación extraído del nombre del directorio.
    """
    ruta_str = buscar_carpeta()
    ruta = Path(ruta_str)

    base = Path("./resultados")
    # Crea el directorio de resultados de forma segura (ignora si ya existe)
    base.mkdir(exist_ok=True)

    # Fragmenta la ruta para aislar el nombre del directorio final
    nombre = str(ruta).split("/")

    # Limpia el prefijo del nombre de la carpeta para usarlo como identificador base de los datos
    datos = nombre[-1].replace("erosion_bed_SI_", "", 1)

    return str(ruta), datos


def Lector(archivo_gz: str, datos: list) -> pl.LazyFrame:
    """
    Lee archivos de volcado (dump) comprimidos de LAMMPS optimizando el uso de memoria.

    Utiliza evaluación perezosa (LazyFrame) de Polars para escanear el archivo,
    omitiendo encabezados innecesarios y filtrando únicamente las columnas requeridas.

    Args:
        archivo_gz (str): Ruta completa al archivo comprimido a procesar.
        datos (list): Lista de cadenas con los nombres de las columnas que se desean extraer.

    Returns:
        pl.LazyFrame: Estructura de datos perezosa de Polars lista para ser procesada.
    """
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    lf_archivo = pl.scan_csv(
        archivo_gz,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,  # Tolerancia para líneas con espacios finales irregulares
        new_columns=nombres_columnas,
        infer_schema_length=10000,
        schema_overrides={
            "type": pl.UInt8
        },  # Optimización de tipo de dato para la columna de ID de tipo
    )
    return lf_archivo.select(datos)


def progress(i: int, N: int, parte=1) -> None:
    """
    Genera y actualiza una barra de progreso visual en la terminal.

    Args:
        i (int): Índice de la iteración actual.
        N (int): Número total de iteraciones esperadas.
        parte (int): Es el modo del mensaje si es por análisis de onda o de ovito:
    Returns:
        None
    """
    mensaje = "onda" if parte else "coordinación"
    if i == 0:
        print(f"Iniciando análisis de {mensaje}")
        print("000/100%[", 22 * "·", "]", sep="", end="")

    # Actualiza la línea de la terminal en la misma posición (usando \r)
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
    """
    Identifica y filtra las partículas que componen la onda de choque generada en la cama.

    Evalúa parámetros cinéticos (velocidad cuadrática) y espaciales (altura y proximidad a fronteras)
    para discriminar la onda del resto del sistema.

    Args:
        lf (pl.LazyFrame): Conjunto de datos de las partículas.
        R (float): Radio de las partículas en el sistema.
        altura (float): Altura de referencia (máxima) de la cama de partículas en reposo.
        limites (dict): Diccionario con los límites espaciales estáticos de la caja.
        usar_caja (bool): Determina si se evalúan los bordes respecto a 'limites' (True)
                          o respecto a los extremos móviles de las partículas (False).
        umbral_techo (int, opcional): Número de radios de tolerancia respecto a la altura. Por defecto es 3.
        umbral_pared (int, opcional): Número de radios de proximidad requerida a las paredes. Por defecto es 20.

    Returns:
        tuple:
            - pl.LazyFrame: Subconjunto de partículas clasificadas como parte de la onda.
            - pl.DataFrame: Resumen métrico (cantidad de partículas involucradas y su velocidad máxima).
    """
    if usar_caja:
        lf_onda = lf.with_columns(
            # Cálculo de la magnitud de la velocidad al cuadrado para criterio cinético
            (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v^2")
        ).filter(
            # 1. Filtro vertical: Descartar partículas eyectadas muy por encima de la cama
            (pl.col("z") < altura - umbral_techo * R)
            # 2. Filtro cinético: Excluir partículas estáticas o con energía despreciable
            & (pl.col("v^2") > UMBRAL_V2_MIN)
            & (
                # 3. Filtro perimetral (Caja Estática): Asegurar que el evento alcance las fronteras
                (pl.col("x") < umbral_pared * R)
                | (limites["x"][1] - umbral_pared * R < pl.col("x"))
                | (pl.col("y") < umbral_pared * R)
                | (limites["y"][1] - umbral_pared * R < pl.col("y"))
            )
        )
    else:
        lf_onda = lf.with_columns(
            # Cálculo de la magnitud de la velocidad al cuadrado
            (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v^2")
        ).filter(
            # 1. Filtro vertical
            (pl.col("z") < altura - umbral_techo * R)
            # 2. Filtro cinético
            & (pl.col("v^2") > UMBRAL_V2_MIN)
            & (
                # 3. Filtro perimetral (Caja Dinámica): Evalúa fronteras basadas en las posiciones actuales
                (pl.col("x") < umbral_pared * R)
                | (pl.col("x").max() - umbral_pared * R < pl.col("x"))
                | (pl.col("y") < umbral_pared * R)
                | (pl.col("y").max() - umbral_pared * R < pl.col("y"))
            )
        )

    # Recopilación materializada de las métricas principales de la onda detectada
    df_res_onda = lf_onda.select(
        [pl.len().alias("cantidad"), pl.col("v^2").max().alias("v^2_max")]
    ).collect()

    return lf_onda, df_res_onda


# 1. ESTA PARTE VA AFUERA DE TU BUCLE FOR (Configuración estática)
def inicializar_pipeline_ovito(cutoff: float) -> Pipeline:
    """
    Inicializa la estructura estática de OVITO una sola vez para ahorrar memoria.
    """
    data = DataCollection()
    particles = data.create_particles()

    # Creamos la propiedad con un array vacío como punto de partida

    particles.create_property("Position", data=npempty((0, 3)))

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    return pipeline


# 2. ESTA ES TU FUNCIÓN EDITADA (Se llama adentro del bucle)
def analisis_ovito(lf: pl.LazyFrame, pipeline: Pipeline, altura: float) -> pl.DataFrame:
    """
    Integra la API de OVITO para realizar análisis topológico avanzado.
    Inyecta posiciones dinámicamente en un pipeline preconfigurado.

    Args:
        lf (pl.LazyFrame): Datos posicionales y de velocidad de las partículas.
        pipeline (Pipeline): Flujo de procesamiento de OVITO ya inicializado.
        altura (float): Umbral vertical de la cama para filtrar partículas.

    Returns:
        pl.DataFrame: DataFrame con las propiedades calculadas 'coordinacion' y 'cluster'.
    """
    # 1. Materializar los datos filtrados en memoria
    df = lf.filter((pl.col("z") > altura) & (pl.col("vz") > 0)).collect()

    # Control de seguridad: si no hay partículas eyectadas, devolvemos columnas vacías
    if df.shape[0] == 0:
        return df.with_columns(
            [
                pl.Series("coordinacion", [], dtype=pl.Float64),
                pl.Series("cluster", [], dtype=pl.UInt32),
            ]
        )

    # 2. Extraer matriz de coordenadas espaciales
    positions = df.select(["x", "y", "z"]).to_numpy()

    # 3a. Clonar los datos estáticos actuales para destrabar el modo "solo lectura"
    datos_mutables = pipeline.source.data.clone()

    # 3b. Inyectar las nuevas posiciones en tu copia (que ahora es mutable)
    datos_mutables.particles_.create_property("Position", data=positions)

    # 3c. Reasignar la copia actualizada como la nueva fuente del pipeline
    pipeline.source.data = datos_mutables

    # 4. Ejecución computacional del pipeline actualizado
    computed_data = pipeline.compute()

    # 5. Extracción de los resultados tabulados
    coordinacion = computed_data.particles["Coordination"][:]
    cluster = computed_data.particles["Cluster"][:]

    # 6. Incorporación y retorno de los atributos al DataFrame original
    return df.with_columns(
        [pl.Series("coordinacion", coordinacion), pl.Series("cluster", cluster)]
    )


def max_altura(ruta: str, R: float, umbral_techo=3) -> float:
    """
    Determina la altura máxima basal de la cama de partículas en el estado inicial de la simulación.

    Args:
        ruta (str): Directorio donde se encuentra el archivo inicial 'output.0.gz'.
        R (float): Radio físico de la partícula en metros [m].
        umbral_techo (int, opcional): Margen de seguridad añadido, expresado en número de radios. Por defecto es 3.

    Returns:
        float: Altura máxima calculada (límite superior de la cama) para usar como umbral de filtrado.

    Raises:
        ValueError: Si no se encuentran partículas de tipo 1 en la simulación.
    """
    ruta_archivo = str(Path(ruta) / "output.0.gz")
    lf0 = Lector(ruta_archivo, ["type", "z"])

    # Localiza la partícula tipo 1 (asumida como material del lecho) con mayor cota Z
    max_z_type1 = (
        lf0.filter(pl.col("type") == 1).select(pl.col("z").max()).collect().item()
    )

    if max_z_type1 is None:
        raise ValueError(
            "Error en la extracción: La altura máxima de la cama no puede ser nula (None). Verifique las IDs de las partículas."
        )

    # Añade el margen de tolerancia a la altura absoluta
    altura = umbral_techo * R + max_z_type1
    return altura


def verificar_analisis_previo(datos: str) -> bool:
    """
    Verifica si una simulación ya fue procesada buscando archivos previos en el directorio de resultados.

    Si encuentra coincidencias, desglosa el nombre del archivo para determinar
    el estado del análisis anterior (completo, parcial por onda, o sin erosión) y la fecha.
    Luego, despliega una alerta interactiva para que el usuario decida si desea sobreescribir.

    Args:
        datos (str): Cadena de texto con los parámetros de la simulación, utilizada
                     como identificador único (prefijo) en los nombres de archivo.

    Returns:
        bool: True si el análisis debe continuar (no existen datos previos o el
              usuario confirma la sobreescritura). False si el usuario decide abortar.
    """
    directorio_resultados = Path("./resultados")

    # Buscamos cualquier archivo que empiece con el ID de esta simulación
    archivos_previos = list(directorio_resultados.glob(f"{datos}_*"))
    if not archivos_previos:
        return True
    # Tomamos el primer archivo encontrado (asumiendo que solo hay uno por simulación)
    archivo_viejo = archivos_previos[0]
    nombre_viejo = archivo_viejo.name

    # Desglosamos el nombre para saber qué tipo de resultado fue y cuándo se hizo
    # Ej: "param1_param2_TOTAL_2026-03-08_18-30-00.parquet"

    if "_TOTAL_" in nombre_viejo:
        estado = "Análisis COMPLETO (Erosión detectada)"
        fecha_previa = nombre_viejo.split("_TOTAL_")[-1].replace(".parquet", "")

    elif "_ONDA_" in nombre_viejo:
        # En este caso el nombre tiene la iteración en el medio, ej: "..._ONDA_80_2026..."
        estado = "Onda detectada (Procesamiento parcial)"
        # Partimos por _ONDA_, tomamos la segunda mitad, y luego partimos por el primer guión bajo
        # para separar el número de iteración del timestamp
        partes = nombre_viejo.split("_ONDA_")[-1].replace(".parquet", "").split("_", 1)
        fecha_previa = partes[1] if len(partes) > 1 else "Fecha desconocida"

    elif "_NADA" in nombre_viejo:
        estado = "SIN EROSIÓN"
        fecha_previa = nombre_viejo.split("_NADA_")[-1].replace(".txt", "")

    else:
        estado = "Estado desconocido"
        fecha_previa = "Fecha no extraíble"

    limpiar_pantalla()
    print("--- ATENCIÓN: SIMULACIÓN YA ANALIZADA ---")
    print(f"Parámetros: {datos}")
    print(f"Resultado: {estado}")
    print(f"Fecha del análisis: {fecha_previa}\n")

    # Le damos al usuario la opción de abortar para no perder tiempo
    respuesta = input(
        "¿Deseas volver a procesar esta simulación sobrescribiendo los datos? (Y/N): "
    )
    if respuesta.lower() != "y":
        print("Análisis abortado por el usuario.")
        return False  # Termina la ejecución de la función aquí mismo
    return True  # Continua con el  análisis


def control_caja(limites_i: dict, limites_f: dict, archivos: list) -> bool:
    """
    Evalúa la consistencia volumétrica de la caja de simulación entre el estado inicial y final.

    Si detecta variaciones en los límites espaciales (por ejemplo, debido a condiciones de
    contorno dinámicas o ensambles NPT), interrumpe el flujo para consultar al usuario si
    desea mantener la caja estática original como volumen de control o utilizar bordes dinámicos.

    Args:
        limites_i (dict): Diccionario con los límites espaciales iniciales de la caja.
        limites_f (dict): Diccionario con los límites espaciales finales de la caja.
        archivos (list): Lista de nombres de los archivos de volcado, utilizada para
                         indicarle al usuario entre qué iteraciones ocurrió el cambio.

    Returns:
        bool: True si se decide utilizar los bordes fijos de la caja para el análisis de
              la onda. False si el usuario prefiere que el código adapte los bordes
              dinámicamente en base a la posición de las partículas.
    """
    # Gestión de divergencia en los límites del volumen de control (ej. condiciones NPT)
    if limites_f != limites_i:
        while True:
            seguir = input(
                f"""Los bordes de las caja cambiaron entre {archivos[0]} y {archivos[-1]}
    En {archivos[0]} son:
    {limites_i}
    En {archivos[-1]} son:
    {limites_f}
    ¿Deseás aun así usar los bordes de la caja para el cálculo? (Y/N): """
            )
            if seguir in "Nn":
                return False

            elif seguir in "Yy":
                return True
            else:
                limpiar_pantalla()
                print("(No se registró ninguna entrada válida. Por favor, usar Y o N)")
    else:
        return True


def preparar_archivos(ruta: str) -> tuple[list, int, int]:
    """
    Escanea el directorio de la simulación, filtra los archivos de volcado (dumps)
    y los ordena cronológicamente. Además, calcula el paso temporal entre volcados.

    Args:
        ruta (str): Ruta absoluta o relativa al directorio de la simulación.

    Returns:
        tuple:
            - list: Lista de nombres de archivos '.gz' ordenados cronológicamente.
            - int: Frecuencia de volcado de datos (timestep) extraída del segundo archivo.
            - int: Cantidad de archivos que hay en el directorio.
    """
    # Identificación y recopilación de los archivos de volcado de la simulación
    archivos = [d.name for d in Path(ruta).glob("output.*.gz")]

    # Ordenamiento alfanumérico estricto asegurando el orden cronológico de los pasos temporales
    archivos.sort(key=lambda x: "{0:0>19}".format(x).lower())

    # Extracción de la frecuencia de volcado de datos iterando los sufijos de archivo
    # Se añade un control de seguridad por si la simulación tiene un solo archivo
    if len(archivos) > 1:
        paso_str = archivos[1].replace("output.", "").replace(".gz", "")
        paso_int = int(paso_str)
    else:
        paso_int = 0  # Valor por defecto seguro si solo hay un frame

    return archivos, paso_int, len(archivos)


def cortar_analisis_onda(
    cantidad: int,
    bool_v2_max: bool,
    max_v2: float,
    i: int,
    file: str,
    datos: str,
    fecha_hora: str,
    lf_onda: pl.LazyFrame,
) -> bool:
    """
    Evalúa si se alcanzó el criterio físico crítico para considerar la propagación de una onda masiva.

    Si se cumple la condición de corte (por cantidad de partículas involucradas o por
    energía cinética máxima), imprime una alerta, serializa los datos de la onda a disco
    y devuelve una señal para interrumpir el escaneo secuencial (Fail Fast).

    Args:
        cantidad (int): Número de partículas detectadas como parte de la onda en el timestep actual.
        bool_v2_max (bool): Bandera que indica si alguna partícula superó el umbral crítico de velocidad.
        max_v2 (float): Valor absoluto de la velocidad cuadrática máxima registrada.
        i (int): Índice numérico de la iteración actual.
        file (str): Nombre del archivo de volcado actual (ej. 'output.80.gz').
        datos (str): Identificador base de los parámetros de la simulación.
        fecha_hora (str): Marca temporal para generar nombres de archivo únicos.
        lf_onda (pl.LazyFrame): Estructura de datos perezosa que contiene las partículas de la onda.

    Returns:
        bool: True si se detectó la onda y se guardó el archivo (señal de corte).
              False si el evento aún no alcanza la magnitud crítica.
    """
    # Criterio de impacto crítico: onda macroscópica o una partícula de altísima energía
    if cantidad > PARTICULAS_CRITICAS or bool_v2_max:
        print(f"""\nSe analizaron {i} archivos y en el archivo {file}:
                Se detectó una onda masiva de {cantidad} partículas
                Con una v² máxima absoluta de {max_v2}""")

        # Generación dinámica de la ruta de salida y materialización a formato columnar Parquet
        ruta_guardado = f"./resultados/{datos}_ONDA_{i}_{fecha_hora}.parquet"
        lf_onda.sink_parquet(ruta_guardado)

        return True  # Señal de abortar el escaneo
    else:
        return False  # Señal de continuar evaluando el siguiente timestep


def ingresar_radio() -> float:
    """
    Solicita de forma interactiva el radio de las partículas y valida la entrada del usuario.

    Implementa un bucle de seguridad que previene la interrupción del script (crashes)
    en caso de ingresos no numéricos. Adicionalmente, se encarga de estandarizar
    las unidades físicas, convirtiendo el valor ingresado de micrómetros [um] a metros [m].

    Args:
        None

    Returns:
        float: Valor del radio de los granos expresado en metros [m].
    """
    # --- BLINDAJE DEL INPUT DEL RADIO ---
    while True:
        try:
            entrada_r = input("Ingrese el valor de radio de los granos en [um]: ")
            R = float(entrada_r) * 1e-6
            break  # Si el número es válido, rompe el bucle y sigue
        except ValueError:
            limpiar_pantalla()
            print(
                "Error: Entrada inválida. Por favor, ingrese un número (ej: 10 o 15.5)\n"
            )
    return R


def data_analysis():
    """
    Orquestador principal que ejecuta el flujo completo de análisis del ensayo de erosión.

    Este pipeline realiza la selección de directorio, calibración inicial de parámetros,
    lectura secuencial iterativa de los *dumps*, detección de ondas de choque, análisis espacial
    mediante OVITO y exportación final a formato columnar Parquet.

    Args:
        None
    Returns:
        None
    """
    ruta, datos = generar_carpeta()
    fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not verificar_analisis_previo(datos):
        return

    R = ingresar_radio()
    limpiar_pantalla()

    # Genera una lista con los archivos ordenados, da la cantidad de steps que hay entre
    # archivos, cuantos archivos hay y una
    archivos, paso_int, LenTotal = preparar_archivos(ruta)

    # Verificación de consistencia volumétrica de la caja entre el estado inicial y final
    limites_i, limites_f = (
        extractor_de_borde(ruta, archivos[0]),
        extractor_de_borde(ruta, archivos[-1]),
    )

    usar_caja = control_caja(limites_i, limites_f, archivos)

    columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    # Calibración del nivel basal en el timestep 0
    altura, lista_resultados = max_altura(ruta, R), []

    pipeline_global = inicializar_pipeline_ovito(cutoff=R)

    # --- VARIABLES PARA EL CONTROL DE LA ONDA ---
    onda_detectada, paso_corte = False, LenTotal

    # === FASE 1: BÚSQUEDA RÁPIDA DE LA ONDA ===
    for i, file in enumerate(archivos):
        progress(i, LenTotal)

        ruta_archivo = str(Path(ruta) / file)
        archivo_lf = Lector(ruta_archivo, columnas)

        lf_onda, df_res_onda = detector_de_ondas(
            archivo_lf, R, altura, limites_f, usar_caja
        )

        cantidad = df_res_onda.item(0, "cantidad")
        max_v2 = df_res_onda.item(0, "v^2_max")
        bool_v2_max = (max_v2 >= UMBRAL_V2_MAX) if isinstance(max_v2, float) else False

        # Criterio de impacto crítico (Trigger de la onda)
        if cortar_analisis_onda(
            cantidad, bool_v2_max, max_v2, i, file, datos, fecha_hora, lf_onda
        ):
            paso_corte = i
            onda_detectada = True
            break

    progress(100, 100)
    print("")
    limpiar_pantalla()

    # === FASE 2: ANÁLISIS PESADO CON OVITO ===

    # Al hacer slicing hasta 'paso_corte', si la onda se detectó en i=80,
    # la lista tomará del 0 al 79 (justo el archivo anterior a la onda, como querías).
    # Si no hubo onda, paso_corte es igual a LenTotal y procesa todos.
    archivos_fase2 = archivos[:paso_corte]
    LenFase2 = len(archivos_fase2)
    parametros = datos.split("_")

    for i, file in enumerate(archivos_fase2):
        progress(i, LenFase2, 0)
        ruta_archivo = str(Path(ruta) / file)
        archivo_lf = Lector(ruta_archivo, columnas)

        df_procesados = analisis_ovito(archivo_lf, pipeline_global, altura)

        # Enriquecimiento del DataFrame usando kwargs (código más limpio)
        df_procesados_final = df_procesados.with_columns(
            ff=pl.lit(parametros[1]),
            np=pl.lit(parametros[3]),
            vel0=pl.lit(parametros[5]),
            angulo=pl.lit(parametros[-5]),
            posx=pl.lit(parametros[-3]),
            posy=pl.lit(parametros[-2]),
            posz=pl.lit(parametros[-1]),
            paso=pl.lit(paso_int),
            i=pl.lit(i),
            onda=pl.lit(onda_detectada),
        )

        lista_resultados.append(df_procesados_final)

    # Evaluación post-ciclo para escenarios sin evento erosivo
    if not lista_resultados:
        with open(f"./resultados/{datos}_NADA_{fecha_hora}.txt", "w") as archivo:
            archivo.write("No se obtuvo erosión significativa en esta simulación")
    else:
        df_temp = pl.concat(lista_resultados, how="vertical_relaxed")
        # Serialización de los resultados consolidados a formato Parquet
        df_temp.write_parquet(f"./resultados/{datos}_TOTAL_{fecha_hora}.parquet")

    if LenFase2 > 0:
        progress(100, 100, 0)

    print("\nSe terminó el análisis exitosamente")
    print("Datos guardados en: ./resultados")


# Punto de entrada de la ejecución
data_analysis()
