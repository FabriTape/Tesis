"""
Utilidades generales para el análisis de simulaciones LAMMPS.

Este módulo proporciona funciones auxiliares para la gestión de archivos,
interacción con el usuario, y operaciones comunes en el pipeline de análisis.
"""

import os
import re
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

# ==========================================
# PARÁMETROS FÍSICOS Y UMBRALES DE ANÁLISIS
# ==========================================
# Zona horaria de Argentina (UTC-3)
TZ_ARG = timezone(timedelta(hours=-3))

# Directorio de salida para los resultados
DIR_RESULTADOS = Path("./resultados")

# Radio por defecto en micrómetros (se convierte a metros)
RADIO_DEFAULT_UM = 0.76

# Umbrales de filtrado por defecto para el criterio de erosión
UMBRAL_VZ_MIN = 0.0  # Velocidad vertical mínima [m/s]
UMBRAL_ENERGIA_MIN = 0.0  # Energía cinética mínima para rama B [m²/s²]
UMBRAL_DESPLAZAMIENTO_LATERAL = 0.0  # Desplazamiento lateral mínimo [m]


def solicitar_entrada_segura(
    mensaje: str,
    tipo: str = "str",
    default: Any = None,
    permitir_back: bool = False,
    permitir_quit: bool = False,
    validador: callable | None = None,
) -> Any:
    """
    Solicita entrada al usuario con manejo seguro de comandos globales y valores por defecto.

    Implementa un sistema robusto de validación de entrada que soporta:
    - Comandos globales 'B' (retroceder) y 'Q' (salir)
    - Valores por defecto cuando el input está vacío
    - Validación de tipo (str, int, float)
    - Validadores personalizados opcionales

    Args:
        mensaje (str): El prompt que se mostrará al usuario.
        tipo (str): Tipo de dato esperado ('str', 'int', 'float'). Por defecto 'str'.
        default (Any): Valor por defecto si el usuario presiona Enter sin ingresar nada.
        permitir_back (bool): Si True, permite 'B' para retroceder al paso anterior.
        permitir_quit (bool): Si True, permite 'Q' para salir con confirmación.

    Returns:
        Any: El valor ingresado y validado, o None si se seleccionó 'B' o 'Q'.

    Raises:
        SystemExit: Si el usuario confirma la salida con 'Q'.
    """
    while True:
        try:
            entrada = input(mensaje).strip()

            # Manejo de comando Back
            if permitir_back and entrada.upper() == "B":
                return None

            # Manejo de comando Quit
            if permitir_quit and entrada.upper() == "Q":
                confirmacion = input("¿Está seguro que desea salir? (Y/N): ").strip()
                if confirmacion.upper() in ("Y", "YES", "S", "SI"):
                    print("Saliendo del programa. ¡Hasta luego!")
                    sys.exit(0)
                continue

            # Valor por defecto si la entrada está vacía
            if entrada == "" and default is not None:
                return default

            # Conversión de tipo
            if tipo == "int":
                return int(entrada)
            elif tipo == "float":
                return float(entrada)
            else:
                return entrada

        except ValueError:
            print(
                f"Error: Entrada inválida. Por favor, ingrese un valor de tipo {tipo}."
            )
        except KeyboardInterrupt:
            print("\n\nInterrupción detectada. Saliendo del programa...")
            sys.exit(0)


def limpiar_pantalla() -> None:
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


def buscar_carpeta() -> list[Path] | None:
    """
    Despliega una interfaz interactiva en la terminal para navegar y seleccionar un directorio
    o múltiples directorios que cumplan con un patrón.

    Args:
        None
    Returns:
        list[Path] | None: Lista de rutas absolutas de los directorios seleccionados,
                           o None si el usuario decide salir.
    """
    mensaje = """\nIngrese el numero de la carpeta a la que desea acceder
[-1] Seleccionar carpeta actual
[-2] Volver atrás
[-3] Procesar todas las carpetas 'erosion_bed_SI_*' en este directorio
[Q] Salir del programa: """

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
        carpetas_erosion = [p for p in items if p.name.startswith("erosion_bed_SI_")]

        print(f"\n--- Ubicación actual: {ruta} ---")
        print("Carpetas disponibles:")
        for i, carpeta in enumerate(items, start=1):
            print(f"[{i}] {carpeta.name}")

        # Informa sobre la presencia de archivos de datos procesables en la carpeta actual
        if datos_gz:
            N = len(datos_gz)
            ejemplo = datos_gz[0].name
            print(f"\nSe encontraron {N} archivos .gz (ej. {ejemplo})")

        if carpetas_erosion:
            print(f"Se encontraron {len(carpetas_erosion)} carpetas 'erosion_bed_SI_*'")

        try:
            entrada = input(mensaje).strip()

            # Comando de salida
            if entrada.upper() == "Q":
                confirmacion = input("¿Está seguro que desea salir? (Y/N): ").strip()
                if confirmacion.upper() in ("Y", "YES", "S", "SI"):
                    print("Saliendo del programa. ¡Hasta luego!")
                    return None

            eleccion = int(entrada)

            if eleccion == -1:
                # Confirma la selección de la carpeta actual
                limpiar_pantalla()
                return [ruta]

            elif eleccion == -2:
                # Navega al directorio padre (sube un nivel)
                ruta = ruta.parent
                limpiar_pantalla()

            elif eleccion == -3:
                if not carpetas_erosion:
                    limpiar_pantalla()
                    print("No hay carpetas 'erosion_bed_SI_*' en este directorio.")
                    continue
                limpiar_pantalla()
                return carpetas_erosion

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
        except KeyboardInterrupt:
            print("\n\nInterrupción detectada. Saliendo del programa...")
            return None


def extraer_metadatos(datos: str) -> dict[str, str]:
    """
    Extrae los parámetros físicos de la simulación a partir de la cadena identificadora (datos)
    utilizando expresiones regulares para mayor robustez ante cambios en el formato.

    Args:
        datos (str): Identificador de la simulación extraído del nombre de la carpeta.

    Returns:
        dict[str, str]: Diccionario con los parámetros 'ff', 'np', 'vel0', 'angulo',
                        'posx', 'posy', 'posz'.
    """
    patrones = {
        "ff": r"ff_([^_]+)",
        "np": r"np_([^_]+)",
        "vel0": r"vel_([^_]+)",
        "angulo": r"angle_([^_]+)",
        "move": r"move_([^_]+)_([^_]+)_([^_]+)",
    }

    metadatos: dict[str, str] = {}
    for clave, patron in patrones.items():
        match = re.search(patron, datos)
        if match:
            if clave == "move":
                metadatos["posx"] = match.group(1)
                metadatos["posy"] = match.group(2)
                metadatos["posz"] = match.group(3)
            else:
                metadatos[clave] = match.group(1)
        else:
            # Valor por defecto o aviso si no se encuentra el parámetro
            print(f"Advertencia: No se encontró el parámetro '{clave}' en '{datos}'")
            if clave == "move":
                metadatos.update({"posx": "0", "posy": "0", "posz": "0"})
            else:
                metadatos[clave] = "N/A"

    return metadatos


def progress(i: int, N: int, parte: int = 1) -> None:
    """
    Genera y actualiza una barra de progreso visual en la terminal.

    Args:
        i (int): Índice de la iteración actual.
        N (int): Número total de iteraciones esperadas.
        parte (int): Es el modo del mensaje si es por análisis de onda o de ovito:
                     1 = onda, 0 = coordinación
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


def verificar_analisis_previo(
    datos: str,
    umbrales_filtro: dict[str, float] | None = None,
    sobreescribir_todo: bool = False,
) -> tuple[bool, bool]:
    """
    Verifica si una simulación ya fue procesada buscando archivos previos en el directorio
    de resultados, considerando tanto los parámetros de simulación como los umbrales de filtro.

    Si encuentra coincidencias, desglosa el nombre del archivo para determinar
    el estado del análisis anterior (completo, parcial por onda, o sin erosión) y la fecha.
    Luego, despliega una alerta interactiva para que el usuario decida si desea sobreescribir.

    Args:
        datos (str): Cadena de texto con los parámetros de la simulación, utilizada
                     como identificador único (prefijo) en los nombres de archivo.
        umbrales_filtro (dict[str, float] | None): Diccionario con los umbrales de filtrado
            utilizados en el análisis actual para comparar con análisis previos.
        sobreescribir_todo (bool): Flag que indica si el usuario ya decidió sobrescribir
                                   todas las simulaciones previamente.

    Returns:
        tuple[bool, bool]: (continuar, sobreescribir_todo)
    """
    # Buscamos cualquier archivo que empiece con el ID de esta simulación
    archivos_previos = list(DIR_RESULTADOS.glob(f"{datos}_*"))
    if not archivos_previos:
        return True, sobreescribir_todo

    if sobreescribir_todo:
        for archivo in archivos_previos:
            try:
                archivo.unlink()
            except OSError as e:
                print(
                    f"Advertencia: no se pudo eliminar el archivo anterior {archivo.name}: {e}"
                )
        return True, sobreescribir_todo

    # Tomamos el primer archivo encontrado (asumiendo que solo hay uno por simulación)
    archivo_viejo = archivos_previos[0]
    nombre_viejo = archivo_viejo.name

    # Verificamos si los umbrales de filtro coinciden (si se proporcionaron)
    if umbrales_filtro:
        # Leemos los metadatos del archivo parquet anterior para comparar umbrales
        try:
            import polars as pl

            df_previo = pl.read_parquet(archivo_viejo, n_rows=1)
            umbrales_previos = {
                "umbral_vz_min": df_previo.get_column("umbral_vz_min").item()
                if "umbral_vz_min" in df_previo.columns
                else None,
                "umbral_energia_min": df_previo.get_column("umbral_energia_min").item()
                if "umbral_energia_min" in df_previo.columns
                else None,
            }

            # Comparamos umbrales (solo si ambos tienen los valores)
            if (
                umbrales_previos["umbral_vz_min"] is not None
                and umbrales_previos["umbral_energia_min"] is not None
            ):
                if (
                    abs(
                        umbrales_previos["umbral_vz_min"]
                        - umbrales_filtro["umbral_vz_min"]
                    )
                    > 1e-10
                    or abs(
                        umbrales_previos["umbral_energia_min"]
                        - umbrales_filtro["umbral_energia_min"]
                    )
                    > 1e-10
                ):
                    print(
                        "Nota: Los umbrales de filtro actuales difieren del análisis previo."
                    )
                    print(
                        f"Previo: vz_min={umbrales_previos['umbral_vz_min']}, "
                        f"energia_min={umbrales_previos['umbral_energia_min']}"
                    )
                    print(
                        f"Actual: vz_min={umbrales_filtro['umbral_vz_min']}, "
                        f"energia_min={umbrales_filtro['umbral_energia_min']}"
                    )
        except Exception:
            # Si no se puede leer el archivo, continuamos sin comparar umbrales
            pass

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
    while True:
        respuesta = (
            input(
                "¿Deseas volver a procesar esta simulación? (Y=Sí / N=No / A=Sí a todo): "
            )
            .strip()
            .upper()
        )
        if respuesta == "A":
            sobreescribir_todo = True
            break
        elif respuesta == "Y":
            break
        elif respuesta == "N":
            print("Análisis omitido para esta simulación.")
            return False, sobreescribir_todo
        else:
            print("Por favor, ingrese Y, N o A.")

    # Si decide re-procesar, eliminamos los archivos anteriores para no duplicar datos
    for archivo in archivos_previos:
        try:
            archivo.unlink()
        except OSError as e:
            print(
                f"Advertencia: no se pudo eliminar el archivo anterior {archivo.name}: {e}"
            )

    return True, sobreescribir_todo


def preparar_archivos(ruta: Path) -> tuple[list[str], int, int]:
    """
    Escanea el directorio de la simulación, filtra los archivos de volcado (dumps)
    y los ordena cronológicamente. Además, calcula el paso temporal entre volcados.

    Args:
        ruta (Path): Ruta absoluta o relativa al directorio de la simulación.

    Returns:
        tuple[list[str], int, int]:
            - list[str]: Lista de nombres de archivos '.gz' ordenados cronológicamente.
            - int: Frecuencia de volcado de datos (timestep) extraída del segundo archivo.
            - int: Cantidad de archivos que hay en el directorio.
    """
    # Identificación y recopilación de los archivos de volcado de la simulación
    archivos = [d.name for d in ruta.glob("output.*.gz")]

    # Ordenamiento alfanumérico estricto asegurando el orden cronológico de los pasos temporales
    archivos.sort(key=lambda x: "{0:0>19}".format(x).lower())

    # Extracción de la frecuencia de volcado de datos iterando los sufijos de archivo
    if len(archivos) > 1:
        paso_str = archivos[1].replace("output.", "").replace(".gz", "")
        paso_int = int(paso_str)
    else:
        paso_int = 0

    return archivos, paso_int, len(archivos)


def ingresar_radio() -> float:
    """
    Solicita de forma interactiva el radio de las partículas y valida la entrada del usuario.

    Implementa un bucle de seguridad que previene la interrupción del script (crashes)
    en caso de ingresos no numéricos. Adicionalmente, se encarga de estandarizar
    las unidades físicas, convirtiendo el valor ingresado de micrómetros [um] a metros [m].
    Si el input está vacío, asigna automáticamente 0.76 um (7.6e-7 m).

    Args:
        None

    Returns:
        float: Valor del radio de los granos expresado en metros [m].
    """
    # --- BLINDAJE DEL INPUT DEL RADIO ---
    while True:
        try:
            entrada_r = input(
                f"Ingrese el valor de radio de los granos en [um] (default: {RADIO_DEFAULT_UM}): "
            ).strip()

            # Valor por defecto si la entrada está vacía
            if entrada_r == "":
                R = RADIO_DEFAULT_UM * 1e-6
                print(f"Usando radio por defecto: {RADIO_DEFAULT_UM} um ({R} m)")
                break

            R = float(entrada_r) * 1e-6
            break  # Si el número es válido, rompe el bucle y sigue

        except ValueError:
            limpiar_pantalla()
            print(
                "Error: Entrada inválida. Por favor, ingrese un número (ej: 10 o 15.5)\n"
            )
    return R


def guardar_resultados(
    lista_resultados: list,
    datos: str,
    fecha_hora: str,
    umbrales_filtro: dict[str, float] | None = None,
) -> None:
    """
    Consolida y exporta los resultados del análisis cinemático y topológico a disco.

    Evalúa si la simulación produjo datos viables (erosión de la cama de partículas).
    Si no hubo eyección significativa, genera un registro de texto (flag) indicando el evento nulo.
    Si existen datos, concatena todos los pasos temporales procesados en un único
    DataFrame y lo serializa de forma eficiente en formato columnar (Parquet).

    Args:
        lista_resultados (list): Colección de DataFrames de Polars procesados durante la Fase 2.
        datos (str): Identificador base extraído del nombre de la carpeta (parámetros físicos).
        fecha_hora (str): Marca temporal del inicio de la ejecución para garantizar unicidad.
        umbrales_filtro (dict[str, float] | None): Diccionario con los umbrales de filtrado
            utilizados en el análisis. Se guardarán como columnas en el archivo parquet.

    Returns:
        None: La función realiza operaciones de escritura en disco (I/O) y no retorna valores.
    """
    import polars as pl

    if not lista_resultados:
        ruta_nada = DIR_RESULTADOS / f"{datos}_NADA_{fecha_hora}.txt"
        with open(ruta_nada, "w") as archivo:
            archivo.write("No se obtuvo erosión significativa en esta simulación")
    else:
        df_temp = pl.concat(lista_resultados, how="vertical_relaxed")

        # Inyectar umbrales de filtro como columnas si se proporcionaron
        if umbrales_filtro:
            df_temp = df_temp.with_columns(
                pl.lit(umbrales_filtro.get("umbral_vz_min", UMBRAL_VZ_MIN)).alias(
                    "umbral_vz_min"
                ),
                pl.lit(
                    umbrales_filtro.get("umbral_energia_min", UMBRAL_ENERGIA_MIN)
                ).alias("umbral_energia_min"),
                pl.lit(
                    umbrales_filtro.get(
                        "umbral_desplazamiento_lateral", UMBRAL_DESPLAZAMIENTO_LATERAL
                    )
                ).alias("umbral_desplazamiento_lateral"),
            )

        # Serialización de los resultados consolidados a formato Parquet
        ruta_total = DIR_RESULTADOS / f"{datos}_TOTAL_{fecha_hora}.parquet"
        df_temp.write_parquet(ruta_total)

    print("\nSe terminó el análisis exitosamente")
    print(f"Datos guardados en: {DIR_RESULTADOS}")


def configurar_umbrales_filtro(
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Solicita al usuario los umbrales de filtrado para el criterio de erosión compuesto.

    Permite configurar los parámetros que definen cuándo una partícula se considera
    erosionada, incluyendo velocidad vertical mínima, energía cinética mínima,
    y desplazamiento lateral. Los valores por defecto se obtienen de config.yaml
    si se proporciona.

    Args:
        config (dict[str, Any] | None): Configuración cargada desde config.yaml.
                                        Si es None, usa valores por defecto.

    Returns:
        dict[str, float]: Diccionario con los umbrales configurados:
            - umbral_vz_min: Velocidad vertical mínima [m/s]
            - umbral_energia_min: Energía cinética mínima [m²/s²]
            - umbral_desplazamiento_lateral: Desplazamiento lateral mínimo [m]
    """
    # Obtener valores por defecto de config o usar constantes
    if config:
        filtros = config.get("filtros", {})
        default_vz = filtros.get("umbral_vz_min", UMBRAL_VZ_MIN)
        default_energia = filtros.get("umbral_energia_min", UMBRAL_ENERGIA_MIN)
        default_desplazamiento = filtros.get(
            "umbral_desplazamiento_lateral", UMBRAL_DESPLAZAMIENTO_LATERAL
        )
    else:
        default_vz = UMBRAL_VZ_MIN
        default_energia = UMBRAL_ENERGIA_MIN
        default_desplazamiento = UMBRAL_DESPLAZAMIENTO_LATERAL

    print("\n--- Configuración de Umbrales de Filtro ---")
    print(
        "Estos umbrales definen el criterio compuesto para detectar partículas erosionadas."
    )
    print("Presione Enter para usar los valores por defecto.\n")

    # Umbral de velocidad vertical
    while True:
        try:
            entrada = input(
                f"Velocidad vertical mínima (vz_min) en [m/s] (default: {default_vz}): "
            ).strip()
            umbral_vz_min = float(entrada) if entrada else default_vz
            break
        except ValueError:
            print("Error: Ingrese un número válido.")

    # Umbral de energía cinética
    while True:
        try:
            entrada = input(
                f"Energía cinética mínima (v²_min) en [m²/s²] (default: {default_energia}): "
            ).strip()
            umbral_energia_min = float(entrada) if entrada else default_energia
            break
        except ValueError:
            print("Error: Ingrese un número válido.")

    # Umbral de desplazamiento lateral
    while True:
        try:
            entrada = input(
                f"Desplazamiento lateral mínimo [m] (default: {default_desplazamiento}): "
            ).strip()
            umbral_desplazamiento_lateral = (
                float(entrada) if entrada else default_desplazamiento
            )
            break
        except ValueError:
            print("Error: Ingrese un número válido.")

    return {
        "umbral_vz_min": umbral_vz_min,
        "umbral_energia_min": umbral_energia_min,
        "umbral_desplazamiento_lateral": umbral_desplazamiento_lateral,
    }
