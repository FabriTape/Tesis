#!/home/tutturuu/spyder-env/bin python34
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import polars as pl


from ovito.data import DataCollection, SimulationCell
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CoordinationAnalysisModifier
from ovito.modifiers import ClusterAnalysisModifier

# Definimos UTC explícitamente para compatibilidad
UTC = timezone.utc

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
    """
    Configura la lectura diferida (Lazy) de un archivo .gz de LAMMPS.
    
    Args:
        archivo_gz (str): Ruta al archivo comprimido.
        datos (list[str]): Lista de columnas a seleccionar.
        
    Returns:
        (pl.LazyFrame) : Selección de los datos originales en un LazyFrame con
                        las columnas especificadas en Args: datos.
    """
    nombres_columnas = ["id", "type", 
                         "x",  "y",  "z", 
                        "vx", "vy", "vz", 
                        "wx", "wy", "wz"]
    
    # scan_csv es lazy, no lee nada hasta que se llama a collect()
    return pl.scan_csv(
        archivo_gz, 
        skip_rows=9,
        has_header=False,
        separator=" ", 
        truncate_ragged_lines=True, # Maneja espacios extra al final
        new_columns=nombres_columnas,
        infer_schema_length=10000,
        schema_overrides={
            "type": pl.UInt8,
            "id": pl.UInt64
        }
    ).select(datos)

def progress(i, N):
    """
    Muestra una barra de progreso visual en la terminal.
    
    Args:
        i (int): Iteración actual.
        N (int): Total de iteraciones.
    Returns:
        None
    """
    pct = i * 100 / N
    bar = int(pct / 5) * "_" + "@ï"
    print(f"\r{int(pct):03d}/100%[{bar}", end="")

def detector_de_ondas(lf, R, altura):
    """
    Filtra partículas que cumplen con criterios de alta energía y posición 
    que llega al borde (onda de choque).
    
    Args:
        lf (pl.LazyFrame): Datos de entrada.
        R (float): Radio de la partícula.
        altura (float): Altura de referencia de la cama de partículas.
        
    Returns:
        tuple:
            - lf_onda (pl.LazyFrame): Subset de datos que entran dentro del
                                      filtrado.
            - df_res_onda (pl.DataFrame): Parámetros de la onda, cantidad de 
                                        particulas que la componen y velocidad
                                        máxima en la onda.
    """
    # 1. Calculamos columnas derivadas (Lazy)
    lf_onda = (
        lf
        .with_columns(
            (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2).alias("v^2")
        )
        .filter(
            (pl.col("z") < altura - 4 * R) &
            (pl.col("v^2") > 4e-4) &
            (
                (pl.col("x") < 10 * R) | 
                (pl.col("x") > pl.col("x").max() - 10 * R) |
                (pl.col("y") < 10 * R) | 
                (pl.col("y") > pl.col("y").max() - 10 * R)
            )
        )
    )
    
    # 2. Collect pequeño solo para las estadísticas
    # Si 'lf' ya viene de memoria, esto es instantáneo.
    df_res_onda = (
        lf_onda
        .select([
            pl.len().alias("cantidad"),
            pl.col("v^2").max().alias("v^2_max")
        ])
        .collect()
    )
    return lf_onda, df_res_onda

def analisis_ovito(lf: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    """
    Realiza análisis estructural (Coordinación y Clusters) de las
    partículas eyectadas usando OVITO.
    
    Args:
        lf (pl.LazyFrame): Datos de las partículas eyectadas.
        cutoff (float): Distancia de corte para definir vecinos.
        
    Returns:
        pl.LazyFrame: DataFrame original con columnas nuevas 'coordinacion' 
                      y 'cluster'.
    """
    # 1 Materializamos los datos para OVITO (Si lf viene de memoria, es rápido)
    df = lf.collect()
    
    if df.height == 0:
        # Retornamos estructura vacía si no hay partículas
        return df.with_columns([
            pl.Series("coordinacion", [], dtype=pl.Int32),
            pl.Series("cluster", [], dtype=pl.Int32)
        ]).lazy()

    # 2. OVITO requiere numpy
    positions = df.select(["x", "y", "z"]).to_numpy()
    
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property('Position', data=positions)
    
    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  
    data.objects.append(cell)
    
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))
    
    computed_data = pipeline.compute()
    
    coordinacion = computed_data.particles['Coordination'][:]
    cluster = computed_data.particles['Cluster'][:]
    
    return df.with_columns([
        pl.Series("coordinacion", coordinacion),
        pl.Series("cluster", cluster)
    ]).lazy()

def data_analysis():
    """
    Función principal de orquestación del análisis.
    
    Pasos:
    1. Solicita configuración al usuario (carpeta, radio).
    2. Calcula parámetros iniciales (altura de la cama).
    3. Itera sobre los archivos .gz 
    4. Ejecuta detección de ondas (Ver función detector_de_ondas)
    5. Guarda resultados en formato Parquet.
    Args:
        None
    Returns:
        None
    """
    ruta, base = generar_carpeta()
    
    try:
        entrada_R = input("Ingrese el valor de radio de los granos en [m]: ")
        R = float(entrada_R)
    except ValueError:
        print("Valor inválido, usando por defecto 1e-6")
        R = 1e-6
        
    limpiar_pantalla()
    
    # Ordenar archivos correctamente
    archivos = [f for f in os.listdir(ruta) if "output." in f 
                and f.endswith(".gz")]
    # Extraemos el número entre 'output.' y '.gz' para ordenar numéricamente
    archivos.sort(key=lambda x: int(x.split('.')[1]))

    LenTotal = len(archivos)
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
        
        path_archivo = os.path.join(ruta, file)
        
        # --- OPTIMIZACIÓN: LEER UNA SOLA VEZ A MEMORIA ---
        # Leemos el archivo del disco AHORA y lo guardamos en RAM.
        # Las funciones subsiguientes trabajarán sobre 'lf_mem' sin tocar 
        # el disco.
        try:
            lf_mem = Lector(path_archivo,columnas)
            
        except Exception as e:
        
            print(f"\nError leyendo {file}: {e}")
            continue

        # 1. Detectar Onda (Usa datos de memoria -> Rápido)
        lf_onda, df_res_onda = detector_de_ondas(lf_mem, R, altura)
        
        cantidad = df_res_onda.item(0, "cantidad")
        max_v2 = df_res_onda.item(0, "v^2_max")
        
        # Validación segura de None
        bool_v2_max = False
        if max_v2 is not None:
             bool_v2_max = (max_v2 >= 4e-2)

        if cantidad > 3000 or bool_v2_max:
            print(f"\n\n[!] Detenido en {file} (Iteración {i})")
            print(f"\tOnda detectada: {cantidad} partículas | V²max: {max_v2}")
            
            ruta_guardado = os.path.join(base, f"onda.{i}.parquet")
            lf_onda.sink_parquet(ruta_guardado)
            break
        
        else:
            # 2. Análisis OVITO (Usa datos de memoria -> Rápido)
            # Como lf_mem ya está cargado, analisis_ovito no relee el disco.
            lf_eyectados = (
                lf_mem
                .filter(pl.col("z") > altura)
                .filter(pl.col("vz") > 0)
            )
            
            # Ejecutamos análisis
            lf_procesados = analisis_ovito(lf_eyectados, R)
            
            # Guardamos resultado
            ruta_guardado = os.path.join(base, f"salida.{i}.parquet")
            lf_procesados.sink_parquet(ruta_guardado)
            
            # Chequeo de proyectil (sobre el resultado en memoria)
            conteo_proyectil = lf_procesados.filter(pl.col("type") == 2).select(pl.len()).collect().item()
            
            if conteo_proyectil == 0:
                # Si no está en eyectados, lo buscamos en el frame completo (que ya está en memoria)
                ruta_proy = os.path.join(base, f"proyectil_hundido.{i}.parquet")
                lf_mem.filter(pl.col("type") == 2).sink_parquet(ruta_proy)

    progress(LenTotal, LenTotal)
    print(f"\n\nAnálisis finalizado. Datos en: {base}")

if __name__ == "__main__":
    data_analysis()