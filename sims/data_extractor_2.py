#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
from gzip import open as opengz
from os import mkdir, listdir, path
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# NOTA: Se eliminó la línea problemática pl.Config.set_global_string_cache(False)

def escrituraparquet(ruta):
    """Transformar archivos gz en parquet usando Polars"""
    Archivos = [archivo for archivo in listdir(ruta) 
                if (archivo.startswith("output.") and ".gz" in archivo)]
    
    if not Archivos:
        return
    
    print("Iniciando conversión de archivos gz a parquet")
    print("000/100%|", 20*"-", "|", sep="", end="")
    
    # Crear directorio si no existe
    os_datos = f"{ruta}/datos"
    if not path.isdir(os_datos):
        mkdir(os_datos)
    
    for i, archivo in enumerate(Archivos, 1):
        nombre = archivo.replace(".gz", "")
        output_path = f"{os_datos}/{nombre}.parquet"
        
        if not path.isfile(output_path):
            print("\r", "{:03d}".format(int(i*100/len(Archivos)//1)), 
                  "/100%|", int((i*100/len(Archivos))//5)*"#", sep="", end="")
            
            # Leer directamente con Polars
            df_pl = pl.read_csv(
                f'{ruta}/{nombre}.gz',
                separator='\\s+',
                skip_rows=9,
                has_header=False,
                new_columns=["id", "type", "x", "y", "z", "vx", "vy", "vz", 
                             "omegax", "omegay", "omegaz", "c_1"],
                infer_schema_length=10000,
                low_memory=True
            )
            
            # Escribir con compresión zstd (más rápido que snappy)
            df_pl.write_parquet(output_path, compression='zstd')
    
    print("\nConversión exitosa")

def checkparquet(ruta):
    """Contar archivos parquet existentes"""
    datos_dir = f"{ruta}/datos"
    if not path.isdir(datos_dir):
        mkdir(datos_dir)
        return 0, 0
    
    archivos = [d for d in listdir(datos_dir) 
                if d.startswith("output.") and d.endswith(".parquet")]
    
    if not archivos:
        return 0, 0
    
    # Extraer números de paso de manera eficiente
    pasos = []
    for f in archivos:
        try:
            num = float(f.replace("output.", "").replace(".parquet", ""))
            pasos.append(num)
        except:
            continue
    
    pasos.sort()
    return len(pasos), pasos[1] if len(pasos) > 1 else 0

def _procesar_lote(args):
    """Procesar un lote de archivos en paralelo - VERSIÓN OPTIMIZADA y EAGER"""
    archivos_info, altura, threshold, N_base, v0_actual_init = args
    resultados = []
    
    for i, file_path in archivos_info:
        # 1. LECTURA DIRECTA EAGER (sin scan_parquet)
        # Asegurarse de que no se use scan_parquet
        df = pl.read_parquet(
            file_path,
            columns=["id", "type", "z", "vx", "vy", "vz"]
        )
        
        # Determinar v0_actual
        v0_actual = v0_actual_init if i == 0 else 0
        
        # 2. CÁLCULOS EN UNA SOLA PASADA CON POLARS NATIVO
        # Usar expresiones y agregaciones directamente, sin lazy
        # Nota: read_parquet ya devuelve un DataFrame, no un LazyFrame
        result_row = df.select([
            pl.col("vx").sum().alias("sum_vx"),
            pl.col("vy").sum().alias("sum_vy"),
            pl.col("vz").sum().alias("sum_vz"),
            (0.5 * (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2).sum()).alias("K_total"),
            ((pl.col("z") > altura) & (pl.col("vz") > threshold)).sum().alias("count_eyeccion"),
            (0.5 * (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2)
                .filter((pl.col("z") > altura) & (pl.col("vz") > threshold))
                .sum()
            ).alias("K_eyectados"),
            (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2)
                .filter(pl.col("z") <= altura)
                .max()
                .alias("max_v2_confined"),
            (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2)
                .filter(pl.col("z") <= altura)
                .min()
                .alias("min_v2_confined")
        ]).row(0)
        
        # 3. EXTRAER RESULTADOS
        sum_vx, sum_vy, sum_vz, K_total, count_eyeccion, K_eyectados, max_val, min_val = result_row
        
        # 4. CALCULAR Vcm
        vcm_x = sum_vx / N_base
        vcm_y = sum_vy / N_base
        vcm_z = (sum_vz + v0_actual) / N_base
        
        # 5. OBTENER vnorm2_altos (solo para partículas altas, ≤100 elementos)
        # Usar el mismo DataFrame df, no volver a leer
        if count_eyeccion > 0:
            vnorm2_altos = df.filter(pl.col("z") > altura).select(
                pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2
            ).to_series().to_list()
        else:
            vnorm2_altos = []
        
        resultados.append({
            'i': i,
            'vcm': [vcm_x, vcm_y, vcm_z],
            'K_total': K_total,
            'count_eyeccion': count_eyeccion,
            'K_eyectados': K_eyectados,
            'max_val': max_val,
            'min_val': min_val,
            'vnorm2_altos': vnorm2_altos
        })
    
    return resultados
def data_analysis(ruta, vel, npa):
    threshold = 0.004
    LenTotal, paso = checkparquet(ruta)
    
    if LenTotal == 0:
        escrituraparquet(ruta)
        LenTotal, paso = checkparquet(ruta)
    
    print(f"Analizando {LenTotal} archivos...")
    
    # Leer solo el primer archivo para obtener metadatos
    df0 = pl.read_parquet(
        f"{ruta}/datos/output.0.parquet",
        columns=["type", "z"]
    )
    
    # Calcular altura
    altura = 1.02 * df0.filter(pl.col("type") == 1)["z"].max()
    
    # Determinar N_base
    N = df0.height
    tipo_arr = df0["type"].to_numpy()
    
    if not np.any(tipo_arr == 2):
        v0_actual_init = vel
        N_base = N + npa
    else:
        v0_actual_init = 0
        N_base = N
    
    # Pre-allocating results
    Data_collect = {
        "K": [],
        "Max": [],
        "Min": [],
        "Vcm": [],
        "N": N_base,
        "Eyección": [-1],
        "Eyectados_K": [-1],
        "Vel_temp": [[], []]
    }
    
    # Preparar lista de archivos para procesamiento paralelo
    archivos_info = []
    for i in range(LenTotal):
        file_path = f"{ruta}/datos/output.{int(paso*i)}.parquet"
        archivos_info.append((i, file_path))
    
    # Dividir en lotes para procesamiento paralelo
    # AJUSTADO: 6 workers para evitar saturación de I/O
    num_workers = min(6, LenTotal)
    lote_size = max(1, LenTotal // num_workers)
    lotes = []
    
    for start in range(0, LenTotal, lote_size):
        end = min(start + lote_size, LenTotal)
        lote = archivos_info[start:end]
        lotes.append((lote, altura, threshold, N_base, v0_actual_init))
    
    print(f"Procesando con {num_workers} workers...")
    print("000/100%|", 20*"-", "|", sep="", end="")
    
    # Procesar lotes en paralelo
    resultados_totales = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_procesar_lote, lote): i 
                   for i, lote in enumerate(lotes)}
        
        completados = 0
        for future in as_completed(futures):
            resultados_totales.extend(future.result())
            completados += 1
            print("\r", "{:03d}".format(int(completados*100/len(lotes)//1)), 
                  "/100%|", int((completados*100/len(lotes))//5)*"#", sep="", end="")
    
    print("\nOrdenando y procesando resultados...")
    
    # Ordenar por índice y procesar acumulación
    resultados_totales.sort(key=lambda x: x['i'])
    
    for idx, res in enumerate(resultados_totales):
        i = res['i']
        
        Data_collect["Vcm"].append(res['vcm'])
        Data_collect["K"].append(res['K_total'])
        Data_collect["Max"].append(res['max_val'])
        Data_collect["Min"].append(res['min_val'])
        
        count_eyeccion = res['count_eyeccion']
        K_eyectados = res['K_eyectados']
        vnorm2_altos = res['vnorm2_altos']
        
        # Lógica de acumulación (optimizada para ≤100 partículas)
        if i == 0:
            Data_collect["Eyección"][0] = count_eyeccion
            Data_collect["Eyectados_K"][0] = K_eyectados
            Data_collect["Vel_temp"][0] = vnorm2_altos
        else:
            if count_eyeccion - Data_collect["Eyección"][-1] >= 0:
                Data_collect["Eyección"].append(count_eyeccion)
                Data_collect["Eyectados_K"].append(K_eyectados)
                Data_collect["Vel_temp"][i % 2] = vnorm2_altos
            else:
                # Manejar partículas "congeladas" - optimizado para pocos elementos
                temp1 = set(Data_collect["Vel_temp"][(i + 1) % 2])
                temp2 = set(vnorm2_altos)
                congelados = list(temp1 - temp2)
                
                current_vel_temp = [x for x in vnorm2_altos if x != 0.0]
                current_vel_temp.extend(congelados)
                Data_collect["Vel_temp"][i % 2] = current_vel_temp
                
                Data_collect["Eyección"].append(count_eyeccion + len(congelados))
                Data_collect["Eyectados_K"].append(0.5 * sum(current_vel_temp))
                
                # Ajustar energía total
                if len(congelados) > 0:
                    Data_collect["K"][-1] += 0.5 * sum(congelados)
    
    return Data_collect