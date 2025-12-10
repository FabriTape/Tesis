#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
from gzip import open as opengz
from os import mkdir, listdir, path
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuración de Polars para máximo rendimiento
pl.Config.set_fmt_str_lengths(1000)

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
            
            # Leer directamente con Polars - MUCHO más rápido
            df_pl = pl.read_csv(
                f'{ruta}/{nombre}.gz',
                separator='\\s+',
                skip_rows=9,
                has_header=False,
                new_columns=["id", "type", "x", "y", "z", "vx", "vy", "vz", 
                             "omegax", "omegay", "omegaz", "c_1"],
                infer_schema_length=10000,  # Inferir schema rápido
                low_memory=True  # Modo bajo consumo de memoria
            )
            
            # Escribir con compresión
            df_pl.write_parquet(output_path, compression='zstd')  # zstd es más rápido que snappy
    
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
    """Procesar un lote de archivos en paralelo"""
    archivos_info, altura, threshold, N_base, v0_actual_init = args
    resultados = []
    
    for i, file_path in archivos_info:
        # Leer SOLO las columnas necesarias
        df = pl.read_parquet(
            file_path,
            columns=["id", "type", "z", "vx", "vy", "vz"]
        )
        
        # Determinar v0_actual
        v0_actual = v0_actual_init if i == 0 else 0
        
        # Calcular todas las métricas en UNA SOLA pasada
        metrics = df.select([
            # Sumas para Vcm
            pl.col("vx").sum().alias("sum_vx"),
            pl.col("vy").sum().alias("sum_vy"),
            pl.col("vz").sum().alias("sum_vz"),
            
            # Vnorm2 como expresión
            (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2).alias("vnorm2"),
            
            # Flags para filtros
            (pl.col("z") > altura).alias("altura_flag"),
            (pl.col("vz") > threshold).alias("velocidad_flag")
        ])
        
        # Extraer arrays numpy para cálculos rápidos
        vnorm2_arr = metrics["vnorm2"].to_numpy()
        altura_flag = metrics["altura_flag"].to_numpy()
        velocidad_flag = metrics["velocidad_flag"].to_numpy()
        
        # Filtro combinado para eyección
        eyeccion_mask = altura_flag & velocidad_flag
        
        # Cálculos rápidos con numpy
        count_eyeccion = np.sum(eyeccion_mask)
        
        # Energías
        K_total = 0.5 * np.sum(vnorm2_arr)
        K_eyectados = 0.5 * np.sum(vnorm2_arr[eyeccion_mask]) if count_eyeccion > 0 else 0
        
        # Máximo y mínimo para partículas no eyectadas
        confinadas_mask = ~altura_flag
        vnorm2_confinadas = vnorm2_arr[confinadas_mask]
        
        if len(vnorm2_confinadas) > 0:
            # Filtrar por threshold
            vnorm2_filtrado = vnorm2_confinadas[vnorm2_confinadas < threshold]
            max_val = vnorm2_filtrado.max() if len(vnorm2_filtrado) > 0 else 0
            min_val = vnorm2_confinadas.min()
        else:
            max_val = vnorm2_arr.max()
            min_val = vnorm2_arr.min()
        
        # Vcm
        sum_vx = metrics["sum_vx"][0]
        sum_vy = metrics["sum_vy"][0]
        sum_vz = metrics["sum_vz"][0]
        
        vcm_x = sum_vx / N_base
        vcm_y = sum_vy / N_base
        vcm_z = (sum_vz + v0_actual) / N_base
        
        # Datos para Vel_temp (solo si hay partículas altas)
        if np.any(altura_flag):
            vnorm2_altos = vnorm2_arr[altura_flag].tolist()
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
    num_workers = min(8, LenTotal)  # Usar hasta 8 workers
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
        
        # Lógica de acumulación
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
                # Manejar partículas "congeladas"
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
