#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
from gzip import open as opengz
from os import mkdir, listdir,path
import numpy as np
import pandas as pd
import numexpr as ne

#read_csv(ruta,skiprows=9,names=["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"],delimiter=" ")

def escrituraparquet(ruta):
    """La funcion escritura sirve para transformar archivos gz en parquet

    Args:
        ruta (str): Es la ruta donde se encuentran los archivos csv
    """

    ## Definimos la carpeta donde estan los outputs de la sim.
    Archivos = [archivo for archivo in listdir(ruta) if (archivo.startswith("output.") and ".gz" in archivo)]
    print("Iniciando conversión de archivos gz a parquet")
    print("000/100%|",20*"-","|",sep="",end="")
    for i,archivo in enumerate(Archivos,1):
        nombre = archivo.replace(".gz","")
        if  not path.isfile(f"{ruta}/datos/{nombre}.parquet"): ## Ver si no lo convertimos antes
            print("\r","{:03d}".format(int(i*100/len(Archivos)//1)),"/100%|",int((i*100/len(Archivos))//5)*"#",sep="",end="")
            ## Escirbir archivo ....
            with opengz(f'{ruta}/{nombre}.gz', 'rb') as csv_file:
                df_pd = pd.read_csv(
            csv_file,
	    sep = '\s+',
            skiprows=9,
            names=["id","type","x","y","z","vx","vy","vz","omegax","omegay","omegaz","c_1"])
        # 2. ESCRITURA EFICIENTE (Directo a Parquet)
        # Pandas usa PyArrow por debajo, es muy eficiente para escribir
                df_pd.to_parquet(f"{ruta}/datos/{nombre}.parquet", compression='snappy', index=False)
    print("conversión exitosa")

def checkparquet(ruta)->int:
    """Esta funcion cuenta la cantidad de archivos que se transformaron en parquet
    Args:
        ruta (str): es la ruta donde se encuentran los archivos output.
    Returns:
        int: Cantidad de carpetas que hay en el directorio de la simulación
        Paso (int): Cantidad de pasos que salta entre archivos
    """
    if not path.isdir(f"{ruta}/datos"): mkdir(f"{ruta}/datos")
    carpetas = [d for d in listdir(f"{ruta}/datos") if d.startswith("output.") and not ".gz" in d]
    if not carpetas:
        print("No hay datos registrados todavía.")
        return 0,0
    paso = sorted([float(i.replace("output.","").replace(".parquet", "")) for i in carpetas]) ## El primer elemnto es el 0, el segundo es el paso
    return len(carpetas), paso[1]


def data_analysis (ruta,vel,npa):
    threshold = 0.004
    LenTotal, paso = checkparquet(ruta)
    if LenTotal==0:
        escrituraparquet(ruta)
    LenTotal, paso = checkparquet(ruta)
    # Pre-allocating lists is slightly faster, but lists are okay for <100k iters
    Data_collect = {
        "K": [],
        "Max": [],
        "Min":[],
        "Vcm": [],
        "N": 0,
        "Eyección": [-1],
        "Eyectados_K":[-1],
	    "Vel_temp":[[],[]]}

    # Variables constantes fuera del bucle
    df0 = pd.read_parquet(f"{ruta}/datos/output.0.parquet",  columns=["type","z","vx","vy","vz"])
    print(f"Iniciando análisis de {LenTotal} archivos...")
    # Filtramos donde type == 1 y buscamos el maximo de z
    altura = 1.02*np.max(df0["z"].to_numpy())
    tipo = df0["type"].to_numpy()
    # Lógica del proyectil (type == 2)
    N = len(tipo)
    mask = (tipo == 2)
    # Definimos N inicial y v0_actual basado en tu lógica
    if not np.any(mask):
        v0_actual_init = vel
        N_base = N + npa
    else:
        v0_actual_init = 0
        N_base = N
    Data_collect["N"] = N_base # Guardamos el N de este paso
    # --- FASE 2: EL BUCLE DE PROCESAMIENTO ---
    print("Iniciando análisis de datos")
    print("000/100%|",20*"-","|",sep="",end="")

    for i in range(LenTotal):
        print("\r","{:03d}".format(int((i+1)*100/LenTotal//1)),"/100%|",int(((i+1)*100/LenTotal)//5)*"#",sep="",end="")
	# 1. Lectura de los datos con las columnas ["id","type","z","vx","vy","vz"]
        Datos = pd.read_parquet(f"{ruta}/datos/output.{int(paso*i)}.parquet",  columns=["id","type","z","vx","vy","vz"])
        v0_actual = v0_actual_init if i==0 else 0
        N = N_base
        vx = Datos["vx"].to_numpy()
        vy = Datos["vy"].to_numpy()
        vz = Datos["vz"].to_numpy()
        z_pos = Datos["z"].to_numpy()
        Vnorm2 = ne.evaluate("vx**2 + vy**2 + vz**2")
        # Calculamos la velocidad del CM
        vcm_x = np.sum(vx) / N
        vcm_y = np.sum(vy) / N
        vcm_z = (np.sum(vz) + v0_actual) / N
        Data_collect["Vcm"].append([vcm_x, vcm_y, vcm_z])
        # Energía Cinética
        Data_collect["K"].append(0.5 * np.sum(Vnorm2))
	# Filtro por altura para hallar erosión
        mask_altura = z_pos > altura
	# Filtro por velocidad
        if np.any(mask_altura):
            vz_candidatos = vz[mask_altura]
            mask_velocidad = vz_candidatos > threshold
	    # Resultado final:
            count_eyeccion = np.sum(mask_velocidad)
    	    # Energía por erosión
            Vnorm2_candidatos = Vnorm2[mask_altura]
            K_eyectados = 0.5 * np.sum(Vnorm2_candidatos[mask_velocidad])
            Data_collect["Vel_temp"][i%2] = Vnorm2_candidatos
           #Max / Min
            Limite = np.array(Vnorm2[~mask_altura])
            Data_collect["Max"].append(Limite[Limite<threshold].max())
            Data_collect["Min"].append(Limite.min())
        else:
            count_eyeccion = 0
            K_eyectados = 0
            	# Max / Min
            Data_collect["Max"].append(np.max(Vnorm2))
            Data_collect["Min"].append(np.min(Vnorm2))
        if i == 0:
    	    Data_collect["Eyección"][0] = count_eyeccion
    	    Data_collect["Eyectados_K"][0] = K_eyectados
        else:
            if count_eyeccion - Data_collect["Eyección"][-1] >= 0:
                Data_collect["Eyección"].append(count_eyeccion)
                Data_collect["Eyectados_K"].append(K_eyectados)
            else:
                congelados = list(set(Data_collect["Vel_temp"][(i+1)%2])-set(Data_collect["Vel_temp"][i%2]))
                Data_collect["Vel_temp"][i%2] = [x for x in Data_collect["Vel_temp"][i%2] if x != 0.0]
                Data_collect["Vel_temp"][i%2].append(*congelados)
                Data_collect["Eyección"].append(count_eyeccion+len(congelados))
                Data_collect["Eyectados_K"].append(0.5*np.sum(Data_collect["Vel_temp"][i%2]))
                Data_collect["K"][-1] += 0.5*np.sum(congelados)

    return Data_collect



    # for i in range(LenTotal):
    #    

    #     # 1. Carga Eficiente
    #     # Usamos read_parquet porque vamos a procesar el archivo entero en este paso.
    #     # scan_parquet es mejor si fueras a filtrar filas ANTES de cargar.
    #     df = pl.read_parquet(f"{ruta}/datos/output.{int(paso*i)}.parquet")
        
    #     # Ajuste de variables dependientes del frame 0
    #     current_N = N_base if i == 0 else df.height 
    #     current_v0 = v0_actual_init if i == 0 else 0
        
    #     # 2. DEFINICIÓN DE EXPRESIONES (La magia de Polars)
    #     # No calculamos nada todavía, solo definimos la fórmula.
        
    #     # Vnorm2: vx^2 + vy^2 + vz^2
    #     expr_vnorm2 = (pl.col("vx")**2 + pl.col("vy")**2 + pl.col("vz")**2)
        
    #     # Filtro de eyección: (z > altura) AND (vz > threshold)
    #     expr_eyeccion = (pl.col("z") > altura) & (pl.col("vz") > threshold)

    #     # 3. EJECUCIÓN AGRUPADA
    #     # Hacemos una sola consulta que devuelve una fila con todos los agregados.
    #     # Esto es MUCHO más rápido que calcular arrays numpy por separado.
    #     metrics = df.select([
    #         # Velocidad CM
    #         pl.col("vx").sum().alias("sum_vx"),
    #         pl.col("vy").sum().alias("sum_vy"),
    #         pl.col("vz").sum().alias("sum_vz"),
            
    #         # Energía Cinética Total (0.5 * sum(v^2))
    #         (0.5 * expr_vnorm2.sum()).alias("K"),
            
    #         # Eyección: count y energía
    #         expr_eyeccion.sum().alias("count_eyeccion"), 
    #         # Sumar booleanos cuenta los True
    #         (0.5 * expr_vnorm2.filter(expr_eyeccion).sum()).alias("K_eyeccion"),
    #         # Max/Min de energía (filtrando los NO eyectados según tu lógica original: ~filtro)
    #         expr_vnorm2.filter(~expr_eyeccion).max().alias("max_v2_confined"),
    #         expr_vnorm2.filter(~expr_eyeccion).min().alias("min_v2_confined")
    #     ])

    #     # 4. EXTRACCIÓN DE RESULTADOS (Row -> Python Dict/List)
    #     row = metrics.row(0) # Tupla con los resultados
    #     # El orden depende del select de arriba:
    #     # 0:sum_vx, 1:sum_vy, 2:sum_vz, 3:K, 4:count_eye, 5:K_eye, 6:max, 7:min
        
    #     # Calculamos Vcm final usando los escalares
    #     vcm_x = row[0] / current_N
    #     vcm_y = row[1] / current_N
    #     vcm_z = (row[2] + current_v0) / current_N
        
    #     # Guardamos en listas
    #     results["Vcm"].append([vcm_x, vcm_y, vcm_z])
    #     results["K"].append(row[3])
    #     results["Eyección"].append(row[4])
    #     results["Eyectados_K"].append(row[5])
    #     results["Max"].append(row[6])
    #     results["Min"].append(row[7])
    
        
    # return results
