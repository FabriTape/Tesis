#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
from gzip import open as opengz
import os
import polars as pl
from Lector import Lector
from ovito.data import DataCollection
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CoordinationAnalysisModifier
from ovito.modifiers import ClusterAnalysisModifier
from ovito.data import SimulationCell

#read_csv(ruta,skiprows=9,names=["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"],delimiter=" ")
'''---> Esta funcion ya no se usa y será removida en futuras versiones del programa <---'''
def escrituraparquet(ruta):
    """La funcion escritura sirve para transformar archivos gz en parquet

    Args:
        ruta (str): Es la ruta donde se encuentran los archivos csv
    """

    ## Definimos la carpeta donde estan los outputs de la sim.
    
    Archivos = [archivo for archivo in os.listdir(ruta) if (archivo.startswith("output.") and ".gz" in archivo)]
    
    print("Iniciando conversión de archivos gz a parquet")
    print("000/100%|",20*"-","|",sep="",end="")
    
    for i,archivo in enumerate(Archivos,1):
        
        nombre = archivo.replace(".gz","")
        
        if  not os.path.isfile(f"{ruta}/datos/{nombre}.parquet"): ## Ver si no lo convertimos antes
        
            print("\r","{:03d}".format(int(i*100/len(Archivos)//1)),"/100%|",int((i*100/len(Archivos))//5)*"#",sep="",end="")
            
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
'''Esta funcion esta obsoleta y en futuras versiones será removida'''
def checkparquet(ruta)->int:
    """Esta funcion cuenta la cantidad de archivos que se transformaron en parquet
    Args:
        ruta (str): es la ruta donde se encuentran los archivos output.
    Returns:
        int: Cantidad de carpetas que hay en el directorio de la simulación
        Paso (int): Cantidad de pasos que salta entre archivos
    """
    if not os.path.isdir(f"{ruta}/datos"): 
        
        os.mkdir(f"{ruta}/datos")
        
    archivos = [d.replace("output.","").replace(".gz", "") for d in os.listdir(ruta) if ("output." in d) and (".gz" in d)]
    
    if not archivos:
        
        print("No hay datos registrados todavía.")
        
        return 0,0
    
    paso = sorted([float(i) for i in archivos]) ## El primer elemnto es el 0, el segundo es el paso
    
    return len(archivos), paso[1]
###############################################################################
def limpiar_pantalla():
    # 'nt' es Windows, 'posix' es Linux/macOS
    os.system('cls' if os.name == 'nt' else 'clear')
def rreplace(s, old, new="", ocurrence=1):
    li = s.rsplit(old, ocurrence)
    return new.join(li)

def buscar_carpeta():
    
    """Muestra los elementos registrados y permite elegir uno.
       Devuelve el nombre de la carpeta seleccionada."""
       
    mensaje = """Ingrese el numero de la carpeta a la que desea acceder
ingresar [-1] para elegir los datos de la carpeta actual
ingresar [-2] para volver atras : """
    valor = True
    ruta = "./"
    ultima_eleccion = []
    while valor:
        
        carpetas = [d for d in os.listdir(ruta) if os.path.isdir(ruta + d) 
                    and not d.startswith(".")]
        
        datos_gz = [d for d in os.listdir(ruta) if "output." in d and ".gz" in d]
        
        print("carpetas disponibles y 3 archivos output, si los hay:")
        
        for i, carpeta in enumerate(carpetas, start=1):
            
            print(f"[{i}] {carpeta}")
        
        for dato in datos_gz[:3]:
        
            print(dato)            
        
        try:
            
            eleccion = int(input(mensaje))
            
            if 1 <= eleccion <= len(carpetas):
                
                ultima_eleccion.append(f"{carpetas[eleccion-1]}/")
                ruta = os.path.join(ruta,ultima_eleccion[-1]) 
                limpiar_pantalla()
                
            elif eleccion == -1:
                
                limpiar_pantalla()
                return ruta
            
            elif eleccion == -2:
                try:
                    
                    ruta = rreplace(ruta, ultima_eleccion[-1])
                    ultima_eleccion.pop(-1)
                    limpiar_pantalla()
                    
                except IndexError:
                    
                    limpiar_pantalla()
                    print("Ya no se puede ir mas atrás, estas en ./")
            else:
                limpiar_pantalla()
                print("Número inválido, intente de nuevo.")
                
        except ValueError:
            
            print("Entrada inválida, por favor ingrese un número.")
            
def progress(i,N):
    print("\r", 
          "{:03d}".format(int((i * 100 / N) // 1)), 
          "/100%|", 
          int((i  * 100 / N) // 5) * "_" + "๑ï", 
          sep="", 
          end=""
          )
    
def detector_de_ondas(lf,R,altura):
    lf_onda = ( lf
     .with_columns((pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2)
                   .alias("v^2"))
     
     .filter(pl.col("v^2") > 4e-4)
     
     .filter((pl.col("x") < 10 * R) | 
             (pl.col("x").max() - 10 * R < pl.col("x")) |
             (pl.col("y") < 10 * R) |
             (pl.col("y").max() - 10 * R < pl.col("y")))
     .filter(pl.col("z") <= altura)
     )
    df_res_onda = (lf_onda
           .select([ pl.len()
                    .alias("cantidad"),
                    pl.col("v^2").max()
                    .alias("v^2_max") ])
           .collect()
           )
    return lf_onda, df_res_onda

def analisis_ovito(lf: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:

    # 1. Colectar a DataFrame
    df = lf.collect()
    
    # 2. Extraer posiciones como NumPy array
    positions = df.select(["x", "y", "z"]).to_numpy()
    
    # 3. Crear colección de datos manualmente (la forma correcta para datos 
    #    en memoria)
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property('Position', data=positions)
    
    cell = SimulationCell()
    cell[:, :] = [[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0]]  # celda grande
    data.objects.append(cell)
    # 4. Crear pipeline con fuente estática
    pipeline = Pipeline(source=StaticSource(data=data))
    
    # 5. Agregar modificadores
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))
    
    # 6. Computar
    computed_data = pipeline.compute()
    
    # 7. Extraer resultados
    coordinacion = computed_data.particles['Coordination'][:]
    cluster = computed_data.particles['Cluster'][:]
    
    # 8. Agregar columnas y devolver LazyFrame
    return df.with_columns([
        pl.Series("coordinacion", coordinacion),
        pl.Series("cluster", cluster)
    ]).lazy()

def data_analysis ():
    ruta = buscar_carpeta()
    base = os.path.join(ruta,"datos_analisis") 
    if not os.path.isdir(base): 
        os.mkdir(base)

    # Radio de los granos
    R = float(input("""ingrese el valor de radio de los granos en [m]
(se puede usar formato xe-y [m]): """))
    
    archivos = [d for d in os.listdir(ruta) if ("output." in d) and (".gz" in d)]
    archivos.sort(key = lambda x: '{0:0>19}'.format(x).lower())

    LenTotal = len(archivos)
    # Variables constantes fuera del bucle
    lf0 = Lector(f"{ruta}/output.0.gz", ["type", "z"])
    
    # Filtramos donde type == 1 y buscamos el maximo de z
    altura = 1.05 * ( lf0
                     .filter(pl.col("type") == 1)
                     .select(pl.col("z").max())
                     .collect()
                     .item()
                     )
    ## Estas son las columnas que se van a leer en el bucle for
    columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]
    # --- FASE 2: EL BUCLE DE PROCESAMIENTO ---
    print("Iniciando análisis de datos")
    print("000/100%|", 22 * "·", "|", sep="", end="")

    for i, file in enumerate(archivos):
        progress(i, LenTotal)
        
        # Lectura de los datos con columnas espcificas
        
        lf_datos = Lector(f"{ruta}/{file}", columnas)
        
    	# Buscamos si la onda llego al borde
        lf_onda, df_res_onda = detector_de_ondas(lf_datos, R, altura)
        
        
        
        cantidad = df_res_onda.item(0, "cantidad")
        max_v2   = df_res_onda.item(0, "v^2_max")
        bool_v2_max = (max_v2 >= 4e-2) if isinstance(max_v2,float) else False
        
        if cantidad > 3000 or bool_v2_max:
            
            print(f"""Se analizaron {i} archivos y en el archivo {file}:
                  Se detectó una onda de {cantidad} partículas
                  Con una v² máxima de {max_v2}""")
            ruta_guardado = os.path.join(base, f"onda.{i}.parquet")
            lf_onda.sink_parquet(ruta_guardado)
            
        else:
            lf_eyectados = ( lf_datos
                            .filter(pl.col("z") > altura)
                            .filter(pl.col("vz") > 0)
                            )
            
            ruta_guardado = os.path.join(base, f"salida.{i}.parquet")
            lf_procesados = analisis_ovito(lf_eyectados, R)
            lf_procesados.sink_parquet(ruta_guardado)
            
    progress(100, 100)
    print("Se terminó el análisis exitosamente")
    print(f"{LenTotal} datos guardados en: {base}")

# if i == 0:
# 	    Data_collect["Eyección"][0] = count_eyeccion
# 	    Data_collect["Eyectados_K"][0] = K_eyectados
# else:
#     if count_eyeccion - Data_collect["Eyección"][-1] >= 0:
#         Data_collect["Eyección"].append(count_eyeccion)
#         Data_collect["Eyectados_K"].append(K_eyectados)
#     else:
#         congelados = list(set(Data_collect["Vel_temp"][(i+1)%2])-set(Data_collect["Vel_temp"][i%2]))
#         Data_collect["Vel_temp"][i%2] = [x for x in Data_collect["Vel_temp"][i%2] if x != 0.0]
#         Data_collect["Vel_temp"][i%2].append(*congelados)
#         Data_collect["Eyección"].append(count_eyeccion+len(congelados))
#         Data_collect["Eyectados_K"].append(0.5*np.sum(Data_collect["Vel_temp"][i%2]))
#         Data_collect["K"][-1] += 0.5*np.sum(congelados)