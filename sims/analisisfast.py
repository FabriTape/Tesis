#!/usr/bin/env python
# coding: utf-8

# In[15]:

from gzip import open as opengz
from os import mkdir, listdir,path
from pandas import read_csv, read_parquet
from os import mkdir, listdir,path
def escrituracsv(movimiento:list,angulo,vel,ff=0.35,np=1):
    """La funcion escritura sirve para transformar archivos gz en csv

    Args:
        itera (int): Es la posición de los datos con indices 0,1,2,...,N siendo N la cantiad de archivos en la carpeta contenedora
        paso (int): Es el espaciado con el que se guaran los datos
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.
    """
    ## Definimos la carpeta donde estan los outputs de la sim.
    ruta = f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}"
    for archivo in listdir(ruta):
        if archivo.startswith("output.") and ".gz" in archivo:
            nombre = archivo.replace(".gz","")
            if  not path.isfile(f"{ruta}/{nombre}/{nombre}.csv"): ## Ver si no lo vonvertimos antes
                mkdir(f"{ruta}/{nombre}")
                ## Escirbir archivo ....
                with opengz(f'{ruta}/{archivo}', 'rt', newline='') as csv_file:
                    csv_data = csv_file.read()
                    with open(f'{ruta}/{nombre}/{nombre}.csv', 'wt') as out_file:
                        out_file.write(csv_data)
                print(f"Escritura de: {ruta}/{nombre}/{nombre}.csv\nFue exitosa :)")
            else:
                print(f"Ya existe el archivo {ruta}/{nombre}/{nombre}.csv")
def lecturacsv(itera,paso,movimiento:list,angulo,vel,ff=0.35,np=1):
    """La funcion lectura sirve para abir archivos csv

    Args:
        itera (int): Es la posición de los datos con indices 0,1,2,...,N siendo N la cantiad de archivos en la carpeta contenedora
        paso (int): Es el espaciado con el que se guaran los datos
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.

    Returns:
        pandas.DataFrame: Devuelve un data frame con los datos ["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"]
    """
    return read_csv(f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}/output.{paso*itera}/output.{paso*itera}.csv",skiprows=9,names=["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"],delimiter=" ")
def checkcsv(movimiento:list,angulo,vel,ff=0.35,np=1)->int:
    """Esta funcion cuenta la cantidad de archivos que se transformaron en csv
    Args:
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.
    Returns:
        int: Cantidad de carpetas que hay en el directorio de la simulación
        Paso (int): Cantidad de pasos que salta entre archivos
    """
    ruta = f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}"
    carpetas = [d for d in listdir(ruta) if d.startswith("output.") and not ".gz" in d]
    if not carpetas:
        print("No hay datos registrados todavía.")
        return 0
    return len(carpetas)


# In[12]:

def escrituraparquet(movimiento:list,angulo,vel,ff=0.35,np=1):
    """La funcion escritura sirve para transformar archivos csv en parquet

    Args:
        itera (int): Es la posición de los datos con indices 0,1,2,...,N siendo N la cantiad de archivos en la carpeta contenedora
        paso (int): Es el espaciado con el que se guaran los datos
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.
    """
    ## Definimos la carpeta donde estan los outputs de la sim.
    ruta = f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}"
    for archivo in listdir(ruta):
        if archivo.startswith("output."):
            nombre = archivo.replace(".csv","")
            if  not path.isfile(f"{ruta}/{nombre}/{nombre}.parquet"): ## Ver si no lo vonvertimos antes
                ## Escirbir archivo ....
                DF = read_csv(f"{ruta}/{nombre}/{nombre}.csv",skiprows=9,names=["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"],delimiter=" ")
                DF.to_parquet(f"{ruta}/{nombre}/{nombre}.parquet",compression="zstd")
                del DF
                print(f"Escritura de: {ruta}/{nombre}/{nombre}.csv\nFue exitosa :)")
            else:
                print(f"Ya existe el archivo {ruta}/{nombre}/{nombre}.parquet")
def lecturaparquet(itera,paso,movimiento:list,angulo,vel,ff=0.35,np=1):
    """La funcion lectura sirve para abir archivos parquet

    Args:
        itera (int): Es la posición de los datos con indices 0,1,2,...,N siendo N la cantiad de archivos en la carpeta contenedora
        paso (int): Es el espaciado con el que se guaran los datos
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.

    Returns:
        pandas.DataFrame: Devuelve un data frame con los datos ["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"]
    """
    return read_parquet(f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}/output.{paso*itera}/output.{paso*itera}.parquet",columns=["id", "type", "x", "y", "z", "vx", "vy", "vz"])
def checkparquet(movimiento:list,angulo,vel,ff=0.35,np=1)->int:
    """Esta funcion cuenta la cantidad de archivos que se transformaron en parquet
    Args:
        movimiento (list): Es el movimiento x,y,z del proyectil
        angulo (float): Es el ángulo con el que impacta el proyectil
        vel (float): velocidad en z del proyectil
        ff (float, optional): Es el  factor de llenado de la cama con la que se simula. Defaults to 0.35.
        np (int, optional): La cantidad de partículas usadas en el proyectil. Defaults to 1.
    Returns:
        int: Cantidad de carpetas que hay en el directorio de la simulación
        Paso (int): Cantidad de pasos que salta entre archivos
    """
    ruta = f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}"
    carpetas = [d for d in listdir(ruta) if d.startswith("output.") and not ".gz" in d]
    if not carpetas:
        print("No hay datos registrados todavía.")
        return 0
    return len(carpetas)


# In[ ]:


algo=pd.read_csv("./erosion_bed_SI_ff_0.35_np_1_vel_-10_angle_0_move_0_0_0/output.50000/output.50000.csv",skiprows=9,names=["id", "type", "x", "y", "z", "vx", "vy", "vz", "omegax", "omegay", "omegaz", "c_1"],delimiter=" ")
print("#"*83,"\n"," "*27,"Etiquetas de los data frames\n","#"*83)
print(list(algo.columns))
print("""
id : Es el número de la partícula
type : Es si es cama o proyectil 1 o 2 respectivamente
x , y , z : no necesita introducción
vx , vy , vz : tampoco necesita introducción
omegax , omegay , omegaz : tampoco necesita introducción
c_1 : creo que es la densidad""")



# In[6]:


## Escribo los archivos de gz a csv de ser necesario
escrituracsv(movimiento=[0,0,0],angulo=0,vel=-10)
## Escribo los archivos de csv a parquet de ser necesario
escrituraparquet(movimiento=[0,0,0],angulo=0,vel=-10)

# In[2583]:
import numexpr as ne
import pandas as pd
import numpy as np
import os

movimiento = [0,0,0]
angulo = 0 
vel = -10
threshold = 0.004

# Asumo que ya tienes tu función checkparquet importada o definida
LenTotal, paso = checkparquet(movimiento, angulo, vel), 50000

# Pre-allocating lists is slightly faster, but lists are okay for <100k iters
Data_collect = {
    "K": np.zeros((LenTotal,1)), "Dispersión": np.zeros((LenTotal,1)), "Max": np.zeros((LenTotal,1)), "Min":np.zeros((LenTotal,1)) ,
    "Vcm": np.zeros((LenTotal,3)), "N": 0, "Eyección": np.zeros((LenTotal,1)),"Eyectados_K":np.zeros((LenTotal,1))
}

# Variables constantes fuera del bucle
v0 = 0
N = 0
altura = 0

print(f"Iniciando análisis de {LenTotal} archivos...")

for i in range(LenTotal):
    # 1. Lectura Optimizada
    # Usamos engine='pyarrow' explícitamente para asegurar velocidad en Parquet
    # Asumo que lecturaparquet devuelve el DF. Si puedes, edita lecturaparquet
    # para que devuelva columns=['type', 'z', 'vx', 'vy', 'vz'] solamente si no usas las otras.
    Datos = lecturaparquet(i, paso, movimiento, angulo, vel)
    
    # 2. Convertir a Numpy Arrays INMEDIATAMENTE
    # Esto elimina el overhead de Pandas para las operaciones matemáticas siguientes
    vx = Datos["vx"].to_numpy()
    vy = Datos["vy"].to_numpy()
    vz = Datos["vz"].to_numpy()
    z_pos = Datos["z"].to_numpy()
    types = Datos["type"].to_numpy()

    # Lógica de inicialización (solo primer frame)
    if i == 0:
        # Vectorización booleana (mask) es más rápida que querys de pandas
        altura = z_pos[types == 1].max()
        
        # Verificar proyectil
        mask_proyectil = (types == 2)
        if not np.any(mask_proyectil): # Equivalente a .empty
            v0_actual = vel
            N = len(types) + 1
            Data_collect["N"] = N
        else:
            v0_actual = 0
            N = len(types) # Asumiendo N constante si está el proyectil
            if Data_collect["N"] == 0: Data_collect["N"] = N
    else:
        # Pequeña corrección lógica: Si v0 solo aplica cuando no está el proyectil,
        # asegúrate de que v0_actual sea consistente en iteraciones siguientes.
        # Asumiré v0=0 para i > 0 según tu lógica original 'else'.
        v0_actual = 0 

    # 3. Matemáticas con Numpy (Mucho más rápido)
    # Velocidad Centro de Masa del frame actual
    # np.sum es más rápido que Sumatoria sobre series pandas grandes
    vcm_x = np.sum(vx) / N
    vcm_y = np.sum(vy) / N
    vcm_z = (np.sum(vz) + v0_actual) / N
    
    # Guardar Vcm (como lista simple para serializar json luego si hace falta)
    Data_collect["Vcm"][i,:] = [vcm_x, vcm_y, vcm_z]

    # Calculo módulo velocidad relativa (Vectorizado)
    # Al usar arrays de numpy (vx, vy, vz) y escalares (vcm_x, etc), 
    # esta operación se ejecuta en C optimizado.
    #Vnorm2 = vx**2 + vy**2 + vz**2
    Vnorm2 = ne.evaluate("vx**2 + vy**2 + vz**2")


    # Energía Cinética
    Data_collect["K"][i] = 0.5 * np.sum(Vnorm2)
    
    # Max / Min
    Data_collect["Max"][i] = np.max(Vnorm2)
    Data_collect["Min"][i] = np.min(Vnorm2)
    
    # 4. Filtro de Eyección Optimizado
    # Usamos las arrays de numpy z_pos y vz que ya extrajimos
    # Esto evita volver a consultar al DataFrame gigante
    filtro = (z_pos > altura) & (vz > threshold)
    count_eyeccion = np.sum(filtro)
    Data_collect["Eyección"][i] = count_eyeccion
    Data_collect["Eyectados_K"][i] = 0.5 * np.sum(Vnorm2[filtro])

    # 5. Gestión de Memoria
    # Solo borramos la referencia al DF y arrays grandes.
    # NO LLAMAR A GC.COLLECT()
    del Datos, vx, vy, vz, z_pos, types, Vnorm2


# In[258]:
import numexpr as ne
import pandas as pd
import numpy as np
import os

movimiento = [0,0,0]
angulo = 0 
vel = -10
threshold = 0.004

# Asumo que ya tienes tu función checkparquet importada o definida
LenTotal, paso = checkparquet(movimiento, angulo, vel), 50000

# Pre-allocating lists is slightly faster, but lists are okay for <100k iters
Data_collect = {
    "K": [], "Dispersión": [], "Max": [], "Min": [], 
    "Vcm": [], "N": 0, "Eyección": [],"Eyectados_K":[]
}

# Variables constantes fuera del bucle
v0 = 0
N = 0
altura = 0

print(f"Iniciando análisis de {LenTotal} archivos...")

for i in range(LenTotal):
    # 1. Lectura Optimizada
    # Usamos engine='pyarrow' explícitamente para asegurar velocidad en Parquet
    # Asumo que lecturaparquet devuelve el DF. Si puedes, edita lecturaparquet
    # para que devuelva columns=['type', 'z', 'vx', 'vy', 'vz'] solamente si no usas las otras.
    Datos = lecturaparquet(i, paso, movimiento, angulo, vel)
    
    # 2. Convertir a Numpy Arrays INMEDIATAMENTE
    # Esto elimina el overhead de Pandas para las operaciones matemáticas siguientes
    vx = Datos["vx"].to_numpy()
    vy = Datos["vy"].to_numpy()
    vz = Datos["vz"].to_numpy()
    z_pos = Datos["z"].to_numpy()
    types = Datos["type"].to_numpy(dtype=np.int8)

    # Lógica de inicialización (solo primer frame)
    if i == 0:
        # Vectorización booleana (mask) es más rápida que querys de pandas
        altura = z_pos[types == 1].max()
        
        # Verificar proyectil
        mask_proyectil = (types == 2)
        if not np.any(mask_proyectil): # Equivalente a .empty
            v0_actual = vel
            N = len(types) + 1
            Data_collect["N"] = N
        else:
            v0_actual = 0
            N = len(types) # Asumiendo N constante si está el proyectil
            if Data_collect["N"] == 0: Data_collect["N"] = N
    else:
        # Pequeña corrección lógica: Si v0 solo aplica cuando no está el proyectil,
        # asegúrate de que v0_actual sea consistente en iteraciones siguientes.
        # Asumiré v0=0 para i > 0 según tu lógica original 'else'.
        v0_actual = 0 

    # 3. Matemáticas con Numpy (Mucho más rápido)
    # Velocidad Centro de Masa del frame actual
    # np.sum es más rápido que Sumatoria sobre series pandas grandes
    vcm_x = np.sum(vx) / N
    vcm_y = np.sum(vy) / N
    vcm_z = (np.sum(vz) + v0_actual) / N
    
    # Guardar Vcm (como lista simple para serializar json luego si hace falta)
    Data_collect["Vcm"].append([vcm_x, vcm_y, vcm_z])

    # Calculo módulo velocidad relativa (Vectorizado)
    # Al usar arrays de numpy (vx, vy, vz) y escalares (vcm_x, etc), 
    # esta operación se ejecuta en C optimizado.
    Vnorm2 = ne.evaluate("(vx - vcm_x)**2 + (vy - vcm_y)**2 + (vz - vcm_z)**2")

    # Energía Cinética
    Data_collect["K"].append(0.5 * np.sum(Vnorm2))
    
    # Max / Min
    Data_collect["Max"].append(np.max(Vnorm2))
    Data_collect["Min"].append(np.min(Vnorm2))
    
    # 4. Filtro de Eyección Optimizado
    # Usamos las arrays de numpy z_pos y vz que ya extrajimos
    # Esto evita volver a consultar al DataFrame gigante
    filtro = (z_pos > altura) & (vz > threshold)
    count_eyeccion = np.sum(filtro)
    Data_collect["Eyección"].append(count_eyeccion)
    Data_collect["Eyectados_K"].append(0.5 * np.sum(Vnorm2[filtro]))
    # 5. Gestión de Memoria
    # Solo borramos la referencia al DF y arrays grandes.
    # NO LLAMAR A GC.COLLECT()
    
# In[13]:


import gc
from numpy import sum as Sumatoria
from numpy import mean , std
movimiento = [0,0,0]
angulo = 0 
vel = -10
threshold = 0.004
## Obtengo la cantidad de archivos y los saltos de las simulaciones
LenTotal,paso = checkparquet(movimiento,angulo,vel),50000
## Tiro toda la grabage a René Descart
gc.collect(2)
## Inicializo mi análisis 
Data_collect = {"K":[],"Dispersión":[],"Max":[],"Min":[],"Vcm":[],"N":0,"Eyección":[]}
for i in range(LenTotal):
    ## Extraigo mis datos
    Datos = lecturaparquet(i,paso,movimiento,angulo,vel)
    ## Busco la altura de la cama
    if i==0:
        altura = Datos[Datos["type"]==1]["z"].max() ## Esta variable NO HAY QUE BORRARLA, NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    ## En caso de que no este el proyectil en output.0 lo pongo a mano jeje
    if Datos[Datos["type"]==2].empty and i==0:
        v0 = vel
        N = len(Datos["type"])+1
        Data_collect["N"] = N
    ## Esto es para añadir el proyectil unicamente 
    else:
        v0=0    
    ## Calculo la velocidad del CM
    Data_collect["Vcm"].append([Sumatoria(Datos["vx"])/N,Sumatoria(Datos["vy"])/N,(Sumatoria(Datos["vz"])+v0)/N])
    ## Calculo el módulo de velocidad relativa a la velocidad del CM 
    Vnorm2 = (Datos["vx"]-Data_collect["Vcm"][i][0])**2+(Datos["vy"]-Data_collect["Vcm"][i][1])**2+(Datos["vz"]-Data_collect["Vcm"][i][2])**2
    ## Calculo la energía cinética de la cama (para añadir la de la partícula, a t=0 sumar 0.5*v_0^2)
    Data_collect["K"].append(Sumatoria(Vnorm2)*0.5)
    ## Busco las velocidades máximas y mínimas
    Data_collect["Max"].append(max(Vnorm2))
    Data_collect["Min"].append(min(Vnorm2))
    ## Pongo un filtro para buscar las partículas eyectadas
    filtro = (Datos["z"]>altura) & (Datos["vz"]>threshold)
    Data_collect["Eyección"].append(filtro.sum())
    ## Limpio la garbage
    del Datos, Vnorm2, filtro
    gc.collect()


# In[16]:


import gc
from numpy import sum as Sumatoria
from numpy import mean , std
movimiento = [0,0,0]
angulo = 0 
vel = -10
threshold = 0.004
## Obtengo la cantidad de archivos y los saltos de las simulaciones
LenTotal,paso = checkcsv(movimiento,angulo,vel),50000

## Tiro toda la grabage a René Descart
gc.collect(2)
## Inicializo mi análisis 
Data_collect = {"K":[],"Dispersión":[],"Max":[],"Min":[],"Vcm":[],"N":0,"Eyección":[]}
for i in range(LenTotal):
    ## Extraigo mis datos
    Datos = lecturacsv(i,paso,movimiento,angulo,vel)
    ## Busco la altura de la cama
    if i==0:
        altura = Datos[Datos["type"]==1]["z"].max() ## Esta variable NO HAY QUE BORRARLA, NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    ## En caso de que no este el proyectil en output.0 lo pongo a mano jeje
    if Datos[Datos["type"]==2].empty and i==0:
        v0 = vel
        N = len(Datos["type"])+1
        Data_collect["N"] = N
    ## Esto es para añadir el proyectil unicamente 
    else:
        v0=0    
    ## Calculo la velocidad del CM
    Data_collect["Vcm"].append([Sumatoria(Datos["vx"])/N,Sumatoria(Datos["vy"])/N,(Sumatoria(Datos["vz"])+v0)/N])
    ## Calculo el módulo de velocidad relativa a la velocidad del CM 
    Vnorm2 = (Datos["vx"]-Data_collect["Vcm"][i][0])**2+(Datos["vy"]-Data_collect["Vcm"][i][1])**2+(Datos["vz"]-Data_collect["Vcm"][i][2])**2
    ## Calculo la energía cinética de la cama (para añadir la de la partícula, a t=0 sumar 0.5*v_0^2)
    Data_collect["K"].append(Sumatoria(Vnorm2)*0.5)
    ## Busco las velocidades máximas y mínimas
    Data_collect["Max"].append(max(Vnorm2))
    Data_collect["Min"].append(min(Vnorm2))
    ## Pongo un filtro para buscar las partículas eyectadas
    filtro = (Datos["z"]>altura) & (Datos["vz"]>threshold)
    Data_collect["Eyección"].append(filtro.sum())
    ## Limpio la garbage
    del Datos, Vnorm2, filtro
    gc.collect()


# In[ ]:


import matplotlib.pyplot as plt 
from numpy import array
from scipy.special import lambertw
from numpy import exp
from numpy import inf
from scipy.optimize import curve_fit
from numpy import linspace
from numpy import sqrt
def modelo(x,a,b,c,d):
    return a*(x-d)*exp(-b*(x-d))+c
def derA(x,a,b,c,d):
    return (x-d)*exp(-b*(x-d))
def derB(x,a,b,c,d):
    return -a*(x-d)**2*exp(-b*(x-d))
def derC(x,a,b,c,d):
    return 1
def derD(x,a,b,c,d):
    return a*(b*(x-d)-1)*exp(-b*(x-d))
def grad(x,popt):
    return array([derA(x,*popt),derB(x,*popt),derC(x,*popt),derD(x,*popt)])
def errorf(absisa,popt,pcov):
    return array([sqrt(grad(x,popt).T @ pcov @ grad(x,popt)) if x!=0 else 0 for x in absisa])
K = array(Data_collect["K"])
y_err = linspace(1,1,27)
y_err[0] = 1e-2
popt, pcov = curve_fit(modelo,linspace(0,26,27),K,sigma=y_err)
vx , vy , vz = array(Data_collect['Vcm'])[:,0] , array(Data_collect['Vcm'])[:,1] , array(Data_collect['Vcm'])[:,2]

X = linspace(0,26,200)
fig,ax = plt.subplots(2,1,figsize=(6,10))
ax[0].plot(K,"r-",label="Energía cinética de la cama")
#ax[0].plot(X,modelo(X,*popt),"b-",label="Energía cinética de la cama")
#ax[0].fill_between(X,modelo(X,*popt)-errorf(X,popt,pcov),modelo(X,*popt)+errorf(X,popt,pcov),alpha=0.5,label=r"error del ajuste")
ax[0].set_xlabel("Paso de tiempo/50000")
ax[0].set_ylabel("Energía cinética K/m")
ax[1].plot(Data_collect['Eyección'],"o",label="Parículas eyectadas")
ax[1].set_xlabel("Paso de tiempo/50000")
ax[1].set_ylabel("Cantidad de parículas eyectadas")
ax[1].legend()
#ax[0].plot((vx**2+vy**2+vz**2)/max((vx**2+vy**2+vz**2)),"o",label="Módulo del momento del sistema\nnormalizado")
ax[0].legend()


# In[ ]:


from numpy import linspace
K_cama = array(Data_collect["K"])
K_cama[0] -= 0.5*vel**2
rango = array(Data_collect["Dispersión"])
plt.plot(K_cama,"g--",label="Energía cinética media")
plt.fill_between(linspace(0,25,27),array(Data_collect["Min"])*0.5,array(Data_collect["Max"])*0.5,alpha=0.5,label=r"rango")
plt.plot(rango,"r--",label="rango")
plt.xlim((0,25))
plt.ylim((0,4.5))
plt.legend()


# In[ ]:


#%matplotlib qt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.plot(vx,vy,zs=vz)


# In[ ]:


plt.plot((vx**2+vy**2+vz**2)*0.5)


# In[ ]:


plt.plot(vx,label="x")
plt.plot(vy,label="y")
plt.plot(vz,label="z")
plt.legend()


# In[ ]:


# Source - https://stackoverflow.com/a
# Posted by Mr. T, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-27, License - CC BY-SA 3.0

from scipy.optimize import curve_fit
import numpy as np
def smooth_data_v1(x_arr,y_arr):
    def func(x, a, b, c):
        return a*np.exp(-b*x)+c

    #create the weighting array
    y_weight = np.empty(len(y_arr))
    print(y_weight)
    #high pseudo-sd values, meaning less weighting in the fit
    y_weight.fill(10)
    print(y_weight)
    #low values for point 0 and the last points, meaning more weighting during the fit procedure 
    y_weight[0] = y_weight[-5:-1] = 0.1
    print(y_weight)
    popt, pcov = curve_fit(func, x_arr, y_arr, p0=(y_arr[0], 1, 1), sigma = y_weight, absolute_sigma = True)
    print("a, b, c:", *popt)
    y_smooth = func(x_arr, *popt)

    return y_smooth


x_orig = np.asarray([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14])
y_orig = np.asarray([ 445057,  447635,  450213,  425089,  391746,  350725,  285433,  269027,
  243835,  230587,  216757,  202927,  189097,  175267,  161437])
plt.plot(x_orig,y_orig,"o")
print(smooth_data_v1(x_orig, y_orig))


# In[ ]:




