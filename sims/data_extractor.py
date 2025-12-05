#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
from gzip import open as opengz
from os import mkdir, listdir,path
from pandas import read_csv, read_parquet

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
                print(f"Escritura de: {ruta}/{nombre}/{nombre}.parquet\nFue exitosa :)")
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

    return read_parquet(f"./erosion_bed_SI_ff_{ff}_np_{np}_vel_{vel}_angle_{angulo}_move_{movimiento[0]}_{movimiento[1]}_{movimiento[2]}/output.{paso*itera}/output.{paso*itera}.parquet",columns=["id", "type","x","y", "z", "vx", "vy", "vz"])
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

