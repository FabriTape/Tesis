#!/home/tutturuu/spyder-env/bin python3
# -*- coding: utf-8 -*-
import polars as pl
import subprocess
def Lector(archivo_gz:str,datos:list):
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]
    
    proc = subprocess.Popen(
        f"zcat {archivo_gz} | tail -n +10 | sed 's/[[:space:]]\+/ /g'", 
        shell=True, 
        stdout=subprocess.PIPE)
    
    df_archivo = pl.scan_csv(
        proc.stdout, # Leemos directo de la memoria del proceso
        separator=" ", # Ahora sí es seguro usar un solo espacio
        has_header=False,
        new_columns=nombres_columnas,
        infer_schema_length=10000,
        schema_overrides={
        "type": pl.UInt8,      # Forzamos 'type' a ser Entero de 8 bits
        "id": pl.UInt64       # Es buena práctica definir los IDs grandes
        })
    return df_archivo.select(datos)

