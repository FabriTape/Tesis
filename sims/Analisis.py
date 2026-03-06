# -*- coding: utf-8 -*-
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource


def Lector(archivo_gz: str, datos: list):
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    df_archivo = pl.scan_csv(
        archivo_gz,
        skip_rows=9,
        has_header=False,
        new_columns=nombres_columnas,
        separator=" ",  # Cambiamos tab por espacio
        truncate_ragged_lines=True,  # Ignora espacios extra al final de la línea
        infer_schema_length=10000,
        schema_overrides={"type": pl.UInt8, "id": pl.UInt64},
    )

    return df_archivo.select(datos)


def limpiar_pantalla():
    """Limpia la terminal según el sistema operativo (Windows o Linux/macOS).
    Args:
        None
    Returns:
        None
    """
    # 'nt' es Windows, 'posix' es Linux/macOS
    os.system("cls" if os.name == "nt" else "clear")


def rreplace(s, old, new="", ocurrence=1):
    """Reemplaza una subcadena buscando desde el final del string (derecha).
    Args:
        s (str) : Cadena de texto original.
        old (str) : Subcadena que se desea reemplazar.
        new (str) : Subcadena nueva que reemplazará a la antigua.
        ocurrence (int) : Cantidad de veces que se realizará el reemplazo.
    Returns:
        (str) : Cadena de texto con los reemplazos realizados.
    """
    li = s.rsplit(old, ocurrence)
    return new.join(li)


# Funcion obtenida de
# https://www.geeksforgeeks.org/python/delete-a-directory-or-file-using-python/
"""Esta funcion esta en desuso y será removida en suturas versiones"""


def borrar_dir(ruta):
    """Elimina todos los archivos de un directorio y luego la carpeta misma.
    Args:
        ruta (str) : Trayectoria de la carpeta que se desea eliminar.
    Returns:
        None
    """
    for filename in os.listdir(ruta):
        file_path = os.path.join(ruta, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the file
    os.rmdir(ruta)


def buscar_carpeta():
    """Permite la navegación interactiva usando objetos de ruta (Path)."""

    mensaje = """\nIngrese el numero de la carpeta a la que desea acceder
ingresar [-1] para elegir los datos de la carpeta actual
ingresar [-2] para volver atras : """

    # Iniciamos con la ruta actual absoluta
    ruta = Path(".").resolve()

    while True:
        # Listar carpetas (filtrando ocultas)
        carpetas = sorted(
            [d for d in ruta.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

        # Listar archivos de simulación (solo para visualización)
        datos_gz = sorted([f.name for f in ruta.glob("output.*.gz")])

        print(f"\n--- Ubicación actual: {ruta} ---")
        print("Carpetas disponibles:")
        for i, carpeta in enumerate(carpetas, start=1):
            print(f"[{i}] {carpeta.name}")

        if datos_gz:
            print("\nArchivos de datos encontrados (muestra):")
            for dato in datos_gz[:3]:
                print(f"  {dato}")

        try:
            eleccion = int(input(mensaje))

            if 1 <= eleccion <= len(carpetas):
                # Ir hacia adelante
                ruta = carpetas[eleccion - 1]
                limpiar_pantalla()

            elif eleccion == -1:
                # Seleccionar carpeta actual
                limpiar_pantalla()
                return str(
                    ruta
                )  # Retornamos como string para que el resto del código no cambie

            elif eleccion == -2:
                # Ir hacia atrás de forma segura
                ruta = ruta.parent
                limpiar_pantalla()

            else:
                limpiar_pantalla()
                print("Número inválido, intente de nuevo.")

        except ValueError:
            print("Entrada inválida, por favor ingrese un número.")


def generar_carpeta():
    """Crea una carpeta para los resultados gestionando la sobreescritura.
    Args:
        None
    Returns:
        ruta (str) : Ruta de la carpeta de origen de los datos.
        base (str) : Ruta de la carpeta de destino para el análisis.
    """
    ruta = buscar_carpeta()
    base = os.path.join(ruta, "datos_analisis")

    base += (datetime.now(UTC) + timedelta(hours=-3)).strftime("_%Y-%m-%d_%H:%M")
    os.mkdir(base)
    return ruta, base


ruta, base = generar_carpeta()
archivos = [d for d in os.listdir(ruta) if ("output." in d) and (".gz" in d)]
