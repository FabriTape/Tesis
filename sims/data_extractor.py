# -*- coding: utf-8 -*-
"""
Extractor de datos para el análisis de simulaciones LAMMPS de erosión.

Este módulo orquesta el pipeline completo de análisis de datos utilizando
una arquitectura basada en máquina de estados. Coordina las dos fases de
procesamiento (detección de onda y análisis topológico) y exportando los
resultados para su posterior visualización.

Arquitectura:
    - Máquina de estados controlada por StateManager
    - Contexto compartido para persistencia de datos entre estados
    - Input wrapper con validación y comandos globales (b/q)
    - Configuración externalizada en config.yaml

Flujo de ejecución:
    1. SELECCION_CARPETAS: Navegación y selección de directorios
    2. CARGAR_CONFIG: Carga de configuración YAML
    3. CONFIGURACION_RADIO: Ingreso del radio de partículas
    4. CONFIGURACION_ONDA: Opción de análisis de onda (con detección de criticidad)
    5. CONFIGURACION_FILTROS: Umbrales de filtrado de erosión
    6. VERIFICACION_PREVIA: Check de análisis previos por simulación
    7. EJECUCION_FASE1: Detección de onda (FASE 1)
    8. EJECUCION_FASE2: Análisis OVITO (FASE 2)
    9. EXPORTACION: Guardado de resultados + plot
    10. COMPLETADO: Fin del procesamiento
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from lammps_utils import control_caja, extractor_de_borde, max_altura
from ovito_utils import ejecutar_fase_ovito, inicializar_pipeline_ovito
from plot import plot_erosion
from state_manager import Estado, StateManager, es_simulacion_critica
from utils import (
    DIR_RESULTADOS,
    TZ_ARG,
    extraer_metadatos,
    guardar_resultados,
    preparar_archivos,
    verificar_analisis_previo,
)


def data_analysis() -> None:
    """
    Orquestador principal que ejecuta el pipeline de extracción utilizando
    máquina de estados.

    Esta función actúa como puente entre la máquina de estados y la lógica
    de procesamiento real. Los handlers de StateManager gestionan la UI y
    navegación, mientras que este módulo ejecuta el procesamiento pesado.

    Args:
        None

    Returns:
        None: La función actúa como controlador de flujo y no retorna valores.
    """
    # Inicializar la máquina de estados
    state_manager = StateManager(config_path=Path("./config.yaml"))
    contexto = state_manager.contexto

    # Estados iniciales (UI interactiva manejada por StateManager)
    # SELECCION_CARPETAS -> CARGAR_CONFIG -> CONFIGURACION_RADIO ->
    # CONFIGURACION_ONDA -> CONFIGURACION_FILTROS

    # Ejecutar estados iniciales de configuración
    while state_manager.estado_actual < Estado.VERIFICACION_PREVIA:
        handler = state_manager.handlers.get(state_manager.estado_actual)
        if handler is None:
            print(f"Error: No hay handler para el estado {state_manager.estado_actual}")
            return

        siguiente_estado = handler()
        if siguiente_estado is not None:
            state_manager.estado_actual = siguiente_estado
        else:
            state_manager.estado_actual += 1

    # Verificar que tenemos los datos necesarios
    if not contexto.rutas or not contexto.R or not contexto.config:
        print("Error: Faltan datos de configuración necesarios.")
        return

    # Crear directorio de resultados
    DIR_RESULTADOS.mkdir(exist_ok=True)

    # Procesar cada ruta seleccionada
    for ruta in contexto.rutas:
        # Resetear datos de simulación
        contexto.reset_simulacion()
        contexto.ruta_actual = ruta

        print(f"\n{'=' * 50}")
        print(f"Procesando: {ruta.name}")
        print(f"{'=' * 50}")

        datos = ruta.name.replace("erosion_bed_SI_", "", 1)
        contexto.datos = datos
        contexto.fecha_hora = datetime.now(TZ_ARG).strftime("%Y-%m-%d_%H-%M-%S")

        # Extraer metadatos de esta simulación individual
        metadatos = extraer_metadatos(datos)

        # Evaluar criticidad para esta simulación específica
        hacer_onda_simulacion = contexto.hacer_onda  # Default global

        if es_simulacion_critica(metadatos, contexto.config):
            print("\n[!] DETECTADA SIMULACIÓN CRÍTICA")
            print(
                f"    v={metadatos.get('vel0', 'N/A')} m/s, "
                f"np={metadatos.get('np', 'N/A')}, "
                f"move=({metadatos.get('posx', '0')}, {metadatos.get('posy', '0')}, {metadatos.get('posz', '0')})"
            )
            print(
                "\n    El análisis de onda es ALTAMENTE RECOMENDADO para esta simulación."
            )

            # Si el default global era False, preguntar específicamente para esta simulación
            if not contexto.hacer_onda:
                from input_handler import InputHandler

                handler = InputHandler()
                print(
                    "\n¿Desea realizar el análisis de onda para ESTA simulación crítica?"
                )
                hacer_onda_critica, signal = handler.safe_input(
                    "    (Y/N, default=Y): ",
                    tipo=bool,
                    default=True,
                    permitir_back=False,
                    permitir_quit=False,
                )
                if signal == "continue" and hacer_onda_critica is not None:
                    hacer_onda_simulacion = hacer_onda_critica
                    print(
                        f"\n✓ Análisis de onda para esta simulación: {'SÍ' if hacer_onda_simulacion else 'NO'}"
                    )
            else:
                print("\n✓ Análisis de onda: SÍ (configuración global)")
        else:
            # No es crítica, usar configuración global
            pass  # hacer_onda_simulacion ya tiene el valor global

        # Estado VERIFICACION_PREVIA (por simulación)
        continuar, contexto.sobreescribir_todo = verificar_analisis_previo(
            datos, contexto.umbrales_filtro, contexto.sobreescribir_todo
        )

        if not continuar:
            print(f"Saltando {ruta.name}...")
            continue

        # Preparación de archivos y caja
        archivos, paso_int, len_total = preparar_archivos(ruta)

        if len_total == 0:
            print(f"No se encontraron archivos output.*.gz en {ruta.name}. Saltando...")
            continue

        contexto.archivos = archivos
        contexto.paso_int = paso_int

        limites_i = extractor_de_borde(ruta, archivos[0])
        limites_f = extractor_de_borde(ruta, archivos[-1])
        usar_caja = control_caja(limites_i, limites_f, archivos)

        contexto.limites_i = limites_i
        contexto.limites_f = limites_f
        contexto.usar_caja = usar_caja

        columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

        try:
            altura = max_altura(ruta, contexto.R)
            contexto.altura = altura
        except ValueError as e:
            print(f"Error calculando altura para {ruta.name}: {e}. Saltando...")
            continue

        # Inicializar pipeline de OVITO
        pipeline_global = inicializar_pipeline_ovito(cutoff=contexto.R)

        # Estado EJECUCION_FASE1: Búsqueda rápida de la onda (Opcional)
        if hacer_onda_simulacion:
            from analisis_onda import ejecutar_fase_onda

            onda_detectada, paso_corte = ejecutar_fase_onda(
                archivos,
                ruta,
                columnas,
                contexto.R,
                altura,
                limites_f,
                usar_caja,
                datos,
                contexto.fecha_hora,
            )
            contexto.onda_detectada = onda_detectada
            contexto.paso_corte = paso_corte
        else:
            contexto.onda_detectada = False
            contexto.paso_corte = len(archivos)

        # Estado EJECUCION_FASE2: Análisis pesado con OVITO
        archivos_fase2 = archivos[: contexto.paso_corte]
        lista_resultados = ejecutar_fase_ovito(
            archivos_fase2,
            ruta,
            columnas,
            pipeline_global,
            altura,
            datos,
            paso_int,
            contexto.onda_detectada,
        )

        contexto.lista_resultados = lista_resultados

        # Estado EXPORTACION: Exportación de Datos + Plot
        guardar_resultados(
            lista_resultados, datos, contexto.fecha_hora, contexto.umbrales_filtro
        )

        # Generación automática de gráficos
        if lista_resultados:
            print("\nGenerando gráfico de erosión...")
            df_concat = pl.concat(lista_resultados, how="vertical_relaxed")

            # Extraer metadatos del DataFrame para el nombre del archivo
            try:
                ff = float(df_concat["ff"].item(0))
                np_ = float(df_concat["np"].item(0))
                vel = float(df_concat["vel0"].item(0))
                angle = float(df_concat["angulo"].item(0))
                move1 = float(df_concat["posx"].item(0))
                move2 = float(df_concat["posy"].item(0))
                move3 = float(df_concat["posz"].item(0))

                plot_erosion(
                    df_concat,
                    ff,
                    np_,
                    vel,
                    angle,
                    move1,
                    move2,
                    move3,
                    DIR_RESULTADOS,
                )
            except (KeyError, ValueError) as e:
                print(f"No se pudieron extraer los metadatos para el gráfico: {e}")

    # Estado COMPLETADO
    print("\n" + "=" * 50)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 50)
    print(f"Datos guardados en: {DIR_RESULTADOS}")


# Punto de entrada de la ejecución
if __name__ == "__main__":
    data_analysis()
