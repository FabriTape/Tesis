"""
Gestor de estados para la máquina de estados del análisis LAMMPS.

Este módulo implementa:
- La máquina de estados que controla el flujo del programa
- La función de evaluación de criticidad física (es_simulacion_critica)
- El contexto compartido que persiste entre estados
- Los handlers para cada estado del sistema
"""

import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import yaml

from input_handler import InputHandler
from utils import limpiar_pantalla


class Estado(IntEnum):
    """
    Enumeración de los estados del sistema.

    Cada estado representa una etapa del proceso de análisis.
    Los valores numéricos permiten navegación hacia adelante/atrás.
    """

    SELECCION_CARPETAS = 0
    CARGAR_CONFIG = 1
    CONFIGURACION_RADIO = 2
    CONFIGURACION_ONDA = 3
    CONFIGURACION_FILTROS = 4
    VERIFICACION_PREVIA = 5
    EJECUCION_FASE1 = 6
    EJECUCION_FASE2 = 7
    EXPORTACION = 8
    COMPLETADO = 9


@dataclass
class Contexto:
    """
    Contexto compartido que persiste entre estados.

    Almacena todos los datos ingresados por el usuario y los resultados
    intermedios de cada fase del análisis.

    Attributes:
        config (dict[str, Any]): Configuración cargada desde config.yaml.
        rutas (list[Path] | None): Directorios seleccionados para procesar.
        R (float | None): Radio de las partículas en metros.
        hacer_onda (bool): Si True, realizar análisis de onda.
        umbrales_filtro (dict[str, float]): Umbrales para el criterio de erosión.
        sobreescribir_todo (bool): Si True, sobreescribir todos los análisis previos.
        ruta_actual (Path | None): Ruta que se está procesando actualmente.
        datos (str | None): Identificador de la simulación actual.
        fecha_hora (str | None): Marca temporal de la ejecución.
        archivos (list[str] | None): Archivos de volcado encontrados.
        paso_int (int | None): Frecuencia de volcado de datos (timestep).
        limites_i (dict | None): Límites iniciales de la caja.
        limites_f (dict | None): Límites finales de la caja.
        usar_caja (bool): Si True, usar límites fijos de la caja.
        altura (float | None): Altura máxima de la cama de partículas.
        onda_detectada (bool): Si True, se detectó una onda en Fase 1.
        paso_corte (int | None): Iteración donde se cortó el análisis.
        lista_resultados (list | None): Resultados de la Fase 2.
    """

    config: dict[str, Any] | None = None
    rutas: list[Path] | None = None
    R: float | None = None
    hacer_onda: bool = False
    umbrales_filtro: dict[str, float] = field(default_factory=dict)
    sobreescribir_todo: bool = False

    # Datos por simulación (se resetean para cada ruta)
    ruta_actual: Path | None = None
    datos: str | None = None
    fecha_hora: str | None = None
    archivos: list[str] | None = None
    paso_int: int | None = None
    limites_i: dict | None = None
    limites_f: dict | None = None
    usar_caja: bool = True
    altura: float | None = None
    onda_detectada: bool = False
    paso_corte: int | None = None
    lista_resultados: list | None = None

    def reset_simulacion(self) -> None:
        """
        Resetea los datos específicos de la simulación actual.

        Se llama al comenzar el procesamiento de cada nueva ruta.
        """
        self.ruta_actual = None
        self.datos = None
        self.fecha_hora = None
        self.archivos = None
        self.paso_int = None
        self.limites_i = None
        self.limites_f = None
        self.usar_caja = True
        self.altura = None
        self.onda_detectada = False
        self.paso_corte = None
        self.lista_resultados = None


class StateManager:
    """
    Gestor de la máquina de estados para el análisis LAMMPS.

    Coordina la transición entre estados, ejecuta el handler correspondiente
    a cada estado, y maneja la navegación hacia adelante/atrás.

    Attributes:
        estado_actual (Estado): El estado actual de la máquina.
        contexto (Contexto): El contexto compartido entre estados.
        input_handler (InputHandler): Manejador de entradas de usuario.
        handlers (dict[Estado, Callable]): Mapeo de estados a sus handlers.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Inicializa la máquina de estados.

        Args:
            config_path (Path | None): Ruta al archivo config.yaml.
                                       Si es None, busca en el directorio actual.
        """
        self.estado_actual = Estado.SELECCION_CARPETAS
        self.contexto = Contexto()
        self.input_handler = InputHandler()

        # Cargar configuración inicial
        if config_path is None:
            config_path = Path("./config.yaml")

        self.contexto.config = self.cargar_config(config_path)

        # Mapeo de estados a handlers
        self.handlers: dict[Estado, Callable[[], Estado | None]] = {
            Estado.SELECCION_CARPETAS: self.handler_seleccion_carpetas,
            Estado.CARGAR_CONFIG: self.handler_cargar_config,
            Estado.CONFIGURACION_RADIO: self.handler_configuracion_radio,
            Estado.CONFIGURACION_ONDA: self.handler_configuracion_onda,
            Estado.CONFIGURACION_FILTROS: self.handler_configuracion_filtros,
            Estado.VERIFICACION_PREVIA: self.handler_verificacion_previa,
            Estado.EJECUCION_FASE1: self.handler_ejecucion_fase1,
            Estado.EJECUCION_FASE2: self.handler_ejecucion_fase2,
            Estado.EXPORTACION: self.handler_exportacion,
            Estado.COMPLETADO: self.handler_completado,
        }

    def cargar_config(self, config_path: Path) -> dict[str, Any]:
        """
        Carga la configuración desde un archivo YAML.

        Args:
            config_path (Path): Ruta al archivo config.yaml.

        Returns:
            dict[str, Any]: Configuración cargada.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            yaml.YAMLError: Si hay un error de sintaxis en el YAML.
        """
        if not config_path.exists():
            print(f"Advertencia: No se encontró {config_path}")
            print("Usando configuración por defecto.")
            return self.config_por_defecto()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def config_por_defecto(self) -> dict[str, Any]:
        """
        Retorna una configuración por defecto si no hay config.yaml.

        Returns:
            dict[str, Any]: Configuración por defecto.
        """
        return {
            "defaults": {"radio_um": 0.76, "hacer_onda": False, "usar_caja": True},
            "analisis_onda": {
                "v_objetivo": -50.0,
                "tolerancia_v": 0.1,
                "np_criticos": [1, 3],
                "move_referencia": [0, 0, 0],
                "tolerancia_move": 1e-6,
            },
            "filtros": {
                "umbral_vz_min": 0.0,
                "umbral_energia_min": 0.0,
                "umbral_desplazamiento_lateral": 0.0,
                "umbral_techo_radios": 5,
            },
            "material": {
                "nombre": "Silice",
                "densidad": 2200.0,
                "modulo_young": 7.0e10,
                "coeficiente_poisson": 0.17,
                "energia_superficie": 0.032,
            },
            "jkr": {
                "usar_modelo": True,
                "cutoff_coordinacion": None,
                "factor_cutoff": 1.5,
            },
            "onda": {
                "umbral_v2_min": 0.0004,
                "umbral_v2_max": 0.04,
                "particulas_criticas": 3000,
                "umbral_techo": 3,
                "umbral_pared": 20,
            },
            "rutas": {
                "resultados": "./resultados",
                "imagenes": "./resultados/Imagenes",
                "config": "./config.yaml",
            },
            "metadatos": {
                "version": "2.0.0",
                "autor": "Tesis",
                "incluir_en_parquet": True,
                "incluir_en_plot": True,
            },
        }

    def ejecutar(self) -> None:
        """
        Ejecuta la máquina de estados hasta completar el análisis.

        El bucle principal continúa hasta que se alcanza el estado COMPLETADO
        o el usuario sale explícitamente.
        """
        while self.estado_actual < Estado.COMPLETADO:
            handler = self.handlers.get(self.estado_actual)

            if handler is None:
                print(f"Error: No hay handler para el estado {self.estado_actual}")
                break

            try:
                siguiente_estado = handler()

                if siguiente_estado is not None:
                    self.estado_actual = siguiente_estado
                else:
                    self.estado_actual += 1

            except KeyboardInterrupt:
                print("\n\nInterrupción detectada.")
                if self.confirmar_salida():
                    print("Saliendo del programa...")
                    sys.exit(0)

        print("\n" + "=" * 50)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 50)

    def confirmar_salida(self) -> bool:
        """
        Solicita confirmación al usuario para salir del programa.

        Returns:
            bool: True si el usuario confirma la salida.
        """
        confirmacion = input("¿Está seguro que desea salir? (Y/N): ").strip().lower()
        return confirmacion in ["y", "yes", "s", "si"]

    # -------------------------------------------------------------------------
    # HANDLERS DE CADA ESTADO
    # -------------------------------------------------------------------------

    def handler_seleccion_carpetas(self) -> Estado | None:
        """
        Handler para el estado de selección de carpetas.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        from utils import buscar_carpeta

        limpiar_pantalla()
        print("=" * 50)
        print("SELECCIÓN DE CARPETAS")
        print("=" * 50)
        print("\nSeleccione las carpetas de simulaciones a procesar.")
        print("Use 'b' para retroceder, 'q' para salir.\n")

        rutas = buscar_carpeta()

        if rutas is None:
            return Estado.COMPLETADO  # Usuario salió

        if len(rutas) == 0:
            print("No se seleccionaron carpetas. Intente nuevamente.")
            return None

        self.contexto.rutas = rutas
        print(f"\n✓ {len(rutas)} carpeta(s) seleccionada(s)")
        return None

    def handler_cargar_config(self) -> Estado | None:
        """
        Handler para el estado de carga de configuración YAML.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        # La configuración ya se cargó en __init__
        # Este estado es principalmente informativo
        config = self.contexto.config

        if config:
            print("\n" + "=" * 50)
            print("CONFIGURACIÓN CARGADA")
            print("=" * 50)
            print(f"Versión: {config.get('metadatos', {}).get('version', 'N/A')}")
            print(
                f"Radio por defecto: {config.get('defaults', {}).get('radio_um', 'N/A')} µm"
            )
            print(f"Material: {config.get('material', {}).get('nombre', 'N/A')}")
            print("=" * 50)

        return None

    def handler_configuracion_radio(self) -> Estado | None:
        """
        Handler para el estado de configuración del radio.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        limpiar_pantalla()
        print("=" * 50)
        print("CONFIGURACIÓN DEL RADIO")
        print("=" * 50)

        default = self.contexto.config.get("defaults", {}).get("radio_um", 0.76)

        print("\nIngrese el radio de las partículas en micrómetros.")
        print(f"Presione Enter para usar el valor por defecto: {default} µm\n")

        while True:
            radio, signal = self.input_handler.input_numerico(
                f"Radio [µm] (default: {default}): ",
                minimo=0.01,
                maximo=1000.0,
                default=default,
            )

            if signal == "back":
                return Estado.SELECCION_CARPETAS
            elif signal == "quit":
                return Estado.COMPLETADO

            if radio is not None:
                self.contexto.R = radio * 1e-6  # Convertir a metros
                print(f"\n✓ Radio configurado: {radio} µm ({self.contexto.R} m)")
                break

        return None

    def handler_configuracion_onda(self) -> Estado | None:
        """
        Handler para el estado de configuración del análisis de onda.

        Determina la configuración GLOBAL de análisis de onda.
        La evaluación de criticidad individual se hace por cada simulación
        en el loop de procesamiento (data_extractor.py).

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        limpiar_pantalla()
        print("=" * 50)
        print("CONFIGURACIÓN DEL ANÁLISIS DE ONDA")
        print("=" * 50)

        # Determinar el default desde configuración global
        default_hacer_onda = self.contexto.config.get("defaults", {}).get(
            "hacer_onda", False
        )

        print("\nEl análisis de onda detecta impactos macroscópicos en la cama.")
        print("Este análisis es más rápido pero consume más memoria.")
        print("\nNota: Para simulaciones críticas (v=-50 m/s, np=1 o 3, move!=0),")
        print("      el análisis de onda se recomendará automáticamente.\n")

        while True:
            hacer_onda, signal = self.input_handler.safe_input(
                "¿Desea realizar el análisis de onda para TODAS las simulaciones? (Y/N): ",
                tipo=bool,
                default=default_hacer_onda,
            )

            if signal == "back":
                return Estado.CONFIGURACION_RADIO
            elif signal == "quit":
                return Estado.COMPLETADO

            if hacer_onda is not None:
                self.contexto.hacer_onda = hacer_onda
                estado_str = "SÍ" if hacer_onda else "NO"
                print(
                    f"\n✓ Análisis de onda: {estado_str} (para todas las simulaciones)"
                )
                break

        return None

    def handler_configuracion_filtros(self) -> Estado | None:
        """
        Handler para el estado de configuración de umbrales de filtrado.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        from utils import configurar_umbrales_filtro

        limpiar_pantalla()
        print("=" * 50)
        print("CONFIGURACIÓN DE UMBRALES DE FILTRADO")
        print("=" * 50)
        print(
            "\nEstos umbrales definen el criterio para detectar partículas erosionadas."
        )
        print("Presione Enter para usar los valores por defecto.\n")

        umbrales = configurar_umbrales_filtro(self.contexto.config)
        self.contexto.umbrales_filtro = umbrales

        print("\n✓ Umbrales configurados:")
        print(f"  - vz_min: {umbrales['umbral_vz_min']} m/s")
        print(f"  - energia_min: {umbrales['umbral_energia_min']} m²/s²")
        print(
            f"  - desplazamiento_lateral: {umbrales['umbral_desplazamiento_lateral']} m"
        )

        return None

    def handler_verificacion_previa(self) -> Estado | None:
        """
        Handler para el estado de verificación de análisis previos.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """

        # Este estado se ejecuta dentro del loop de procesamiento de rutas
        # La lógica real está en data_extractor.py
        return None

    def handler_ejecucion_fase1(self) -> Estado | None:
        """
        Handler para el estado de ejecución de la Fase 1 (detección de onda).

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        # La ejecución real está en data_extractor.py
        # Este handler es un placeholder para la máquina de estados
        return None

    def handler_ejecucion_fase2(self) -> Estado | None:
        """
        Handler para el estado de ejecución de la Fase 2 (análisis OVITO).

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        # La ejecución real está en data_extractor.py
        return None

    def handler_exportacion(self) -> Estado | None:
        """
        Handler para el estado de exportación de resultados.

        Returns:
            Estado | None: El siguiente estado o None para continuar.
        """
        # La exportación real está en data_extractor.py
        return Estado.COMPLETADO

    def handler_completado(self) -> Estado | None:
        """
        Handler para el estado final de completado.

        Returns:
            Estado | None: None para terminar el bucle.
        """
        print("\n" + "=" * 50)
        print("ANÁLISIS COMPLETADO")
        print("=" * 50)
        return None


def es_simulacion_critica(metadatos: dict[str, Any], config: dict[str, Any]) -> bool:
    """
    Evalúa si una simulación cumple con las condiciones físicas críticas
    que requieren análisis de onda recomendado.

    Condiciones:
    1. Velocidad: |vel0 - v_objetivo| < tolerancia_v
    2. Proyectil (np): np debe estar en [1, 3]
    3. Estado de Movimiento: ||move - move_referencia|| > tolerancia_move

    Args:
        metadatos (dict[str, Any]): Diccionario con los parámetros extraídos
                                    del nombre de la carpeta (vel0, np, posx,
                                    posy, posz).
        config (dict[str, Any]): Configuración cargada desde config.yaml.

    Returns:
        bool: True si la simulación es crítica, False en caso contrario.

    Example:
        >>> metadatos = {"vel0": "-50.0", "np": "1", "posx": "0.1",
        ...              "posy": "0", "posz": "0"}
        >>> config = {"analisis_onda": {"v_objetivo": -50.0,
        ...                             "tolerancia_v": 0.1,
        ...                             "np_criticos": [1, 3],
        ...                             "move_referencia": [0, 0, 0],
        ...                             "tolerancia_move": 1e-6}}
        >>> es_simulacion_critica(metadatos, config)
        True
    """
    params = config.get("analisis_onda", {})

    # 1. Verificación de velocidad con tolerancia
    try:
        vel0 = float(metadatos.get("vel0", 0))
    except (ValueError, TypeError):
        return False

    v_objetivo = params.get("v_objetivo", -50.0)
    tolerancia_v = params.get("tolerancia_v", 0.1)

    if abs(vel0 - v_objetivo) > tolerancia_v:
        return False

    # 2. Verificación de cantidad de partículas del proyectil
    try:
        np_val = int(metadatos.get("np", 0))
    except (ValueError, TypeError):
        return False

    np_criticos = params.get("np_criticos", [1, 3])

    if np_val not in np_criticos:
        return False

    # 3. Verificación del vector de movimiento (norma euclidiana)
    try:
        move = [
            float(metadatos.get("posx", 0)),
            float(metadatos.get("posy", 0)),
            float(metadatos.get("posz", 0)),
        ]
    except (ValueError, TypeError):
        return False

    move_ref = params.get("move_referencia", [0, 0, 0])
    tolerancia_move = params.get("tolerancia_move", 1e-6)

    norma_desplazamiento = math.sqrt(
        (move[0] - move_ref[0]) ** 2
        + (move[1] - move_ref[1]) ** 2
        + (move[2] - move_ref[2]) ** 2
    )

    if norma_desplazamiento <= tolerancia_move:
        return False

    # Todas las condiciones se cumplieron
    return True
