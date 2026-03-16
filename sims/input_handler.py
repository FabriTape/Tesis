"""
Manejador de entradas de usuario con validación y comandos globales.

Este módulo proporciona un wrapper seguro para la función input() que permite:
- Validación de tipos (str, int, float, bool)
- Valores por defecto cuando el usuario presiona Enter
- Comandos globales 'b' (back) y 'q' (quit) con confirmación
- Manejo robusto de interrupciones (KeyboardInterrupt)
"""

import sys
from typing import Any, Literal, Type, TypeVar

T = TypeVar("T")


class InputHandler:
    """
    Manejador de entradas de usuario con validación y comandos globales.

    Proporciona un método unificado para solicitar entrada al usuario con
    validación de tipos, valores por defecto, y comandos de navegación.

    Attributes:
        allow_back (bool): Si True, permite el comando 'b' para retroceder.
        allow_quit (bool): Si True, permite el comando 'q' para salir.

    Example:
        >>> handler = InputHandler()
        >>> radio, signal = handler.safe_input(
        ...     "Radio [um]: ", tipo=float, default=0.76
        ... )
        >>> if signal == 'back':
        ...     # Retroceder al estado anterior
        ...     pass
    """

    def __init__(self, allow_back: bool = True, allow_quit: bool = True) -> None:
        """
        Inicializa el manejador de entradas.

        Args:
            allow_back (bool): Si True, permite el comando 'b' para retroceder.
                               Por defecto es True.
            allow_quit (bool): Si True, permite el comando 'q' para salir.
                               Por defecto es True.
        """
        self.allow_back = allow_back
        self.allow_quit = allow_quit

    def safe_input(
        self,
        prompt: str,
        tipo: Type[T] = str,
        default: Any = None,
        permitir_back: bool | None = None,
        permitir_quit: bool | None = None,
    ) -> tuple[T | None, Literal["continue", "back", "quit"]]:
        """
        Wrapper seguro para input() con validación de tipos y comandos globales.

        Args:
            prompt (str): Mensaje a mostrar al usuario.
            tipo (Type[T]): Tipo de dato esperado (str, int, float, bool).
            default (Any): Valor por defecto si el usuario presiona Enter.
            permitir_back (bool | None): Sobrescribe allow_back para esta llamada.
            permitir_quit (bool | None): Sobrescribe allow_quit para esta llamada.

        Returns:
            tuple[T | None, Literal["continue", "back", "quit"]]:
                - valor: El valor ingresado y validado, o None si se seleccionó
                         'back' o 'quit'.
                - señal: 'continue' para continuar normal, 'back' para retroceder,
                         'quit' para salir del programa.

        Raises:
            SystemExit: Si el usuario confirma la salida con 'q'.

        Example:
            >>> handler = InputHandler()
            >>> # Para booleanos con default dinámico
            >>> valor, señal = handler.safe_input(
            ...     "¿Realizar análisis? (Y/N): ",
            ...     tipo=bool,
            ...     default=True
            ... )
            >>>
            >>> # Para floats con default
            >>> radio, señal = handler.safe_input(
            ...     "Radio [um] (default: 0.76): ",
            ...     tipo=float,
            ...     default=0.76
            ... )
        """
        allow_back = permitir_back if permitir_back is not None else self.allow_back
        allow_quit = permitir_quit if permitir_quit is not None else self.allow_quit

        while True:
            try:
                entrada = input(prompt).strip()

                # Comando Back
                if allow_back and entrada.lower() == "b":
                    return None, "back"

                # Comando Quit
                if allow_quit and entrada.lower() == "q":
                    if self.confirmar_salida():
                        print("Saliendo del programa. ¡Hasta luego!")
                        sys.exit(0)
                    continue

                # Valor por defecto si está vacío
                if entrada == "" and default is not None:
                    return default, "continue"

                # Conversión de tipo
                if tipo is bool:
                    if entrada.lower() in ["y", "yes", "s", "si", "true", "1"]:
                        return True, "continue"
                    elif entrada.lower() in ["n", "no", "false", "0", ""]:
                        return False, "continue"
                    else:
                        print("Error: ingrese Y/N, True/False, o 1/0")
                        continue
                elif tipo is int:
                    return int(entrada), "continue"
                elif tipo is float:
                    return float(entrada), "continue"
                else:
                    return entrada, "continue"  # type: ignore[return-value]

            except ValueError:
                print(f"Error: ingrese un valor válido de tipo {tipo.__name__}")
            except KeyboardInterrupt:
                print("\n\nInterrupción detectada.")
                if allow_quit:
                    print("Saliendo del programa...")
                    sys.exit(0)
                return None, "quit"

    def confirmar_salida(self, intentos: int = 0) -> bool:
        """
        Solicita confirmación al usuario para salir del programa.

        Args:
            intentos (int): Número de intentos previos fallidos.

        Returns:
            bool: True si el usuario confirma la salida, False en caso contrario.
        """
        if intentos >= 2:
            print("Salida forzada. Demasiados intentos inválidos.")
            return True

        confirmacion = input("¿Está seguro que desea salir? (Y/N): ").strip().lower()

        if confirmacion in ["y", "yes", "s", "si"]:
            return True
        elif confirmacion in ["n", "no", ""]:
            return False
        else:
            print("Entrada inválida. Por favor ingrese Y o N.")
            return self.confirmar_salida(intentos + 1)

    def input_numerico(
        self,
        prompt: str,
        minimo: float | None = None,
        maximo: float | None = None,
        default: float | None = None,
    ) -> tuple[float | None, Literal["continue", "back", "quit"]]:
        """
        Solicita un valor numérico con validación de rango opcional.

        Args:
            prompt (str): Mensaje a mostrar al usuario.
            minimo (float | None): Valor mínimo permitido.
            maximo (float | None): Valor máximo permitido.
            default (float | None): Valor por defecto si el usuario presiona Enter.

        Returns:
            tuple[float | None, Literal["continue", "back", "quit"]]:
                - valor: El valor numérico validado, o None si se seleccionó
                         'back' o 'quit'.
                - señal: 'continue', 'back', o 'quit'.

        Example:
            >>> handler = InputHandler()
            >>> radio, _ = handler.input_numerico(
            ...     "Radio [um]: ", minimo=0.1, maximo=100.0, default=0.76
            ... )
        """
        while True:
            valor, signal = self.safe_input(prompt, tipo=float, default=default)

            if signal != "continue" or valor is None:
                return valor, signal

            # Validación de rango
            if minimo is not None and valor < minimo:
                print(f"Error: el valor debe ser mayor o igual a {minimo}")
                continue

            if maximo is not None and valor > maximo:
                print(f"Error: el valor debe ser menor o igual a {maximo}")
                continue

            return valor, signal

    def input_opcion_multiple(
        self,
        prompt: str,
        opciones: list[str],
        default: str | None = None,
    ) -> tuple[str | None, Literal["continue", "back", "quit"]]:
        """
        Solicita al usuario seleccionar una opción de una lista predefinida.

        Args:
            prompt (str): Mensaje a mostrar al usuario.
            opciones (list[str]): Lista de opciones válidas.
            default (str | None): Opción por defecto si el usuario presiona Enter.

        Returns:
            tuple[str | None, Literal["continue", "back", "quit"]]:
                - valor: La opción seleccionada, o None si se seleccionó
                         'back' o 'quit'.
                - señal: 'continue', 'back', o 'quit'.

        Example:
            >>> handler = InputHandler()
            >>> opcion, _ = handler.input_opcion_multiple(
            ...     "Tipo de análisis: ",
            ...     opciones=["completo", "rapido", "personalizado"],
            ...     default="completo"
            ... )
        """
        opciones_lower = [o.lower() for o in opciones]
        opciones_str = "/".join(opciones)

        while True:
            valor, signal = self.safe_input(
                f"{prompt} ({opciones_str}): ", tipo=str, default=default
            )

            if signal != "continue" or valor is None:
                return valor, signal

            if valor.lower() not in opciones_lower:
                print(f"Error: opción inválida. Las opciones son: {opciones_str}")
                continue

            return valor, signal


# Instancia global para uso conveniente
input_handler = InputHandler()


def safe_input(
    prompt: str,
    tipo: type = str,
    default: Any = None,
    permitir_back: bool = True,
    permitir_quit: bool = True,
) -> tuple[Any | None, Literal["continue", "back", "quit"]]:
    """
    Función conveniente para usar el input handler global.

    Args:
        prompt (str): Mensaje a mostrar al usuario.
        tipo (type): Tipo de dato esperado (str, int, float, bool).
        default (Any): Valor por defecto si el usuario presiona Enter.
        permitir_back (bool): Si True, permite el comando 'b' para retroceder.
        permitir_quit (bool): Si True, permite el comando 'q' para salir.

    Returns:
        tuple[Any | None, Literal["continue", "back", "quit"]]:
            - valor: El valor ingresado y validado, o None.
            - señal: 'continue', 'back', o 'quit'.
    """
    handler = InputHandler(allow_back=permitir_back, allow_quit=permitir_quit)
    return handler.safe_input(prompt, tipo=tipo, default=default)
