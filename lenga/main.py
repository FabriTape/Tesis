"""Módulo de inicio y punto de entrada principal para LENGA.

Este módulo inicializa el bucle de eventos de la aplicación Qt (QApplication),
parsea las opciones de línea de comandos y despliega la interfaz principal.

Uso:
    python3 -m lenga.main
    python3 -m lenga.main --dry-run
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from lenga.gui.main_window import MainWindow


def main() -> None:
    """Punto de entrada principal para la aplicación LENGA GUI.

    Parsea los argumentos por consola (--dry-run / --mock), instancia el objeto
    QApplication de PyQt5, configura metadatos del sistema y despliega la
    ventana principal MainWindow.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: Al finalizar la ejecución del bucle principal de la GUI.
    """
    # 1. Configurar parser de argumentos de consola
    parser = argparse.ArgumentParser(
        description="LENGA: Plataforma Unificada de Simulación y Análisis de Medios Granulares."
    )
    parser.add_argument(
        "--dry-run",
        "--mock",
        action="store_true",
        help="Inicia la aplicación en modo simulación (Mock local sin requerir cluster SSH)."
    )
    args = parser.parse_args()

    # 2. Inicializar bucle de eventos QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("LENGA Granular Physics Platform")
    app.setOrganizationName("Universidad de Chile / Tesis Granular")

    # 3. Instanciar y desplegar ventana principal (MainWindow)
    window = MainWindow(dry_run=args.dry_run)
    window.show()

    # 4. Entrar al bucle de eventos de la GUI
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
