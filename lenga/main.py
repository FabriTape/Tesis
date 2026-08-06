"""
Punto de entrada principal para iniciar la aplicación LENGA GUI.

Uso:
    python3 -m lenga.main
    python3 -m lenga.main --dry-run
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from lenga.gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="LENGA: Plataforma Unificada de Simulación y Análisis Granular.")
    parser.add_argument("--dry-run", "--mock", action="store_true", help="Inicia la aplicación en modo simulación (Mock local).")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("LENGA Granular Physics Platform")
    app.setOrganizationName("Universidad de Chile / Tesis")

    window = MainWindow(dry_run=args.dry_run)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
