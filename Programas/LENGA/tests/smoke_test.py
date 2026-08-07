"""Smoke tests headless para LENGA+.

Verifica sin display que la aplicación compila, que la ventana principal se
instancia con sus 5 pestañas, que el diálogo de login muestra placeholders
transparentes (HintLineEdit) y que el chequeo de datos reporta la estructura
esperada.

Uso:
    QT_QPA_PLATFORM=offscreen python3 tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from PyQt5.QtWidgets import QApplication, QMessageBox

# Evitar que los diálogos modales bloqueen el test headless
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)

from lenga.gui.login_dialog import HintLineEdit, LoginDialog
from lenga.gui.main_window import MainWindow

EXPECTED_TABS = [
    "Cluster y Nodos",
    "Análisis CARBOXI",
    "Descarga SCP",
    "Generador de Figuras",
    "Generar Tablas",
]


class LengaSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_instantiates(self):
        window = MainWindow(dry_run=True)
        self.assertEqual(window.windowTitle(), "LENGA+ — Plataforma de Simulación y Análisis de Medios Granulares")
        tab_labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        self.assertEqual(tab_labels, EXPECTED_TABS)
        self.assertTrue(hasattr(window, "btn_about"))

    def test_login_hint_line_edit(self):
        hint = HintLineEdit(placeholder="Ejemplo: cluster.dominio.com")
        hint.setText("")
        self.assertEqual(hint.text(), "")
        self.assertEqual(hint.placeholderText(), "Ejemplo: cluster.dominio.com")
        # El placeholder debe renderizarse sin errores en modo offscreen
        pixmap = hint.grab()
        self.assertFalse(pixmap.isNull())
        # Con texto real, no debe mostrar el hint
        hint.setText("usuario@cluster")
        self.assertEqual(hint.text(), "usuario@cluster")

    def test_data_check_report(self):
        from lenga.data_check import build_warning_message, check_local_datasets

        report = check_local_datasets()
        self.assertIn("9_mov", report)
        self.assertIn("49_mov", report)
        for name, entry in report.items():
            self.assertIn("available", entry)
            self.assertIn("csv_ok", entry)
            self.assertIn("cache_ok", entry)
            self.assertIsInstance(build_warning_message(report), str)

    def test_tables_engine_catalog(self):
        from lenga.tables_generator import TablesGeneratorEngine

        engine = TablesGeneratorEngine(output_dir=Path(__file__).resolve().parent.parent / "tablas")
        self.assertIn("9_mov", engine.TABLES_FOR_DATASET)
        self.assertIn("49_mov", engine.TABLES_FOR_DATASET)
        self.assertGreater(len(engine.TABLES_FOR_DATASET["9_mov"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
