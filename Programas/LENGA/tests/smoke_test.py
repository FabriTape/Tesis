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

    def test_download_view_remote_sims(self):
        from lenga.gui.download_view import DownloadViewWidget
        from lenga.ssh_client import SSHClientManager

        ssh = SSHClientManager("cluster.dominio.com", 22, "usuario", "")
        ssh.is_connected = True
        view = DownloadViewWidget(ssh)

        fake_out = (
            "/home/usuario/cluster/scripts/Buffer/sims_SI_erosion_bed_100um__ff_0.35_vel_-50_"
            "move_0_NP_1_angle_0/erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0"
        )
        commands = []
        ssh.run_command = lambda cmd, timeout=60: commands.append(cmd) or (True, fake_out, "")
        resolved = view._resolve_remote_sim_dir("erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0")
        self.assertEqual(
            resolved,
            "/home/usuario/cluster/scripts/Buffer/sims_SI_erosion_bed_100um__ff_0.35_vel_-50_"
            "move_0_NP_1_angle_0/erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0",
        )
        # El comando usa $HOME (sin hardcodear usuario) y termina siempre en éxito
        self.assertIn('"$HOME"', commands[0])
        self.assertTrue(commands[0].rstrip().endswith("true"))

    def test_download_view_mock_explore(self):
        from lenga.gui.download_view import DownloadViewWidget
        from lenga.ssh_client import SSHClientManager

        ssh = SSHClientManager("mock", 22, "usuario", "")
        ssh.mock_mode = True
        view = DownloadViewWidget(ssh)
        resolved = view._resolve_remote_sim_dir("erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0")
        self.assertEqual(resolved, "./Buffer/erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0")

    def test_remote_browse_dialog(self):
        from lenga.gui.remote_browse_dialog import RemoteBrowseDialog
        from lenga.ssh_client import SSHClientManager

        ssh = SSHClientManager("cluster.dominio.com", 22, "usuario", "")
        ssh.is_connected = True

        def fake_run(cmd, timeout=60):
            if "echo $HOME" in cmd:
                return True, "/home/usuario\n", ""
            return True, "carpeta_sim/\narchivo.csv\n", ""

        ssh.run_command = fake_run
        dlg = RemoteBrowseDialog(ssh, start_path="~", select_files=True)
        self.assertEqual(dlg.current_path, "/home/usuario")
        texts = [dlg.lst.item(i).text() for i in range(dlg.lst.count())]
        self.assertIn("carpeta_sim/", texts)
        self.assertIn("archivo.csv", texts)

        dlg._on_click(dlg.lst.item(1))
        self.assertEqual(dlg.selected_remote_path(), "/home/usuario/archivo.csv")

        # Doble click entra en la carpeta
        dlg._on_double_click(dlg.lst.item(0))
        self.assertEqual(dlg.current_path, "/home/usuario/carpeta_sim")
        self.assertEqual(dlg.selected_remote_path(), "/home/usuario/carpeta_sim")

    def test_remote_browse_dialog_filters_ssh_warning(self):
        from lenga.gui.remote_browse_dialog import RemoteBrowseDialog
        from lenga.ssh_client import SSHClientManager

        ssh = SSHClientManager("cluster.dominio.com", 22, "usuario", "")
        ssh.is_connected = True

        def fake_run(cmd, timeout=60):
            if "echo $HOME" in cmd:
                return True, "/home/usuario\n", ""
            return (
                True,
                "Warning: Permanently added 'cluster.dominio.com' (ED25519) "
                "to the list of known hosts.\ncarpeta_sim/\n",
                "",
            )

        ssh.run_command = fake_run
        dlg = RemoteBrowseDialog(ssh, start_path="~")
        texts = [dlg.lst.item(i).text() for i in range(dlg.lst.count())]
        self.assertEqual(texts, ["carpeta_sim/"])

    def test_analysis_view_has_remote_browse(self):
        from lenga.gui.analysis_view import AnalysisViewWidget
        from lenga.ssh_client import SSHClientManager

        ssh = SSHClientManager("cluster.dominio.com", 22, "usuario", "")
        ssh.is_connected = True
        runner = object.__new__(type("FakeRunner", (), {}))
        runner.ssh = ssh
        view = AnalysisViewWidget(runner)
        self.assertEqual(view.btn_browse_remote.text(), "Examinar remoto...")
        self.assertEqual(view.txt_remote_dir.text(), "~")

    def test_local_browse_dialog(self):
        from lenga.gui.local_browse_dialog import LocalBrowseDialog

        tmp = Path(__file__).resolve().parent / "_tmp_browse_test"
        (tmp / "sub").mkdir(parents=True, exist_ok=True)
        (tmp / "dato.csv").write_text("a,b\n")

        dlg = LocalBrowseDialog(start_path=str(tmp), select_files=True)
        texts = [dlg.lst.item(i).text() for i in range(dlg.lst.count())]
        self.assertIn("sub/", texts)
        self.assertIn("dato.csv", texts)

        # Click en un archivo lo selecciona
        idx = texts.index("dato.csv")
        dlg._on_click(dlg.lst.item(idx))
        self.assertEqual(dlg.selected_local_path(), str(tmp / "dato.csv"))

        # Doble click en una carpeta navega
        idx_dir = texts.index("sub/")
        dlg._on_double_click(dlg.lst.item(idx_dir))
        self.assertEqual(dlg.current_path, str(tmp / "sub"))

        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
