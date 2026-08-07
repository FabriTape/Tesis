"""
Vista / Pestaña del Análisis de Resultados con CARBOXI y Consolidación CSV en LENGA.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lenga.carboxi_runner import CarboxiRunner
from lenga.gui.remote_browse_dialog import RemoteBrowseDialog
from lenga.ssh_client import SSHClientManager


class AnalysisWorker(QThread):
    """Thread worker para ejecutar CARBOXI y merge_all en segundo plano."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, runner: CarboxiRunner, remote_dir: str, node: str):
        super().__init__()
        self.runner = runner
        self.remote_dir = remote_dir
        self.node = node

    def run(self):
        def _callback(msg: str):
            self.log_signal.emit(msg)

        success, res_msg = self.runner.run_analysis_pipeline(
            remote_base_dir=self.remote_dir,
            target_node=self.node,
            progress_callback=_callback
        )
        self.finished_signal.emit(success, res_msg)


class AnalysisViewWidget(QWidget):
    """Panel visual para lanzar el análisis CARBOXI y consolidación en el cluster."""

    def __init__(self, carboxi_runner: CarboxiRunner, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.runner = carboxi_runner

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Etapa 2: Análisis Remoto de Simulaciones Granulares con CARBOXI")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Config Panel
        grp_config = QGroupBox("Parámetros de Ejecución Remota")
        h_config = QHBoxLayout(grp_config)

        self.txt_remote_dir = QLineEdit("~")
        self.txt_remote_dir.setPlaceholderText("Directorio base en el cluster")
        self.btn_browse_remote = QPushButton("Examinar remoto...")
        self.btn_browse_remote.clicked.connect(self._browse_remote_dir)

        self.txt_node = QLineEdit("gpu2")
        self.txt_node.setPlaceholderText("Nodo para ejecutar el análisis (ej: gpu2)")

        h_config.addWidget(QLabel("Directorio Remoto:"))
        h_config.addWidget(self.txt_remote_dir)
        h_config.addWidget(self.btn_browse_remote)
        h_config.addWidget(QLabel("Nodo Objetivo:"))
        h_config.addWidget(self.txt_node)

        layout.addWidget(grp_config)

        # Buttons
        h_btns = QHBoxLayout()
        self.btn_run_full = QPushButton("Ejecutar Análisis CARBOXI Completo + Merge")
        self.btn_run_full.setObjectName("PrimaryButton")
        self.btn_run_full.clicked.connect(self._start_full_analysis)

        self.btn_merge_only = QPushButton("Consolidar CSV (Solo merge_all)")
        self.btn_merge_only.clicked.connect(self._start_merge_only)

        h_btns.addWidget(self.btn_run_full)
        h_btns.addWidget(self.btn_merge_only)
        layout.addLayout(h_btns)

        # Console Output Box
        grp_console = QGroupBox("Consola de Logs en Tiempo Real")
        v_console = QVBoxLayout(grp_console)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: monospace; background-color: #11111b; color: #a6e3a1;")
        v_console.addWidget(self.txt_log)

        layout.addWidget(grp_console)

    def log(self, text: str):
        self.txt_log.append(f"> {text}")

    def _browse_remote_dir(self):
        ssh = self.runner.ssh
        if not ssh.is_connected and not ssh.mock_mode:
            QMessageBox.warning(self, "Sin Conexión SSH", "Inicie sesión SSH para examinar directorios remotos.")
            return
        if ssh.mock_mode:
            self.log("Exploración no disponible en modo simulación.")
            QMessageBox.information(
                self, "Modo Simulación",
                "Para examinar directorios remotos es necesario conectarse a un cluster real.\n"
                "El modo simulación (mock) no puede listar carpetas remotas.",
            )
            return

        dlg = RemoteBrowseDialog(
            ssh,
            start_path=self.txt_remote_dir.text().strip() or "~",
            select_files=False,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            remote_dir = dlg.selected_remote_path()
            if remote_dir:
                self.txt_remote_dir.setText(remote_dir)
                self.log(f"Directorio remoto seleccionado: {remote_dir}")

    def _start_full_analysis(self):
        remote_dir = self.txt_remote_dir.text().strip()
        node = self.txt_node.text().strip()

        self.txt_log.clear()
        self.log(f"Iniciando análisis CARBOXI remoto en {node}:{remote_dir}...")
        self.btn_run_full.setEnabled(False)
        self.btn_merge_only.setEnabled(False)

        self.worker = AnalysisWorker(self.runner, remote_dir, node)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self._on_analysis_finished)
        self.worker.start()

    def _on_analysis_finished(self, success: bool, msg: str):
        self.btn_run_full.setEnabled(True)
        self.btn_merge_only.setEnabled(True)

        if success:
            self.log("Pipeline finalizado exitosamente.")
            self.log(msg)
            QMessageBox.information(self, "Análisis Exitoso", "El análisis CARBOXI y la consolidación CSV finalizaron correctamente.")
        else:
            self.log(f"Error en la ejecución: {msg}")
            QMessageBox.critical(self, "Fallo en Análisis", f"Ocurrió un error: {msg}")

    def _start_merge_only(self):
        remote_dir = self.txt_remote_dir.text().strip()
        self.log("Ejecutando consolidación manual merge_all...")
        success, msg = self.runner.trigger_merge_only(remote_dir)
        if success:
            self.log("Merge consolidado.")
            self.log(msg)
            QMessageBox.information(self, "Merge Exitoso", msg)
        else:
            self.log(f"Error en merge: {msg}")
            QMessageBox.critical(self, "Fallo en Merge", msg)
