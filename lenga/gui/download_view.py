"""
Vista / Pestaña de Descarga SCP de Resultados en LENGA.
Transfiere archivos y carpetas consolidadas desde el cluster hacia LENGA/datos/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lenga.ssh_client import SSHClientManager


class DownloadWorker(QThread):
    """Thread worker para descarga SCP en segundo plano."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, ssh_client: SSHClientManager, remote_path: str, local_path: str):
        super().__init__()
        self.ssh = ssh_client
        self.remote_path = remote_path
        self.local_path = local_path

    def run(self):
        def _callback(msg: str):
            self.progress_signal.emit(msg)

        success, msg = self.ssh.download_file(self.remote_path, self.local_path, progress_callback=_callback)
        self.finished_signal.emit(success, msg)


class DownloadViewWidget(QWidget):
    """Panel para gestionar la descarga mediante SCP hacia la carpeta local LENGA/datos/."""

    def __init__(self, ssh_client: SSHClientManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ssh = ssh_client
        self.local_dir = Path(__file__).resolve().parent.parent.parent / "datos"
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Etapa 3: Descarga SCP de Archivos desde el Cluster")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Config Layout
        grp_download = QGroupBox("Configuración de Transferencia SCP")
        v_down = QVBoxLayout(grp_download)

        h_remote = QHBoxLayout()
        self.txt_remote = QLineEdit("/home/fbazzoni/resultados/merge_resultados.csv")
        self.txt_remote.setPlaceholderText("Ruta remota del archivo o carpeta .csv")
        h_remote.addWidget(QLabel("Ruta Remota:"))
        h_remote.addWidget(self.txt_remote)
        v_down.addLayout(h_remote)

        h_local = QHBoxLayout()
        self.txt_local = QLineEdit(str(self.local_dir / "merge_resultados.csv"))
        self.btn_browse = QPushButton("Examinar...")
        self.btn_browse.clicked.connect(self._browse_local_path)
        h_local.addWidget(QLabel("Destino Local:"))
        h_local.addWidget(self.txt_local)
        h_local.addWidget(self.btn_browse)
        v_down.addLayout(h_local)

        self.btn_download = QPushButton("📥 Descargar Archivos con SCP")
        self.btn_download.setObjectName("PrimaryButton")
        self.btn_download.clicked.connect(self._start_download)
        v_down.addWidget(self.btn_download)

        layout.addWidget(grp_download)

        # Status Label
        self.lbl_status = QLabel("Listo para iniciar transferencia SCP.")
        self.lbl_status.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.lbl_status)

        # Local files explorer table
        grp_local_files = QGroupBox("Archivos de Datos Descargados (`LENGA/datos/`)")
        v_files = QVBoxLayout(grp_local_files)

        self.tbl_files = QTableWidget()
        self.tbl_files.setColumnCount(3)
        self.tbl_files.setHorizontalHeaderLabels(["Nombre de Archivo", "Tamaño", "Última Modificación"])
        self.tbl_files.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v_files.addWidget(self.tbl_files)

        layout.addWidget(grp_local_files)

        self.refresh_local_files()

    def _browse_local_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Seleccionar ubicación de destino local", str(self.local_dir), "Archivos CSV (*.csv);;Todos los Archivos (*)"
        )
        if file_path:
            self.txt_local.setText(file_path)

    def _start_download(self):
        remote_path = self.txt_remote.text().strip()
        local_path = self.txt_local.text().strip()

        if not remote_path or not local_path:
            QMessageBox.warning(self, "Campos Vacíos", "Especifique las rutas remota y local.")
            return

        self.btn_download.setEnabled(False)
        self.lbl_status.setText("Iniciando transferencia SCP...")
        self.lbl_status.setStyleSheet("color: #89b4fa; font-weight: bold;")

        self.worker = DownloadWorker(self.ssh, remote_path, local_path)
        self.worker.progress_signal.connect(lambda msg: self.lbl_status.setText(msg))
        self.worker.finished_signal.connect(self._on_download_finished)
        self.worker.start()

    def _on_download_finished(self, success: bool, msg: str):
        self.btn_download.setEnabled(True)
        if success:
            self.lbl_status.setText(f"✓ {msg}")
            self.lbl_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            QMessageBox.information(self, "Descarga Completa", msg)
            self.refresh_local_files()
        else:
            self.lbl_status.setText(f"❌ {msg}")
            self.lbl_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
            QMessageBox.critical(self, "Fallo SCP", msg)

    def refresh_local_files(self):
        """Lista los archivos descargados en el directorio datos/."""
        if not self.local_dir.exists():
            return

        files = list(self.local_dir.glob("*"))
        self.tbl_files.setRowCount(len(files))
        for row, fpath in enumerate(files):
            size_kb = f"{fpath.stat().st_size / 1024:.1f} KB" if fpath.is_file() else "[Carpeta]"
            import time
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fpath.stat().st_mtime))

            self.tbl_files.setItem(row, 0, QTableWidgetItem(fpath.name))
            self.tbl_files.setItem(row, 1, QTableWidgetItem(size_kb))
            self.tbl_files.setItem(row, 2, QTableWidgetItem(mtime))
