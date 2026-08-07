"""Vista / Pestaña de Descarga SCP de Resultados en LENGA.

Soporta descarga de archivos consolidados (merge_resultados.csv) y la
descarga de simulaciones específicas individuales desde el cluster hacia LENGA/datos/.
"""

from __future__ import annotations

import os
import shlex
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
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
    """Panel para gestionar descargas mediante SCP globales y de simulaciones específicas."""

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

        title = QLabel("Etapa 3: Descarga SCP de Resultados y Simulaciones Específicas")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Tabs para alternar entre Descarga Global y Descarga de Simulación Específica
        self.download_tabs = QTabWidget()

        # ---------------------------------------------------------------------
        # PESTAÑA 1: DESCARGA GLOBAL (ARCHIVOS CONSOLIDADOS)
        # ---------------------------------------------------------------------
        tab_global = QWidget()
        v_global = QVBoxLayout(tab_global)

        grp_global = QGroupBox("Configuración de Transferencia SCP Global")
        v_g = QVBoxLayout(grp_global)

        h_remote = QHBoxLayout()
        self.txt_remote = QLineEdit("/home/fbazzoni/resultados/merge_resultados.csv")
        self.txt_remote.setPlaceholderText("Ruta remota del archivo o carpeta .csv")
        h_remote.addWidget(QLabel("Ruta Remota:"))
        h_remote.addWidget(self.txt_remote)
        v_g.addLayout(h_remote)

        h_local = QHBoxLayout()
        self.txt_local = QLineEdit(str(self.local_dir / "merge_resultados.csv"))
        self.btn_browse = QPushButton("Examinar...")
        self.btn_browse.clicked.connect(self._browse_local_path)
        h_local.addWidget(QLabel("Destino Local:"))
        h_local.addWidget(self.txt_local)
        h_local.addWidget(self.btn_browse)
        v_g.addLayout(h_local)

        self.btn_download = QPushButton("Descargar Archivo Consolidado SCP")
        self.btn_download.setObjectName("PrimaryButton")
        self.btn_download.clicked.connect(self._start_download_global)
        v_g.addWidget(self.btn_download)

        v_global.addWidget(grp_global)
        self.download_tabs.addTab(tab_global, "Descarga Consolidada (Global)")

        # ---------------------------------------------------------------------
        # PESTAÑA 2: DESCARGA DE SIMULACIÓN ESPECÍFICA (INDIVIDUAL)
        # ---------------------------------------------------------------------
        tab_specific = QWidget()
        v_specific = QVBoxLayout(tab_specific)

        grp_spec = QGroupBox("Descarga de Simulación Específica")
        v_s = QVBoxLayout(grp_spec)

        h_sim_id = QHBoxLayout()
        self.txt_sim_name = QLineEdit("erosion_bed_SI_ff_0.35_np_1_vel_-50_angle_0_move_0_0_0")
        self.txt_sim_name.setPlaceholderText("Ej: erosion_bed_SI_ff_0.35... o ID de simulación")
        self.btn_explore_sims = QPushButton("Explorar en Cluster")
        self.btn_explore_sims.clicked.connect(self._explore_remote_simulations)
        h_sim_id.addWidget(QLabel("ID / Carpeta Simulación:"))
        h_sim_id.addWidget(self.txt_sim_name)
        h_sim_id.addWidget(self.btn_explore_sims)
        v_s.addLayout(h_sim_id)

        h_component = QHBoxLayout()
        self.cbo_component = QComboBox()
        self.cbo_component.addItem("Carpeta Completa de la Simulación", "all")
        self.cbo_component.addItem("Solo Trayectoria (parquet / dump)", "trajectory")
        self.cbo_component.addItem("Solo Resultados y Metadata (.json / .toml / .csv)", "metadata")
        h_component.addWidget(QLabel("Componentes a Descargar:"))
        h_component.addWidget(self.cbo_component)
        v_s.addLayout(h_component)

        h_dest = QHBoxLayout()
        self.txt_spec_local = QLineEdit(str(self.local_dir))
        self.btn_browse_spec = QPushButton("Examinar...")
        self.btn_browse_spec.clicked.connect(self._browse_spec_local)
        h_dest.addWidget(QLabel("Carpeta Destino Local:"))
        h_dest.addWidget(self.txt_spec_local)
        h_dest.addWidget(self.btn_browse_spec)
        v_s.addLayout(h_dest)

        self.btn_download_spec = QPushButton("Descargar Simulación Específica")
        self.btn_download_spec.setObjectName("PrimaryButton")
        self.btn_download_spec.clicked.connect(self._start_download_specific)
        v_s.addWidget(self.btn_download_spec)

        v_specific.addWidget(grp_spec)
        self.download_tabs.addTab(tab_specific, "🎯 Simulación Específica")

        layout.addWidget(self.download_tabs)

        # Status Label
        self.lbl_status = QLabel("Listo para iniciar transferencia SCP.")
        self.lbl_status.setStyleSheet("color: #a6adc8; font-weight: 500;")
        layout.addWidget(self.lbl_status)

        # Local files explorer table
        grp_local_files = QGroupBox("Archivos y Simulaciones Descargadas (`LENGA/datos/`)")
        v_files = QVBoxLayout(grp_local_files)

        self.tbl_files = QTableWidget()
        self.tbl_files.setColumnCount(3)
        self.tbl_files.setHorizontalHeaderLabels(["Nombre de Archivo / Carpeta", "Tamaño", "Última Modificación"])
        self.tbl_files.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v_files.addWidget(self.tbl_files)

        layout.addWidget(grp_local_files)

        self.refresh_local_files()

    def _browse_local_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Seleccionar ubicación de destino local", str(self.local_dir), "Archivos CSV (*.csv);;Todos (*)"
        )
        if file_path:
            self.txt_local.setText(file_path)

    def _browse_spec_local(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino", str(self.local_dir))
        if dir_path:
            self.txt_spec_local.setText(dir_path)

    def _explore_remote_simulations(self):
        if not self.ssh.is_connected and not self.ssh.mock_mode:
            QMessageBox.warning(self, "Sin Conexión SSH", "Inicie sesión SSH para explorar simulaciones en el cluster.")
            return

        self.lbl_status.setText("Explorando carpetas de simulaciones en el cluster...")
        success, stdout, err = self.ssh.run_command("ls -d /home/*/sims/* 2>/dev/null || ls -d ./sims/* 2>/dev/null || echo ''")
        if success and stdout.strip():
            sim_list = [Path(line.strip()).name for line in stdout.splitlines() if line.strip()]
            if sim_list:
                selected_sim, ok = QMessageBox.getItem(
                    self, "Simulaciones Encontradas en Cluster", "Seleccione la simulación a descargar:", sim_list, 0, False
                )
                if ok and selected_sim:
                    self.txt_sim_name.setText(selected_sim)
                    self.lbl_status.setText(f"Simulación seleccionada: {selected_sim}")
                return
        QMessageBox.information(self, "Exploración", "No se encontraron simulaciones o se utilizó el modo simulación.")

    def _start_download_global(self):
        remote_path = self.txt_remote.text().strip()
        local_path = self.txt_local.text().strip()
        self._execute_scp_download(remote_path, local_path, self.btn_download)

    def _start_download_specific(self):
        sim_name = self.txt_sim_name.text().strip()
        comp_type = self.cbo_component.currentData()
        local_base = Path(self.txt_spec_local.text().strip())

        if not sim_name:
            QMessageBox.warning(self, "Nombre Incompleto", "Especifique el nombre o ID de la simulación.")
            return

        # Sanitizar nombre de simulación
        safe_name = shlex.quote(sim_name)
        remote_base_path = f"/home/{self.ssh.username}/sims/{sim_name}" if self.ssh.username else f"./sims/{sim_name}"

        if comp_type == "trajectory":
            remote_target = f"{remote_base_path}/*.parquet"
            local_target = str(local_base / f"{sim_name}_trajectory.parquet")
        elif comp_type == "metadata":
            remote_target = f"{remote_base_path}/*.json"
            local_target = str(local_base / f"{sim_name}_metadata.json")
        else:
            remote_target = remote_base_path
            local_target = str(local_base / sim_name)

        self._execute_scp_download(remote_target, local_target, self.btn_download_spec)

    def _execute_scp_download(self, remote_path: str, local_path: str, trigger_btn: QPushButton):
        if not remote_path or not local_path:
            QMessageBox.warning(self, "Campos Vacíos", "Especifique las rutas remota y local.")
            return

        trigger_btn.setEnabled(False)
        self.lbl_status.setText("Iniciando transferencia SCP...")
        self.lbl_status.setStyleSheet("color: #89b4fa; font-weight: bold;")

        self.worker = DownloadWorker(self.ssh, remote_path, local_path)
        self.worker.progress_signal.connect(lambda msg: self.lbl_status.setText(msg))
        self.worker.finished_signal.connect(lambda success, msg: self._on_download_finished(success, msg, trigger_btn))
        self.worker.start()

    def _on_download_finished(self, success: bool, msg: str, trigger_btn: QPushButton):
        trigger_btn.setEnabled(True)
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
        """Lista los archivos y carpetas descargados en el directorio datos/."""
        if not self.local_dir.exists():
            return

        files = list(self.local_dir.glob("*"))
        self.tbl_files.setRowCount(len(files))
        for row, fpath in enumerate(files):
            size_kb = f"{fpath.stat().st_size / 1024:.1f} KB" if fpath.is_file() else "[Carpeta]"
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fpath.stat().st_mtime))

            self.tbl_files.setItem(row, 0, QTableWidgetItem(fpath.name))
            self.tbl_files.setItem(row, 1, QTableWidgetItem(size_kb))
            self.tbl_files.setItem(row, 2, QTableWidgetItem(mtime))
