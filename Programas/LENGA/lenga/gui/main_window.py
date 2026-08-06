"""Ventana Principal Shell (MainWindow) para LENGA.

Integra en una interfaz gráfica por pestañas los módulos de:
1. Estado del Cluster & Monitoreo de Nodos.
2. Análisis Remoto con CARBOXI & Consolidación CSV.
3. Gestor de Descarga SCP de Resultados.
4. Generador e Inspector Interactivo de Figuras.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lenga.carboxi_runner import CarboxiRunner
from lenga.cluster_manager import ClusterManager
from lenga.gui.analysis_view import AnalysisViewWidget
from lenga.gui.cluster_view import ClusterViewWidget
from lenga.gui.download_view import DownloadViewWidget
from lenga.gui.figures_view import FiguresViewWidget
from lenga.gui.login_dialog import LoginDialog
from lenga.gui.simulation_dialog import SimulationDialog
from lenga.gui.style import DARK_STYLESHEET
from lenga.ssh_client import SSHClientManager


class MainWindow(QMainWindow):
    """Ventana Principal de la Aplicación LENGA.

    Args:
        dry_run (bool): Si es True, inicia la aplicación directamente en modo simulación (Mock).
    """

    def __init__(self, dry_run: bool = False) -> None:
        """Inicializa la ventana principal.

        Args:
            dry_run (bool): Modo simulación sin requerir credenciales reales SSH.
        """
        super().__init__()
        self.ssh_client = SSHClientManager()
        if dry_run:
            self.ssh_client.configure("toko04", 22, "fbazzoni", "", mock_mode=True)

        self.cluster_mgr = ClusterManager(self.ssh_client)
        self.carboxi_runner = CarboxiRunner(self.ssh_client)

        self.setWindowTitle("LENGA — Plataforma de Simulación y Análisis de Medios Granulares")
        self.resize(1150, 780)
        self.setMinimumSize(960, 640)

        # Aplicar el tema oscuro global (Catppuccin Mocha / Dark stylesheet)
        self.setStyleSheet(DARK_STYLESHEET)

        self._init_ui()

        # Al iniciar: cargar automáticamente el estado de los nodos (solicita login si hace falta)
        QTimer.singleShot(400, self._auto_bootstrap)

    def _auto_bootstrap(self) -> None:
        """Carga la lista de nodos al arrancar sin requerir clicks adicionales.

        Si ya hay conexión SSH (o modo simulación), consulta `sinfo`/`squeue`
        directamente; si no, abre el diálogo de autenticación (al aceptar, la
        lista se refresca sola).
        """
        if self.ssh_client.is_connected or self.ssh_client.mock_mode:
            self.cluster_view.refresh_status()
        else:
            self.open_login_dialog()

    def _init_ui(self) -> None:
        """Construye los elementos de la barra superior, navegación por pestañas y footer."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # ---------------------------------------------------------------------
        # BARRA SUPERIOR DE HERRAMIENTAS Y ESTADO DE SESIÓN SSH
        # ---------------------------------------------------------------------
        top_bar = QHBoxLayout()

        title_lbl = QLabel("LENGA")
        title_lbl.setObjectName("HeaderTitle")
        top_bar.addWidget(title_lbl)

        subtitle_lbl = QLabel("|  Simulación & Análisis Granular (Tesis)")
        subtitle_lbl.setStyleSheet("color: #a6adc8; font-size: 13px; font-weight: 500;")
        top_bar.addWidget(subtitle_lbl)

        top_bar.addStretch()

        # Insignia de estado de la conexión SSH
        self.lbl_ssh_status = QLabel("SSH: Desconectado")
        self.lbl_ssh_status.setObjectName("BadgeWarning")
        top_bar.addWidget(self.lbl_ssh_status)

        self.btn_login = QPushButton("🔑 Iniciar Sesión SSH")
        self.btn_login.clicked.connect(self.open_login_dialog)
        top_bar.addWidget(self.btn_login)

        self.btn_sim_form = QPushButton("⚡ Nueva Simulación (LAMMPS)")
        self.btn_sim_form.setObjectName("PrimaryButton")
        self.btn_sim_form.clicked.connect(self.open_simulation_dialog)
        top_bar.addWidget(self.btn_sim_form)

        main_layout.addLayout(top_bar)

        # ---------------------------------------------------------------------
        # PANEL DE NAVEGACIÓN POR PESTAÑAS (QTabWidget)
        # ---------------------------------------------------------------------
        self.tabs = QTabWidget()

        self.cluster_view = ClusterViewWidget(self.cluster_mgr)
        self.analysis_view = AnalysisViewWidget(self.carboxi_runner)
        self.download_view = DownloadViewWidget(self.ssh_client)
        self.figures_view = FiguresViewWidget()

        tab_bar = self.tabs.tabBar()
        tab_bar.setUsesScrollButtons(True)
        tab_bar.setExpanding(False)

        self.tabs.addTab(self.cluster_view, "Cluster y Nodos")
        self.tabs.addTab(self.analysis_view, "Análisis CARBOXI")
        self.tabs.addTab(self.download_view, "Descarga SCP")
        self.tabs.addTab(self.figures_view, "Generador de Figuras")

        main_layout.addWidget(self.tabs)

        # ---------------------------------------------------------------------
        # BARRA DE ESTADO INFERIOR
        # ---------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("LENGA listo. Inicie sesión SSH para comenzar.")

        self.update_ssh_status_badge()

    def open_login_dialog(self) -> None:
        """Abre el diálogo modal de autenticación SSH."""
        dlg = LoginDialog(self.ssh_client, parent=self)
        if dlg.exec_() == LoginDialog.Accepted:
            self.update_ssh_status_badge()
            self.cluster_view.refresh_status()

    def open_simulation_dialog(self) -> None:
        """Abre la ventana modal del formulario para parametrizar y enviar simulaciones."""
        if not self.ssh_client.is_connected and not self.ssh_client.mock_mode:
            res = QMessageBox.question(
                self,
                "Sin Conexión SSH",
                "Aún no ha iniciado sesión SSH con el cluster.\n¿Desea iniciar sesión o continuar en Modo Simulación?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if res == QMessageBox.Yes:
                self.open_login_dialog()
            else:
                self.ssh_client.mock_mode = True

        dlg = SimulationDialog(self.cluster_mgr, self.cluster_view.cached_nodes, parent=self)
        dlg.exec_()

    def update_ssh_status_badge(self) -> None:
        """Actualiza la insignia visual del estado de la conexión SSH en la barra superior."""
        if self.ssh_client.is_connected:
            mode_str = " (Mock)" if self.ssh_client.mock_mode else ""
            self.lbl_ssh_status.setText(f"✓ SSH: Conectado a {self.ssh_client.host}{mode_str}")
            self.lbl_ssh_status.setObjectName("BadgeSuccess")
            self.lbl_ssh_status.setStyle(self.lbl_ssh_status.style())
            self.status_bar.showMessage(f"Conectado vía SSH a {self.ssh_client.username}@{self.ssh_client.host}")
        elif self.ssh_client.mock_mode:
            self.lbl_ssh_status.setText("✓ SSH: Modo Simulación / Mock")
            self.lbl_ssh_status.setObjectName("BadgeSuccess")
            self.lbl_ssh_status.setStyle(self.lbl_ssh_status.style())
            self.status_bar.showMessage("Ejecutando en Modo Simulación (Mock local)")
        else:
            self.lbl_ssh_status.setText("⚠️ SSH: Desconectado")
            self.lbl_ssh_status.setObjectName("BadgeWarning")
            self.lbl_ssh_status.setStyle(self.lbl_ssh_status.style())
            self.status_bar.showMessage("Sin conexión SSH activa.")
