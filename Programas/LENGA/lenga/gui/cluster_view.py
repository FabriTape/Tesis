"""
Vista / Pestaña del Monitoreo de Nodos del Cluster y Estado de Colas en LENGA.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lenga.cluster_manager import ClusterManager, ClusterNode
from lenga.gui.simulation_dialog import SimulationDialog


class ClusterViewWidget(QWidget):
    """Panel visual para consultar el estado de nodos y lanzar simulaciones."""

    def __init__(self, cluster_mgr: ClusterManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cluster_mgr = cluster_mgr
        self.cached_nodes: List[ClusterNode] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header bar
        header_layout = QHBoxLayout()
        title = QLabel("Monitoreo de Nodos y Recursos del Cluster (SLURM)")
        title.setObjectName("SectionTitle")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.btn_refresh = QPushButton("Actualizar Estado Nodos")
        self.btn_refresh.clicked.connect(self.refresh_status)
        header_layout.addWidget(self.btn_refresh)

        self.btn_open_form = QPushButton("⚡ Abrir Formulario de Simulaciones")
        self.btn_open_form.setObjectName("PrimaryButton")
        self.btn_open_form.clicked.connect(self._open_simulation_dialog)
        header_layout.addWidget(self.btn_open_form)

        layout.addLayout(header_layout)

        # GroupBox Tabla de Nodos
        grp_nodes = QGroupBox("Estado Actual de Nodos (`sinfo`)")
        v_nodes = QVBoxLayout(grp_nodes)

        self.tbl_nodes = QTableWidget()
        self.tbl_nodes.setColumnCount(6)
        self.tbl_nodes.setHorizontalHeaderLabels(["Nombre Nodo", "Partición", "Estado", "CPUs", "GPUs", "Trabajos Activos"])
        self.tbl_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v_nodes.addWidget(self.tbl_nodes)

        layout.addWidget(grp_nodes)

        # Status Footer
        self.lbl_info = QLabel("Presione 'Actualizar Estado Nodos' para sincronizar con el cluster SSH.")
        self.lbl_info.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.lbl_info)

    def refresh_status(self):
        """Consulta el estado del cluster vía SSH."""
        self.btn_refresh.setEnabled(False)
        self.lbl_info.setText("Consultando `sinfo` y `squeue` en el cluster SSH...")

        success, nodes, msg = self.cluster_mgr.fetch_nodes_status()
        self.btn_refresh.setEnabled(True)

        if not success:
            self.lbl_info.setText(f"Error: {msg}")
            QMessageBox.warning(self, "Error de Conexión", msg)
            return

        self.cached_nodes = nodes
        self.lbl_info.setText(f"✓ Sincronizado: {len(nodes)} nodos encontrados en el cluster.")

        self.tbl_nodes.setRowCount(len(nodes))
        for row, node in enumerate(nodes):
            self.tbl_nodes.setItem(row, 0, QTableWidgetItem(node.name))
            self.tbl_nodes.setItem(row, 1, QTableWidgetItem(node.partition))
            
            item_state = QTableWidgetItem(node.state_display)
            if node.state_display == "Libre":
                item_state.setForeground(Qt.green)
            elif node.state_display == "Ocupado":
                item_state.setForeground(Qt.yellow)
            else:
                item_state.setForeground(Qt.red)

            self.tbl_nodes.setItem(row, 2, item_state)
            self.tbl_nodes.setItem(row, 3, QTableWidgetItem(node.cpus))
            self.tbl_nodes.setItem(row, 4, QTableWidgetItem(node.gpus))
            self.tbl_nodes.setItem(row, 5, QTableWidgetItem(str(node.active_jobs)))

    def _open_simulation_dialog(self):
        dlg = SimulationDialog(self.cluster_mgr, self.cached_nodes, parent=self)
        if dlg.exec_() == SimulationDialog.Accepted:
            self.refresh_status()
