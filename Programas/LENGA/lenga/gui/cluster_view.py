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

        self.btn_open_form = QPushButton("Abrir Formulario de Simulaciones")
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
        self.tbl_nodes.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_nodes.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_nodes.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_nodes.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_nodes.setAlternatingRowColors(False)
        self.tbl_nodes.verticalHeader().setVisible(True)
        v_nodes.addWidget(self.tbl_nodes)

        layout.addWidget(grp_nodes, 1)

        # Status Footer
        self.lbl_info = QLabel("Presione 'Actualizar Estado Nodos' para sincronizar con el cluster SSH.")
        self.lbl_info.setStyleSheet("color: #a6adc8; font-weight: 500;")
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
        unique_nodes_count = len(set(n.name for n in nodes))
        self.lbl_info.setText(
            f"✓ Sincronizado: {len(nodes)} entradas de nodo/partición ({unique_nodes_count} nodos físicos únicos) en el cluster."
        )

        from PyQt5.QtGui import QColor, QFont

        self.tbl_nodes.setRowCount(len(nodes))
        for row, node in enumerate(nodes):
            self.tbl_nodes.setItem(row, 0, QTableWidgetItem(node.name))
            self.tbl_nodes.setItem(row, 1, QTableWidgetItem(node.partition))
            
            item_state = QTableWidgetItem(node.state_display)
            font = item_state.font()
            font.setBold(True)
            item_state.setFont(font)

            if node.state_display == "Libre":
                item_state.setForeground(QColor("#5C9449"))
            elif node.state_display == "Ocupado":
                item_state.setForeground(QColor("#E5C07B"))
            elif node.state_display == "Mixto":
                item_state.setForeground(QColor("#D95C14"))
            else:
                item_state.setForeground(QColor("#EE6060"))

            self.tbl_nodes.setItem(row, 2, item_state)
            self.tbl_nodes.setItem(row, 3, QTableWidgetItem(node.cpus))
            self.tbl_nodes.setItem(row, 4, QTableWidgetItem(node.gpus))
            self.tbl_nodes.setItem(row, 5, QTableWidgetItem(str(node.active_jobs)))

    def _open_simulation_dialog(self):
        dlg = SimulationDialog(self.cluster_mgr, self.cached_nodes, parent=self)
        if dlg.exec_() == SimulationDialog.Accepted:
            self.refresh_status()
