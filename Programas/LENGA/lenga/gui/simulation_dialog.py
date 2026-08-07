"""Ventana Modal / Formulario de Envío de Simulaciones LAMMPS para LENGA.

Permite configurar los parámetros del experimento granular y seleccionar los nodos destino
(enforzando la restricción estricta de seleccionar MÁXIMO 2 NODOS).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lenga.cluster_manager import ClusterManager, ClusterNode, SimulationConfig


class SimulationDialog(QDialog):
    """Ventana modal de formulario para parametrizar y lanzar simulaciones granulares.

    Args:
        cluster_mgr (ClusterManager): Instancia del gestor de cluster SLURM.
        available_nodes (List[ClusterNode]): Lista de nodos detectados en el cluster.
        parent (Optional[QWidget]): Widget padre de la ventana modal.
    """

    def __init__(
        self, cluster_mgr: ClusterManager, available_nodes: List[ClusterNode], parent: Optional[QWidget] = None
    ) -> None:
        """Inicializa el formulario de simulaciones.

        Args:
            cluster_mgr (ClusterManager): Gestor del cluster.
            available_nodes (List[ClusterNode]): Nodos conocidos del cluster.
            parent (Optional[QWidget]): Widget padre.
        """
        super().__init__(parent)
        self.cluster_mgr = cluster_mgr
        self.available_nodes = available_nodes
        self.setWindowTitle("Formulario de Envío de Simulaciones — LENGA+")
        from pathlib import Path
        from PyQt5.QtGui import QIcon
        logo_path = Path(__file__).resolve().parent / "assets/LengaPlus.png"
        self.setWindowIcon(QIcon(str(logo_path)))
        self.resize(750, 680)
        self.setModal(True)

        self._init_ui()

    def _init_ui(self) -> None:
        """Construye las secciones del formulario y la tabla de selección de nodos."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header Title
        title = QLabel("Configuración de Experimentos y Selección de Nodos")
        title.setObjectName("HeaderTitle")
        layout.addWidget(title)

        # ---------------------------------------------------------------------
        # SECCIÓN 1: FORMULARIO DE PARÁMETROS EXPERIMENTALES
        # ---------------------------------------------------------------------
        grp_params = QGroupBox("Parámetros del Barrido Experimental ([[experiment]])")
        grid_params = QFormLayout(grp_params)
        grid_params.setSpacing(8)

        self.txt_fills = QLineEdit("0.15, 0.35")
        self.txt_fills.setToolTip("Fracciones de llenado separadas por coma")

        self.txt_vels = QLineEdit("-50.0")
        self.txt_vels.setToolTip("Velocidades de impacto [m/s] separadas por coma")

        self.txt_angles = QLineEdit("90.0")
        self.txt_angles.setToolTip("Ángulos de impacto separados por coma")

        self.txt_moves = QLineEdit("-3, -2, -1, 0, 1, 2, 3")
        self.txt_moves.setToolTip("Desplazamientos de grano (move) separados por coma")

        self.txt_orientations = QLineEdit("90_0_1_0")
        self.txt_orientations.setToolTip("Orientaciones de lecho separadas por coma")

        self.spin_np = QSpinBox()
        self.spin_np.setRange(1, 100)
        self.spin_np.setValue(1)

        self.spin_nsteps = QSpinBox()
        self.spin_nsteps.setRange(1000, 100000000)
        self.spin_nsteps.setSingleStep(100000)
        self.spin_nsteps.setValue(2000000)

        grid_params.addRow("Fracciones Llenado (fills):", self.txt_fills)
        grid_params.addRow("Velocidades (vels):", self.txt_vels)
        grid_params.addRow("Ángulos de Impacto (angles):", self.txt_angles)
        grid_params.addRow("Movimientos (moves):", self.txt_moves)
        grid_params.addRow("Orientaciones Lecho:", self.txt_orientations)
        grid_params.addRow("Partículas Proyectil (np):", self.spin_np)
        grid_params.addRow("Pasos de Simulación (nsteps):", self.spin_nsteps)

        layout.addWidget(grp_params)

        # ---------------------------------------------------------------------
        # SECCIÓN 2: SELECCIÓN DE NODOS DEL CLUSTER (MÁXIMO 2 NODOS)
        # ---------------------------------------------------------------------
        grp_nodes = QGroupBox("Selección de Nodos Destino (MÁXIMO 2 NODOS)")
        v_nodes = QVBoxLayout(grp_nodes)

        # Badge indicador de conteo de nodos
        self.lbl_node_counter = QLabel("Nodos Seleccionados: 0 / 2 (Límite: Máximo 2 nodos)")
        self.lbl_node_counter.setStyleSheet("font-weight: bold; color: #89b4fa;")
        v_nodes.addWidget(self.lbl_node_counter)

        # Tabla de selección de nodos
        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(5)
        self.table_nodes.setHorizontalHeaderLabels(["Seleccionar", "Nodo", "Partición", "Estado", "Trabajos Activos"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_nodes.setSelectionMode(QTableWidget.NoSelection)

        v_nodes.addWidget(self.table_nodes)
        layout.addWidget(grp_nodes)

        # ---------------------------------------------------------------------
        # SECCIÓN 3: BOTONES DE ACCIÓN
        # ---------------------------------------------------------------------
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_submit = QPushButton("Lanzar Simulaciones al Cluster")
        self.btn_submit.setObjectName("SuccessButton")
        self.btn_submit.clicked.connect(self._handle_submit)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_submit)
        layout.addLayout(btn_layout)

        # Poblar la tabla de nodos después de instanciar los botones
        self._populate_nodes_table()

    def _populate_nodes_table(self) -> None:
        """Pobla la tabla de nodos con los nodos disponibles y sus estados."""
        nodes_to_show = self.available_nodes if self.available_nodes else [
            ClusterNode("gpu2", "gpu", "idle", "Libre", "16", "1", 0),
            ClusterNode("toko04", "XL", "alloc", "Ocupado", "32", "0", 1),
            ClusterNode("node01", "cpu", "idle", "Libre", "16", "0", 0),
            ClusterNode("node02", "cpu", "idle", "Libre", "16", "0", 0),
        ]

        self.table_nodes.setRowCount(len(nodes_to_show))
        self.check_boxes: List[QCheckBox] = []

        for row, node in enumerate(nodes_to_show):
            chk = QCheckBox()
            if row == 0:
                chk.setChecked(True)
            chk.stateChanged.connect(self._update_node_selection_count)
            self.check_boxes.append(chk)

            # Contenedor centrado para el CheckBox
            widget_chk = QWidget()
            l_chk = QHBoxLayout(widget_chk)
            l_chk.addWidget(chk)
            l_chk.setAlignment(Qt.AlignCenter)
            l_chk.setContentsMargins(0, 0, 0, 0)

            item_node = QTableWidgetItem(node.name)
            item_partition = QTableWidgetItem(node.partition)
            item_state = QTableWidgetItem(node.state_display)
            item_jobs = QTableWidgetItem(str(node.active_jobs))

            # Colorear estado
            if node.state_display == "Libre":
                item_state.setForeground(Qt.green)
            elif node.state_display == "Ocupado":
                item_state.setForeground(Qt.yellow)
            else:
                item_state.setForeground(Qt.red)

            self.table_nodes.setCellWidget(row, 0, widget_chk)
            self.table_nodes.setItem(row, 1, item_node)
            self.table_nodes.setItem(row, 2, item_partition)
            self.table_nodes.setItem(row, 3, item_state)
            self.table_nodes.setItem(row, 4, item_jobs)

        self._update_node_selection_count()

    def _get_selected_node_names(self) -> List[str]:
        """Obtiene la lista de nombres de nodos marcados por el usuario.

        Returns:
            List[str]: Lista de nombres de nodos seleccionados.
        """
        selected = []
        for row in range(self.table_nodes.rowCount()):
            chk = self.check_boxes[row]
            if chk.isChecked():
                node_name = self.table_nodes.item(row, 1).text()
                selected.append(node_name)
        return selected

    def _update_node_selection_count(self) -> None:
        """Actualiza el contador de selección de nodos y valida el límite (máximo 2 nodos)."""
        selected = self._get_selected_node_names()
        count = len(selected)

        has_submit = hasattr(self, "btn_submit")

        if count == 0:
            self.lbl_node_counter.setText("Nodos Seleccionados: 0 / 2 (⚠️ Seleccione al menos 1 nodo)")
            self.lbl_node_counter.setStyleSheet("font-weight: bold; color: #f38ba8;")
            if has_submit:
                self.btn_submit.setEnabled(False)
        elif count <= 2:
            self.lbl_node_counter.setText(f"Nodos Seleccionados: {count} / 2 (✓ Configuración válida)")
            self.lbl_node_counter.setStyleSheet("font-weight: bold; color: #a6e3a1;")
            if has_submit:
                self.btn_submit.setEnabled(True)
        else:
            self.lbl_node_counter.setText(f"Nodos Seleccionados: {count} / 2 (❌ ¡LÍMITE EXCEDIDO! Máximo 2 nodos)")
            self.lbl_node_counter.setStyleSheet("font-weight: bold; color: #f38ba8;")
            if has_submit:
                self.btn_submit.setEnabled(False)

    def _handle_submit(self) -> None:
        """Valida el formulario y envía el trabajo al cluster mediante ClusterManager."""
        selected_nodes = self._get_selected_node_names()
        if len(selected_nodes) > 2:
            QMessageBox.critical(self, "Error de Selección", "No puede seleccionar más de 2 nodos para enviar simulaciones.")
            return
        if len(selected_nodes) == 0:
            QMessageBox.warning(self, "Selección Vacía", "Por favor seleccione al menos 1 nodo.")
            return

        try:
            fills = [float(x.strip()) for x in self.txt_fills.text().split(",") if x.strip()]
            vels = [float(x.strip()) for x in self.txt_vels.text().split(",") if x.strip()]
            angles = [float(x.strip()) for x in self.txt_angles.text().split(",") if x.strip()]
            moves = [int(x.strip()) for x in self.txt_moves.text().split(",") if x.strip()]
            orientations = [x.strip() for x in self.txt_orientations.text().split(",") if x.strip()]
        except ValueError as err:
            QMessageBox.critical(self, "Error de Formato", f"Revise los valores numéricos ingresados: {str(err)}")
            return

        config = SimulationConfig(
            fills=fills,
            vels=vels,
            angles=angles,
            moves=moves,
            orientations=orientations,
            np_proyectil=self.spin_np.value(),
            nsteps=self.spin_nsteps.value(),
            selected_nodes=selected_nodes
        )

        self.btn_submit.setEnabled(False)
        success, msg = self.cluster_mgr.submit_simulations(config)
        self.btn_submit.setEnabled(True)

        if success:
            QMessageBox.information(self, "Simulaciones Enviadas", msg)
            self.accept()
        else:
            QMessageBox.critical(self, "Fallo al Enviar", msg)
