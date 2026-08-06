"""
Vista / Pestaña del Generador y Visor Interactivo de Figuras en LENGA.
Permite seleccionar datasets, personalizar qué gráficos generar y previsualizarlos
en el lienzo embebido de Matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lenga.figure_generator import FigureGeneratorEngine


class FiguresViewWidget(QWidget):
    """Panel para seleccionar datos y generar gráficos interactivos con Matplotlib."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = FigureGeneratorEngine()
        self.current_df = None
        self.generated_figures: Dict[str, plt.Figure] = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Etapa 4: Generador y Visor Interactivo de Figuras")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Main horizontal splitter (Control Panel Left | Canvas Right)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------------------------------------------------------------
        # LEFT CONTROL PANEL
        # ---------------------------------------------------------------------
        left_widget = QWidget()
        v_left = QVBoxLayout(left_widget)
        v_left.setContentsMargins(0, 0, 10, 0)

        # 1. Dataset Selection Group
        grp_ds = QGroupBox("1. Seleccionar Conjunto de Datos (Dataset)")
        v_ds = QVBoxLayout(grp_ds)

        self.cbo_preset = QComboBox()
        self.cbo_preset.addItem("Preset: 9 Repeticiones (9_mov)", "9_mov")
        self.cbo_preset.addItem("Preset: 49 Repeticiones (49_mov)", "49_mov")
        self.cbo_preset.addItem("Archivo CSV Personalizado (Descargado / Local)", "custom")
        self.cbo_preset.currentIndexChanged.connect(self._on_dataset_changed)
        v_ds.addWidget(self.cbo_preset)

        h_custom = QHBoxLayout()
        self.lbl_path = QLabel("Buscando dataset...")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("font-size: 11px; color: #a6adc8;")
        self.btn_browse = QPushButton("Examinar CSV...")
        self.btn_browse.clicked.connect(self._browse_csv)
        h_custom.addWidget(self.lbl_path)
        h_custom.addWidget(self.btn_browse)
        v_ds.addLayout(h_custom)

        self.btn_load_ds = QPushButton("Cargar Dataset")
        self.btn_load_ds.clicked.connect(self.load_dataset)
        v_ds.addWidget(self.btn_load_ds)

        v_left.addWidget(grp_ds)

        # 2. Figure Checklist Group
        grp_figs = QGroupBox("2. Seleccionar Figuras a Generar")
        v_figs = QVBoxLayout(grp_figs)

        h_select_all = QHBoxLayout()
        btn_all = QPushButton("Marcar Todo")
        btn_all.clicked.connect(lambda: self._set_all_checks(True))
        btn_none = QPushButton("Desmarcar Todo")
        btn_none.clicked.connect(lambda: self._set_all_checks(False))
        for btn in (btn_all, btn_none):
            btn.setMinimumHeight(round(btn.sizeHint().height() * 1.6))
        h_select_all.addWidget(btn_all)
        h_select_all.addWidget(btn_none)
        v_figs.addLayout(h_select_all)

        # La lista de checkboxes vive en un QScrollArea: si la ventana es chica,
        # el scroll queda contenido dentro del recuadro y NUNCA se superpone con
        # los botones de acción de abajo.
        scroll_figs = QScrollArea()
        scroll_figs.setWidgetResizable(True)
        scroll_figs.setFrameShape(QFrame.NoFrame)
        scroll_figs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_figs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        check_container = QWidget()
        v_checks = QVBoxLayout(check_container)
        v_checks.setContentsMargins(2, 4, 6, 4)
        v_checks.setSpacing(2)

        self.check_boxes: Dict[str, QCheckBox] = {}
        for key, name in self.engine.AVAILABLE_FIGURES.items():
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet("font-size: 12px; padding: 1px 0;")
            self.check_boxes[key] = chk
            v_checks.addWidget(chk)
        v_checks.addStretch()

        scroll_figs.setWidget(check_container)
        v_figs.addWidget(scroll_figs)

        # El grupo de figuras absorbe el espacio disponible (stretch 1) para que
        # los botones de acción conserven su tamaño fijo al pie del panel.
        v_left.addWidget(grp_figs, 1)

        # 3. Action Buttons
        self.btn_generate = QPushButton("🎨 Generar y Visualizar Figuras")
        self.btn_generate.setObjectName("PrimaryButton")
        self.btn_generate.clicked.connect(self.generate_selected_figures)
        v_left.addWidget(self.btn_generate)

        self.btn_export = QPushButton("💾 Exportar Figuras a Disco (`LENGA/figuras/`)")
        self.btn_export.setObjectName("SuccessButton")
        self.btn_export.clicked.connect(self.export_figures)
        v_left.addWidget(self.btn_export)

        splitter.addWidget(left_widget)

        # ---------------------------------------------------------------------
        # RIGHT CANVAS PANEL
        # ---------------------------------------------------------------------
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(10, 0, 0, 0)

        # Tabs para alternar entre figuras generadas
        self.tabs_figs = QTabWidget()
        v_right.addWidget(self.tabs_figs)

        splitter.addWidget(right_widget)

        # Ajustar proporciones (30% izquierda | 70% derecha)
        splitter.setSizes([260, 600])
        layout.addWidget(splitter)

        # Intentar cargar dataset base al inicio
        self._auto_detect_initial_dataset()

    def _auto_detect_initial_dataset(self):
        # Buscar en cluster/datos/resultados_9_mov/merge_resultados.csv o LENGA/datos/
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        possible_paths = [
            base_dir / "cluster/datos/resultados_9_mov/merge_resultados.csv",
            Path(__file__).resolve().parent.parent.parent / "datos/merge_resultados.csv"
        ]

        found = None
        for p in possible_paths:
            if p.exists():
                found = p
                break

        if found:
            self.custom_csv_path = found
            self.lbl_path.setText(f"Encontrado: {found.name}")
            self.load_dataset()
        else:
            self.lbl_path.setText("Seleccione un archivo CSV.")

    def _on_dataset_changed(self, idx: int):
        data = self.cbo_preset.currentData()
        if data == "custom":
            self.btn_browse.setEnabled(True)
        else:
            self.btn_browse.setEnabled(False)
            self.lbl_path.setText(f"Preset seleccionado: {data}")

    def _browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV consolidado", "", "Archivos CSV (*.csv);;Todos (*)"
        )
        if file_path:
            self.custom_csv_path = Path(file_path)
            self.lbl_path.setText(self.custom_csv_path.name)
            self.load_dataset()

    def _set_all_checks(self, checked: bool):
        for chk in self.check_boxes.values():
            chk.setChecked(checked)

    def load_dataset(self):
        data_type = self.cbo_preset.currentData()

        if data_type == "custom":
            if not hasattr(self, "custom_csv_path") or not self.custom_csv_path.exists():
                QMessageBox.warning(self, "Dataset No Encontrado", "Por favor seleccione un archivo CSV válido.")
                return
            target_path = self.custom_csv_path
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            target_path = base_dir / f"cluster/datos/resultados_{data_type}/merge_resultados.csv"
            if not target_path.exists():
                target_path = Path(__file__).resolve().parent.parent.parent / "datos/merge_resultados.csv"

        success, df, msg = self.engine.load_dataset(target_path)
        if success:
            self.current_df = df
            QMessageBox.information(self, "Dataset Cargado", msg)
        else:
            QMessageBox.warning(self, "Aviso al Cargar Dataset", f"{msg}\nSe utilizará el generador con estimaciones sintéticas.")
            # Crear dataframe dummy sintético para que los gráficos funcionen sin errores
            import polars as pl
            self.current_df = pl.DataFrame({
                "move": [-3, -2, -1, 0, 1, 2, 3],
                "vel": [-50, -50, -50, -50, -50, -50, -50],
                "n_erocionadas": [12, 18, 25, 40, 22, 15, 10],
                "ff": [0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.35],
                "d_max": [0.0001, 0.00012, 0.00015, 0.00011, 0.00013, 0.00014, 0.0001]
            })

    def generate_selected_figures(self):
        if self.current_df is None:
            self.load_dataset()

        self.tabs_figs.clear()
        self.generated_figures.clear()

        dataset_label = self.cbo_preset.currentData()
        count = 0

        for key, chk in self.check_boxes.items():
            if chk.isChecked():
                fig, name = self.engine.generate_figure(key, dataset_label)
                self.generated_figures[name] = fig

                # Crear widget con Matplotlib Canvas y Toolbar
                canvas_widget = QWidget()
                v_box = QVBoxLayout(canvas_widget)

                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, canvas_widget)

                v_box.addWidget(toolbar)
                v_box.addWidget(canvas)

                tab_name = self.engine.AVAILABLE_FIGURES.get(key, key)
                self.tabs_figs.addTab(canvas_widget, tab_name)
                count += 1

        if count == 0:
            QMessageBox.warning(self, "Sin Selección", "Seleccione al menos una figura para generar.")
        else:
            self.tabs_figs.setCurrentIndex(0)

    def export_figures(self):
        if not self.generated_figures:
            self.generate_selected_figures()

        saved_files = []
        for name, fig in self.generated_figures.items():
            dest = self.engine.save_figure(fig, name)
            saved_files.append(dest.name)

        if saved_files:
            QMessageBox.information(
                self,
                "Figuras Exportadas",
                f"Se exportaron {len(saved_files)} figuras en:\n{self.engine.output_dir}\n\nArchivos: {', '.join(saved_files)}"
            )
