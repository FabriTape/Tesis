"""Vista / Pestaña del Generador y Visor Interactivo de Figuras y Tablas en LENGA.

Permite seleccionar datasets, realizar comparaciones inter-dataset (9_mov vs 49_mov),
personalizar qué gráficos generar, visualizarlos en el lienzo Matplotlib y
explorar la sección de tablas de datos procesados.
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
    QHeaderView,
    QLabel,
    QListView,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lenga.figure_generator import FigureGeneratorEngine


class FiguresViewWidget(QWidget):
    """Panel para seleccionar datos, comparar datasets, ver tablas y generar gráficos interactivos."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = FigureGeneratorEngine()
        self.current_df = None
        self.processed_tables: Dict[str, object] = {}
        self.generated_figures: Dict[str, plt.Figure] = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Etapa 4: Generador de Figuras, Comparación de Datasets y Tablas")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Main horizontal splitter (Control Panel Left | Main View Right)
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
        self.cbo_preset.setView(QListView())
        self.cbo_preset.addItem("Preset: 9 Repeticiones (9_mov)", "9_mov")
        self.cbo_preset.addItem("Preset: 49 Repeticiones (49_mov)", "49_mov")
        self.cbo_preset.addItem("Comparar Datasets (9_mov vs 49_mov)", "compare")
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

        # Lista scrolleable con fondo oscuro uniforme (#181825)
        scroll_figs = QScrollArea()
        scroll_figs.setWidgetResizable(True)
        scroll_figs.setFrameShape(QFrame.NoFrame)
        scroll_figs.setStyleSheet("QScrollArea { background-color: #181825; border: 1px solid #313244; border-radius: 6px; }")
        scroll_figs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_figs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        check_container = QWidget()
        check_container.setObjectName("CheckContainer")
        check_container.setStyleSheet("background-color: #181825; border-radius: 4px;")
        v_checks = QVBoxLayout(check_container)
        v_checks.setContentsMargins(6, 6, 6, 6)
        v_checks.setSpacing(4)

        self.check_boxes: Dict[str, QCheckBox] = {}
        for key, name in self.engine.AVAILABLE_FIGURES.items():
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet("font-size: 12px; padding: 2px 0; color: #cdd6f4;")
            self.check_boxes[key] = chk
            v_checks.addWidget(chk)
        v_checks.addStretch()

        scroll_figs.setWidget(check_container)
        v_figs.addWidget(scroll_figs)

        v_left.addWidget(grp_figs, 1)

        # 3. Action Buttons
        self.btn_generate = QPushButton("Generar y Visualizar Figuras")
        self.btn_generate.setObjectName("PrimaryButton")
        self.btn_generate.clicked.connect(self.generate_selected_figures)
        v_left.addWidget(self.btn_generate)

        self.btn_export = QPushButton("Exportar Figuras a Disco (`LENGA/figuras/`)")
        self.btn_export.setObjectName("SuccessButton")
        self.btn_export.clicked.connect(self.export_figures)
        v_left.addWidget(self.btn_export)

        splitter.addWidget(left_widget)

        # ---------------------------------------------------------------------
        # RIGHT PANEL: PESTAÑAS PRINCIPALES (GRÁFICOS VS TABLAS)
        # ---------------------------------------------------------------------
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(10, 0, 0, 0)

        self.main_right_tabs = QTabWidget()

        # Sub-pestaña 1: Canvas de Gráficos Interactivos
        self.tabs_figs = QTabWidget()
        self.main_right_tabs.addTab(self.tabs_figs, "Gráficos Interactivos")

        # Sub-pestaña 2: Sección de Tablas de Datos
        self.table_view_widget = QWidget()
        v_table_layout = QVBoxLayout(self.table_view_widget)

        h_tbl_ctrl = QHBoxLayout()
        h_tbl_ctrl.addWidget(QLabel("Seleccionar Tabla de Datos:"))

        self.cbo_table_select = QComboBox()
        self.cbo_table_select.setView(QListView())
        self.cbo_table_select.addItem("Distribución de Erosión (df_dist)", "df_dist")
        self.cbo_table_select.addItem("Estadística de Clusters (df_cluster)", "df_cluster")
        self.cbo_table_select.addItem("Dinámica de Proyectiles (df_proyectil)", "df_proyectil")
        self.cbo_table_select.currentIndexChanged.connect(self._render_selected_table)
        h_tbl_ctrl.addWidget(self.cbo_table_select)
        h_tbl_ctrl.addStretch()

        self.lbl_table_info = QLabel("Cargue un dataset para explorar sus tablas.")
        self.lbl_table_info.setStyleSheet("color: #a6adc8; font-weight: 500;")
        h_tbl_ctrl.addWidget(self.lbl_table_info)

        v_table_layout.addLayout(h_tbl_ctrl)

        self.tbl_data_display = QTableWidget()
        self.tbl_data_display.setAlternatingRowColors(False)
        self.tbl_data_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_data_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl_data_display.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_data_display.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_data_display.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v_table_layout.addWidget(self.tbl_data_display, 1)
        v_table_layout.addWidget(self.tbl_data_display, 1)

        self.main_right_tabs.addTab(self.table_view_widget, "Vista de Tablas de Datos")

        v_right.addWidget(self.main_right_tabs, 1)
        splitter.addWidget(right_widget)

        splitter.setSizes([270, 630])
        layout.addWidget(splitter)

        self._auto_detect_initial_dataset()

    def _auto_detect_initial_dataset(self):
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
        elif data == "compare":
            self.btn_browse.setEnabled(False)
            self.lbl_path.setText("Modo Comparativo: Se procesarán los presets 9_mov y 49_mov en paralelo.")
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

        if data_type == "compare":
            cfg9, df_dist9, df_cluster9, df_proj9 = self.engine.process_dataset("9_mov")
            cfg49, df_dist49, df_cluster49, df_proj49 = self.engine.process_dataset("49_mov")
            self.processed_tables = {
                "df_dist": df_dist9,
                "df_cluster": df_cluster9,
                "df_proyectil": df_proj9
            }
            self.current_df = df_dist9
            self._render_selected_table()
            QMessageBox.information(
                self, "Modo Comparativo Listo", "Se cargaron y estructuraron exitosamente los datasets de 9 y 49 repeticiones para comparación."
            )
            return

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

        cfg, df_dist, df_cluster, df_proyectil = self.engine.process_dataset(
            data_type if data_type in ("9_mov", "49_mov") else "9_mov"
        )
        self.processed_tables = {
            "df_dist": df_dist,
            "df_cluster": df_cluster,
            "df_proyectil": df_proyectil
        }
        self.current_df = df_dist
        self._render_selected_table()

    def _render_selected_table(self):
        key = self.cbo_table_select.currentData()
        if key not in self.processed_tables or self.processed_tables[key] is None:
            self.lbl_table_info.setText("Sin datos para mostrar.")
            self.tbl_data_display.setRowCount(0)
            return

        df_obj = self.processed_tables[key]
        if hasattr(df_obj, "to_pandas"):
            df = df_obj.to_pandas()
        else:
            df = df_obj

        self.tbl_data_display.setRowCount(len(df))
        self.tbl_data_display.setColumnCount(len(df.columns))
        self.tbl_data_display.setHorizontalHeaderLabels([str(c) for c in df.columns])

        if len(df.columns) > 8:
            self.tbl_data_display.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        else:
            self.tbl_data_display.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.lbl_table_info.setText(f"Mostrando tabla '{key}': {len(df):,} filas x {len(df.columns)} columnas.")

        # Poblar filas (limitado a las primeras 500 filas para rendimiento fluido)
        max_rows = min(500, len(df))
        for row_idx in range(max_rows):
            row_data = df.iloc[row_idx]
            for col_idx, col_name in enumerate(df.columns):
                val = row_data[col_name]
                if isinstance(val, (float, int)):
                    val_str = f"{val:.5g}"
                else:
                    val_str = str(val)
                item = QTableWidgetItem(val_str)
                self.tbl_data_display.setItem(row_idx, col_idx, item)

        # Ajustar ancho de columnas al máximo entre título de encabezado y valores
        self.tbl_data_display.resizeColumnsToContents()
        for col_idx in range(len(df.columns)):
            calculated_w = self.tbl_data_display.columnWidth(col_idx)
            header_str = str(df.columns[col_idx])
            # Estimar ancho mínimo del encabezado (caracteres * 9px + padding)
            min_header_w = max(len(header_str) * 9 + 20, 75)
            final_w = max(calculated_w + 18, min_header_w)
            self.tbl_data_display.setColumnWidth(col_idx, final_w)

    def generate_selected_figures(self):
        if not self.processed_tables:
            self.load_dataset()

        self.tabs_figs.clear()
        self.generated_figures.clear()

        dataset_label = self.cbo_preset.currentData()
        target_labels = ["9_mov", "49_mov"] if dataset_label == "compare" else [dataset_label]
        count = 0

        for lbl in target_labels:
            for key, chk in self.check_boxes.items():
                if chk.isChecked():
                    fig, name = self.engine.generate_figure(key, lbl if lbl in ("9_mov", "49_mov") else "9_mov")
                    fig_key = f"{name}_{lbl}" if dataset_label == "compare" else name
                    self.generated_figures[fig_key] = fig

                    canvas_widget = QWidget()
                    v_box = QVBoxLayout(canvas_widget)

                    canvas = FigureCanvas(fig)
                    toolbar = NavigationToolbar(canvas, canvas_widget)

                    v_box.addWidget(toolbar)
                    v_box.addWidget(canvas)

                    base_title = self.engine.AVAILABLE_FIGURES.get(key, key)
                    tab_title = f"[{lbl}] {base_title}" if dataset_label == "compare" else base_title
                    self.tabs_figs.addTab(canvas_widget, tab_title)
                    count += 1

        if count == 0:
            QMessageBox.warning(self, "Sin Selección", "Seleccione al menos una figura para generar.")
        else:
            self.tabs_figs.setCurrentIndex(0)
            self.main_right_tabs.setCurrentIndex(0)

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
