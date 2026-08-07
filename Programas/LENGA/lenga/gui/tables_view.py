"""
Vista / Pestaña del Generador de Tablas (LaTeX) en LENGA.
Permite seleccionar un dataset y generar las tablas TAL CUAL de
``cluster/generador_figuras.py``, mostrando el código LaTeX resultante.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lenga.tables_generator import TablesGeneratorEngine


class TablesViewWidget(QWidget):
    """Panel para generar y visualizar las tablas LaTeX del análisis granular."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = TablesGeneratorEngine()
        self.generated_tables: Dict[str, str] = {}
        self.check_boxes: Dict[str, QCheckBox] = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Etapa 5: Generador de Tablas (LaTeX)")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # Main horizontal splitter (Control Panel Left | LaTeX Right)
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
        from PyQt5.QtWidgets import QListView
        self.cbo_preset.setView(QListView())
        self.cbo_preset.addItem("Preset: 9 Repeticiones (9_mov)", "9_mov")
        self.cbo_preset.addItem("Preset: 49 Repeticiones (49_mov)", "49_mov")
        self.cbo_preset.currentIndexChanged.connect(self._on_dataset_changed)
        v_ds.addWidget(self.cbo_preset)

        self.lbl_dataset = QLabel("Dataset seleccionado: 9_mov")
        self.lbl_dataset.setStyleSheet("font-size: 11px; color: #a6adc8;")
        v_ds.addWidget(self.lbl_dataset)

        v_left.addWidget(grp_ds)

        # 2. Table Checklist Group
        grp_tab = QGroupBox("2. Seleccionar Tablas a Generar")
        v_tab = QVBoxLayout(grp_tab)

        self.scroll_tables = QScrollArea()
        self.scroll_tables.setWidgetResizable(True)
        self.scroll_tables.setFrameShape(QFrame.NoFrame)
        self.scroll_tables.setStyleSheet("QScrollArea { background-color: #181825; border: 1px solid #313244; border-radius: 6px; }")
        self.scroll_tables.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_tables.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.check_container = QWidget()
        self.check_container.setStyleSheet("background-color: #181825; border-radius: 4px;")
        self.v_checks = QVBoxLayout(self.check_container)
        self.v_checks.setContentsMargins(6, 6, 6, 6)
        self.v_checks.setSpacing(4)

        self.scroll_tables.setWidget(self.check_container)
        v_tab.addWidget(self.scroll_tables)

        v_left.addWidget(grp_tab, 1)

        # 3. Action Buttons
        self.btn_generate = QPushButton("Generar Tablas")
        self.btn_generate.setObjectName("PrimaryButton")
        self.btn_generate.clicked.connect(self.generate_selected_tables)
        v_left.addWidget(self.btn_generate)

        self.btn_export = QPushButton("Exportar Tablas a Disco (`LENGA/tablas/`)")
        self.btn_export.setObjectName("SuccessButton")
        self.btn_export.clicked.connect(self.export_tables)
        v_left.addWidget(self.btn_export)

        self.btn_copy = QPushButton("Copiar Tabla Actual al Portapapeles")
        self.btn_copy.clicked.connect(self.copy_current_table)
        v_left.addWidget(self.btn_copy)

        splitter.addWidget(left_widget)

        # ---------------------------------------------------------------------
        # RIGHT LATEX PANEL
        # ---------------------------------------------------------------------
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(10, 0, 0, 0)

        self.tabs_latex = QTabWidget()
        v_right.addWidget(self.tabs_latex, 1)

        splitter.addWidget(right_widget)

        splitter.setSizes([270, 630])
        layout.addWidget(splitter, 1)

        self._rebuild_checklist()

    def _on_dataset_changed(self, idx: int):
        data = self.cbo_preset.currentData()
        self.lbl_dataset.setText(f"Dataset seleccionado: {data}")
        self._rebuild_checklist()

    def _rebuild_checklist(self):
        """Reconstruye los checkboxes con las tablas aplicables al dataset elegido."""
        while self.v_checks.count():
            item = self.v_checks.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.check_boxes.clear()
        dataset = self.cbo_preset.currentData()
        for key in self.engine.TABLES_FOR_DATASET.get(dataset, []):
            chk = QCheckBox(self.engine.AVAILABLE_TABLES.get(key, key))
            chk.setChecked(True)
            chk.setStyleSheet("font-size: 12px; padding: 3px 0; color: #cdd6f4;")
            self.check_boxes[key] = chk
            self.v_checks.addWidget(chk)
        self.v_checks.addStretch()

    def generate_selected_tables(self):
        """Genera las tablas seleccionadas y las muestra en las pestañas del panel derecho."""
        selected = [k for k, chk in self.check_boxes.items() if chk.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Sin Selección", "Seleccione al menos una tabla para generar.")
            return

        dataset = self.cbo_preset.currentData()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            all_tables = self.engine.generate_tables(dataset)
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Error al Generar Tablas",
                f"Falló la generación de tablas para '{dataset}':\n{err}",
            )
            return
        QApplication.restoreOverrideCursor()

        self.generated_tables = all_tables
        self.tabs_latex.clear()

        for key in selected:
            latex = all_tables.get(key, "% Tabla no disponible para este dataset")
            tab_name = self.engine.AVAILABLE_TABLES.get(key, key)

            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            editor.setPlainText(latex)
            editor.setStyleSheet(
                "font-family: 'JetBrains Mono', 'Fira Code', monospace; "
                "font-size: 12px; color: #cdd6f4; background-color: #1e1e2e;"
            )

            self.tabs_latex.addTab(editor, tab_name)

        if self.tabs_latex.count() > 0:
            self.tabs_latex.setCurrentIndex(0)
            QMessageBox.information(
                self, "Tablas Generadas",
                f"Se generaron {len(selected)} tablas para el dataset '{dataset}'.",
            )

    def export_tables(self):
        """Guarda las tablas generadas como archivos .tex en LENGA/tablas/<dataset>/."""
        if not self.generated_tables:
            self.generate_selected_tables()

        dataset = self.cbo_preset.currentData()
        saved = []
        for key, latex in self.generated_tables.items():
            dest = self.engine.save_table(key, latex, dataset)
            saved.append(dest.name)

        if saved:
            QMessageBox.information(
                self,
                "Tablas Exportadas",
                f"Se exportaron {len(saved)} tablas en:\n{self.engine.output_dir / dataset}\n\n"
                f"Archivos: {', '.join(saved)}",
            )

    def copy_current_table(self):
        """Copia el LaTeX de la pestaña activa al portapapeles."""
        idx = self.tabs_latex.currentIndex()
        if idx < 0:
            return
        editor = self.tabs_latex.widget(idx)
        if isinstance(editor, QPlainTextEdit) and editor.toPlainText():
            QApplication.clipboard().setText(editor.toPlainText())
            QMessageBox.information(self, "Copiado", "Código LaTeX copiado al portapapeles.")
