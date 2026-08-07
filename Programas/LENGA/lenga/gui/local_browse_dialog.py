"""Explorador de carpetas/archivos locales para LENGA+.

Espejo local de :class:`RemoteBrowseDialog` (misma experiencia de navegación)
pero sobre el sistema de archivos de la máquina, para mantener consistencia
de selección en toda la aplicación.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class LocalBrowseDialog(QDialog):
    """Diálogo modal para navegar directorios locales.

    Args:
        start_path (str): Directorio inicial ('' o '~' se resuelven al home local).
        select_files (bool): Si True, permite seleccionar archivos individuales.
        parent: Widget padre opcional.
    """

    def __init__(self, start_path: str = "~", select_files: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.select_files = select_files
        self.current_path = self._resolve_start(start_path)
        self.selected_path: Optional[str] = None

        self.setWindowTitle("Explorador de Archivos — Local")
        self.resize(620, 500)
        self._init_ui()
        self._load(self.current_path)

    def _resolve_start(self, start_path: str) -> str:
        if not start_path or start_path == "~":
            return str(Path.home())
        return str(Path(start_path))

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        h_path = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.returnPressed.connect(self._go)
        h_path.addWidget(self.txt_path, 1)
        self.btn_go = QPushButton("Ir")
        self.btn_go.clicked.connect(self._go)
        h_path.addWidget(self.btn_go)
        self.btn_up = QPushButton("Subir")
        self.btn_up.clicked.connect(self._up)
        h_path.addWidget(self.btn_up)
        layout.addLayout(h_path)

        self.lst = QListWidget()
        self.lst.itemDoubleClicked.connect(self._on_double_click)
        self.lst.itemClicked.connect(self._on_click)
        layout.addWidget(self.lst, 1)

        self.lbl_status = QLabel()
        layout.addWidget(self.lbl_status)

        h_btns = QHBoxLayout()
        h_btns.addStretch()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        h_btns.addWidget(self.btn_cancel)
        self.btn_ok = QPushButton("Seleccionar")
        self.btn_ok.setObjectName("PrimaryButton")
        self.btn_ok.clicked.connect(self.accept)
        h_btns.addWidget(self.btn_ok)
        layout.addLayout(h_btns)

    def _go(self) -> None:
        path = self.txt_path.text().strip()
        if path:
            self._load(path)

    def _up(self) -> None:
        parent = str(Path(self.current_path).parent)
        if parent != self.current_path:
            self._load(parent)

    def _list_dir(self, path: str) -> List[Tuple[str, bool]]:
        """Lista el contenido local. Devuelve (nombre, es_directorio)."""
        p = Path(path)
        if not p.is_dir():
            return []
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return []
        return [(e.name, e.is_dir()) for e in entries]

    def _load(self, path: str) -> None:
        self.current_path = path
        self.selected_path = None
        self.txt_path.setText(path)
        self.lst.clear()

        items = self._list_dir(path)
        for name, is_dir in items:
            item = QListWidgetItem(name + "/" if is_dir else name)
            item.setData(Qt.UserRole, (str(Path(path) / name), is_dir))
            self.lst.addItem(item)

        self.lbl_status.setText(
            f"{len(items)} elementos" if items else "Directorio vacío o sin permisos de lectura."
        )

    def _on_double_click(self, item: QListWidgetItem) -> None:
        full_path, is_dir = item.data(Qt.UserRole)
        if is_dir:
            self._load(full_path)

    def _on_click(self, item: QListWidgetItem) -> None:
        full_path, is_dir = item.data(Qt.UserRole)
        if is_dir:
            self.selected_path = full_path
        elif self.select_files:
            self.selected_path = full_path
        else:
            self.selected_path = None

    def selected_local_path(self) -> Optional[str]:
        """Devuelve la ruta local elegida (archivo clickeado o carpeta actual)."""
        if self.selected_path:
            return self.selected_path
        return self.current_path
