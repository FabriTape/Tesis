"""Explorador de carpetas/archivos remotos vía SSH para LENGA+.

Permite navegar el sistema de archivos del cluster como si fuera local
(ls remoto por paso), sin depender de rutas hardcodeadas. Útil para elegir
simulaciones o archivos de resultado que no viven en $HOME.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
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

from lenga.ssh_client import SSHClientManager


class RemoteBrowseDialog(QDialog):
    """Diálogo modal para navegar directorios remotos del cluster via SSH.

    Args:
        ssh_client (SSHClientManager): Cliente SSH conectado al cluster.
        start_path (str): Directorio inicial ('~' se resuelve al home remoto).
        select_files (bool): Si True, permite seleccionar archivos individuales.
        parent: Widget padre opcional.
    """

    def __init__(
        self,
        ssh_client: SSHClientManager,
        start_path: str = "~",
        select_files: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ssh = ssh_client
        self.select_files = select_files
        self.current_path = self._resolve_home(start_path)
        self.selected_path: Optional[str] = None

        self.setWindowTitle("Explorador Remoto — Cluster SSH")
        self.resize(620, 500)
        self._init_ui()
        self._load(self.current_path)

    def _resolve_home(self, start_path: str) -> str:
        """Resuelve '~' al home del usuario remoto vía `echo $HOME`."""
        if start_path != "~":
            return start_path
        ok, stdout, _ = self.ssh.run_command("echo $HOME")
        if ok and stdout:
            for line in stdout.splitlines():
                line = line.strip()
                if line and not self._is_ssh_warning(line):
                    return line
        return "/"

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
        parent = str(PurePosixPath(self.current_path).parent)
        if parent != self.current_path:
            self._load(parent)

    @staticmethod
    def _is_ssh_warning(line: str) -> bool:
        """Descarta líneas de aviso/error de ssh mezcladas por pexpect con el stdout."""
        return (
            line.startswith("Warning: Permanently added")
            or "to the list of known hosts" in line
            or line.startswith("Pseudo-terminal will not be allocated")
        )

    def _list_dir(self, path: str) -> List[Tuple[str, bool]]:
        """Lista el contenido remoto. Devuelve (nombre, es_directorio)."""
        cmd = f"ls -1p {shlex.quote(path)} 2>/dev/null; true"
        ok, stdout, _ = self.ssh.run_command(cmd)
        items: List[Tuple[str, bool]] = []
        if ok and stdout:
            for line in stdout.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                if self._is_ssh_warning(line):
                    continue
                if line.endswith("/"):
                    items.append((line[:-1], True))
                else:
                    items.append((line, False))
        return items

    def _load(self, path: str) -> None:
        self.current_path = path
        self.selected_path = None
        self.txt_path.setText(path)
        self.lst.clear()

        items = self._list_dir(path)
        items.sort(key=lambda t: (not t[1], t[0].lower()))

        for name, is_dir in items:
            item = QListWidgetItem(name + "/" if is_dir else name)
            item.setData(Qt.UserRole, (f"{path.rstrip('/')}/{name}", is_dir))
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

    def selected_remote_path(self) -> Optional[str]:
        """Devuelve la ruta remota elegida (archivo clickeado o carpeta actual)."""
        if self.selected_path:
            return self.selected_path
        return self.current_path
