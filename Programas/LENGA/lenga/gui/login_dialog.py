"""Diálogo de Autenticación e Inicio de Sesión SSH para LENGA.

Pide correo/usuario y contraseña para autenticar con el cluster de cómputo.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lenga.ssh_client import SSHClientManager


class HintLineEdit(QLineEdit):
    """QLineEdit con placeholder tenue (semitransparente) dibujado a mano.

    El stylesheet global de LENGA fija el color del texto y evita que el
    placeholder use la paleta, así que se pinta el texto de ayuda manualmente
    con baja opacidad para que se lea como una sugerencia, no como un valor.
    """

    def __init__(
        self,
        placeholder: str = "",
        hint_color: QColor = QColor(236, 239, 241, 85),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._hint_color = hint_color
        self.setPlaceholderText(placeholder)

    def paintEvent(self, event) -> None:
        ph = self.placeholderText()
        self.setPlaceholderText("")
        super().paintEvent(event)
        self.setPlaceholderText(ph)

        if ph and not self.text():
            painter = QPainter(self)
            painter.setPen(self._hint_color)
            font = self.font()
            font.setItalic(True)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(10, 0, -8, 0),
                Qt.AlignVCenter | Qt.AlignLeft,
                ph,
            )


class ConnectionWorker(QThread):
    """Thread worker en segundo plano para autenticar con SSH sin bloquear la GUI.

    Attributes:
        finished_signal (pyqtSignal): Señal emitida al finalizar la prueba (exito: bool, msg: str).
    """
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self, ssh_client: SSHClientManager, host: str, port: int, user: str, password: str, mock: bool
    ) -> None:
        """Inicializa el worker de autenticación.

        Args:
            ssh_client (SSHClientManager): Administrador SSH.
            host (str): Servidor/Host.
            port (int): Puerto SSH.
            user (str): Correo o usuario SSH.
            password (str): Contraseña SSH.
            mock (bool): Modo simulación.
        """
        super().__init__()
        self.ssh = ssh_client
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.mock = mock

    def run(self) -> None:
        """Ejecuta la prueba de conexión en el hilo secundario."""
        self.ssh.configure(self.host, self.port, self.user, self.password, mock_mode=self.mock)
        success, msg = self.ssh.test_connection()
        self.finished_signal.emit(success, msg)


class LoginDialog(QDialog):
    """Diálogo modal para solicitar credenciales de acceso SSH al cluster.

    Args:
        ssh_client (SSHClientManager): Gestor de conexión SSH.
        parent (Optional[QWidget]): Widget padre de la ventana.
    """

    def __init__(self, ssh_client: SSHClientManager, parent: Optional[QWidget] = None) -> None:
        """Inicializa el diálogo de inicio de sesión.

        Args:
            ssh_client (SSHClientManager): Objeto cliente SSH.
            parent (Optional[QWidget]): Padre del diálogo.
        """
        super().__init__(parent)
        self.ssh = ssh_client
        self.setWindowTitle("Autenticación SSH — Cluster LENGA+")
        from pathlib import Path
        from PyQt5.QtGui import QIcon
        logo_path = Path(__file__).resolve().parent / "assets/LengaPlus.png"
        self.setWindowIcon(QIcon(str(logo_path)))
        self.resize(420, 360)
        self.setModal(True)

        self._init_ui()

    def _init_ui(self) -> None:
        """Construye los widgets del formulario de autenticación."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        # Header Title
        title = QLabel("Conexión al Cluster de Cómputo")
        title.setObjectName("HeaderTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Ingrese sus credenciales de acceso SSH para ingresar al cluster:")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Formulario
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.input_host = HintLineEdit("Ejemplo: cluster.dominio.com")

        self.input_port = QSpinBox()
        self.input_port.setRange(1, 65535)
        self.input_port.setValue(22)

        self.input_user = HintLineEdit("Ejemplo: usuario@cluster.dominio.com")

        self.input_password = QLineEdit()
        self.input_password.setEchoMode(QLineEdit.Password)
        self.input_password.setPlaceholderText("Contraseña SSH")

        self.chk_mock = QCheckBox("Modo Simulación / Mock (desarrollo local sin cluster)")
        self.chk_mock.setChecked(True)

        form_layout.addRow("Servidor / Host:", self.input_host)
        form_layout.addRow("Puerto SSH:", self.input_port)
        form_layout.addRow("Correo / Usuario:", self.input_user)
        form_layout.addRow("Contraseña:", self.input_password)
        
        layout.addLayout(form_layout)
        layout.addWidget(self.chk_mock)

        # Status Message Label
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_connect = QPushButton("Conectar SSH")
        self.btn_connect.setObjectName("PrimaryButton")
        self.btn_connect.clicked.connect(self._handle_connect)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_connect)
        layout.addLayout(btn_layout)

    def _handle_connect(self) -> None:
        """Maneja el evento del botón Conectar e inicia la verificación asíncrona."""
        host = self.input_host.text().strip()
        port = self.input_port.value()
        user = self.input_user.text().strip()
        password = self.input_password.text()
        mock = self.chk_mock.isChecked()

        if not mock and (not host or not user):
            QMessageBox.warning(self, "Campos Incompletos", "Por favor complete el servidor y el correo/usuario SSH.")
            return

        self.btn_connect.setEnabled(False)
        self.lbl_status.setText("Conectando y verificando credenciales SSH...")
        self.lbl_status.setStyleSheet("color: #89b4fa; font-weight: bold;")

        self.worker = ConnectionWorker(self.ssh, host, port, user, password, mock)
        self.worker.finished_signal.connect(self._on_connection_finished)
        self.worker.start()

    def _on_connection_finished(self, success: bool, msg: str) -> None:
        """Callback al finalizar el worker de conexión SSH.

        Args:
            success (bool): Si la autenticación fue exitosa.
            msg (str): Mensaje devuelto por la verificación.
        """
        self.btn_connect.setEnabled(True)
        if success:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.accept()
        else:
            self.lbl_status.setText(f"Error: {msg}")
            self.lbl_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
