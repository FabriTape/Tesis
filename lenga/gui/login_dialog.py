"""
Diálogo de Autenticación e Inicio de Sesión SSH para LENGA.
Pide correo/usuario y contraseña para autenticar con el cluster de cómputo.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
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


class ConnectionWorker(QThread):
    """Thread en segundo plano para verificar credenciales SSH sin congelar la GUI."""
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, ssh_client: SSHClientManager, host: str, port: int, user: str, password: str, mock: bool):
        super().__init__()
        self.ssh = ssh_client
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.mock = mock

    def run(self):
        self.ssh.configure(self.host, self.port, self.user, self.password, mock_mode=self.mock)
        success, msg = self.ssh.test_connection()
        self.finished_signal.emit(success, msg)


class LoginDialog(QDialog):
    """Diálogo de inicio de sesión SSH."""

    def __init__(self, ssh_client: SSHClientManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ssh = ssh_client
        self.setWindowTitle("Autenticación SSH — LENGA Cluster")
        self.resize(420, 360)
        self.setModal(True)

        self._init_ui()

    def _init_ui(self):
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

        self.input_host = QLineEdit("toko04")
        self.input_host.setPlaceholderText("Ej: toko04.cluster.local o IP")

        self.input_port = QSpinBox()
        self.input_port.setRange(1, 65535)
        self.input_port.setValue(22)

        self.input_user = QLineEdit("fbazzoni")
        self.input_user.setPlaceholderText("Correo o usuario SSH")

        self.input_password = QLineEdit()
        self.input_password.setEchoMode(QLineEdit.Password)
        self.input_password.setPlaceholderText("Contraseña SSH")

        self.chk_mock = QCheckBox("Modo Simulación / Mock (desarrollo local sin cluster)")
        self.chk_mock.setChecked(True)  # Habilitado por defecto para facilitar pruebas de desarrollo

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

    def _handle_connect(self):
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

    def _on_connection_finished(self, success: bool, msg: str):
        self.btn_connect.setEnabled(True)
        if success:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.accept()
        else:
            self.lbl_status.setText(f"Error: {msg}")
            self.lbl_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
