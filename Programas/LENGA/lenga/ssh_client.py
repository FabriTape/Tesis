"""Gestor de conexiones SSH y transferencias SCP para LENGA.

Este módulo implementa el cliente de comunicación seguro con el cluster de cómputo,
soportando autenticación por usuario/correo y contraseña utilizando las herramientas
del sistema via pexpect / subprocess.
"""

from __future__ import annotations

import os
import shlex
import shutil
from typing import Callable, Optional, Tuple


class SSHClientManager:
    """Administra la sesión SSH, comandos remotos y transferencias SCP con el cluster.

    Attributes:
        host (str): Dirección IP o hostname del cluster SSH.
        port (int): Puerto de la conexión SSH (default: 22).
        username (str): Usuario o correo de inicio de sesión SSH.
        password (str): Contraseña del usuario SSH.
        is_connected (bool): Indica si se verificó exitosamente la sesión SSH.
        mock_mode (bool): Si es True, simula comandos locales sin conectar a internet/SSH.
    """

    def __init__(self, host: str = "", port: int = 22, username: str = "", password: str = "") -> None:
        """Inicializa el administrador de clientes SSH con credenciales opcionales.

        Args:
            host (str): Servidor/Host SSH.
            port (int): Puerto SSH (default: 22).
            username (str): Usuario o correo SSH.
            password (str): Contraseña del usuario SSH.
        """
        self.host = host.strip()
        self.port = port
        self.username = username.strip()
        self.password = password
        self.is_connected = False
        self.mock_mode = False

    def configure(
        self, host: str, port: int, username: str, password: str, mock_mode: bool = False
    ) -> None:
        """Configura y actualiza las credenciales de la sesión SSH activa.

        Args:
            host (str): Servidor o IP del cluster.
            port (int): Puerto SSH.
            username (str): Correo o usuario SSH.
            password (str): Contraseña SSH.
            mock_mode (bool): Si es True, activa el modo simulación local sin cluster.

        Returns:
            None
        """
        self.host = host.strip()
        self.port = port
        self.username = username.strip()
        self.password = password
        self.mock_mode = mock_mode
        self.is_connected = False

    def test_connection(self) -> Tuple[bool, str]:
        """Prueba la conectividad SSH ejecutando un comando de diagnóstico remoto.

        Returns:
            Tuple[bool, str]: Tupla con (éxito: bool, mensaje_resultado: str).

        Raises:
            None
        """
        # 1. En modo simulación (mock), retornar éxito inmediatamente
        if self.mock_mode:
            self.is_connected = True
            return True, "Conexión SSH exitosa (Modo Simulación / Mock local)."

        # 2. Validar que host y usuario no estén vacíos
        if not self.host or not self.username:
            return False, "Host y Usuario/Correo son requeridos para la conexión SSH."

        # 3. Ejecutar comando de prueba 'echo SSH OK'
        success, stdout, stderr = self.run_command("echo 'SSH OK'")
        if success and "SSH OK" in stdout:
            self.is_connected = True
            return True, "Conexión SSH establecida correctamente con el cluster."
        else:
            self.is_connected = False
            error_msg = stderr.strip() if stderr else "No se pudo autenticar o conectar al servidor SSH."
            return False, error_msg

    def run_command(self, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
        """Ejecuta un comando remoto vía SSH respondiendo automáticamente a la contraseña.

        Si el comando no se encuentra (exit 127), se reintenta envuelto en un shell de
        login (`bash -lc`) para que se carguen los PATH de los módulos/Slurm del cluster.

        Args:
            command (str): Comando bash a ejecutar en el servidor remoto.
            timeout (int): Tiempo máximo de espera en segundos (default: 60).

        Returns:
            Tuple[bool, str, str]: Tupla con (éxito: bool, stdout: str, stderr: str).
        """
        # 1. Redirigir si está activado el modo simulación
        if self.mock_mode:
            return self._mock_command(command)

        user_host = f"{self.username}@{self.host}"
        base_ssh = (
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-p {self.port} {user_host}"
        )

        ok, output, error = self._exec_ssh(f"{base_ssh} {shlex.quote(command)}", timeout)

        # 2. Si el comando remoto no se encontró (exit 127), reintentar con shell de login
        #    para que el perfil del cluster agregue los binarios de Slurm al PATH.
        if not ok and "127" in error:
            inner = command.replace("'", "'\\''")
            login_cmd = f"bash -lc '{inner}'"
            ok, output, error = self._exec_ssh(f"{base_ssh} {shlex.quote(login_cmd)}", timeout)

        return ok, output, error

    def _exec_ssh(self, ssh_cmd: str, timeout: int) -> Tuple[bool, str, str]:
        """Ejecuta una sesión pexpect SSH respondiendo a contraseña y fingerprint."""
        import pexpect

        try:
            child = pexpect.spawn(ssh_cmd, encoding="utf-8", timeout=timeout)
            index = child.expect([
                r"[pP]assword:",
                r"Are you sure you want to continue connecting",
                pexpect.EOF,
                pexpect.TIMEOUT
            ])

            # Confirmar fingerprint si es primera conexión
            if index == 1:
                child.sendline("yes")
                index = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])

            # Responder contraseña si es solicitada
            if index == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                output = child.before
                child.close()
                if child.exitstatus == 0:
                    return True, output, ""
                else:
                    return False, output, f"Exit code {child.exitstatus}"
            elif index == 2:
                output = child.before
                child.close()
                if child.exitstatus == 0:
                    return True, output, ""
                else:
                    return False, output, f"Exit status {child.exitstatus}"
            else:
                child.close()
                return False, "", "Tiempo de espera agotado (Timeout en SSH)."

        except Exception as err:
            return False, "", str(err)

    def download_file(
        self, remote_path: str, local_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """Descarga un archivo o directorio remoto usando el protocolo SCP.

        Args:
            remote_path (str): Ruta absoluta o relativa del archivo remoto en el cluster.
            local_path (str): Ruta de destino en el sistema de archivos local.
            progress_callback (Optional[Callable[[str], None]]): Callback para reportar el progreso.

        Returns:
            Tuple[bool, str]: Tupla con (exito: bool, mensaje: str).
        """
        if self.mock_mode:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if progress_callback:
                progress_callback("Descarga SCP simulada completada.")
            return True, f"Descarga simulada de {remote_path} a {local_path} exitosa."

        import pexpect

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        user_host = f"{self.username}@{self.host}"
        scp_cmd = f"scp -r -o StrictHostKeyChecking=no -P {self.port} {user_host}:{remote_path} {local_path}"

        try:
            if progress_callback:
                progress_callback(f"Iniciando descarga SCP de {remote_path}...")
                
            child = pexpect.spawn(scp_cmd, encoding="utf-8", timeout=300)
            index = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            
            if index == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                child.close()
                if child.exitstatus == 0:
                    if progress_callback:
                        progress_callback("Descarga SCP finalizada con éxito.")
                    return True, f"Archivo descargado en {local_path}"
                else:
                    return False, f"Error en SCP (código {child.exitstatus}): {child.before}"
            elif index == 1:
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo descargado en {local_path}"
                else:
                    return False, f"Error SCP: {child.before}"
            else:
                child.close()
                return False, "Timeout durante la transferencia SCP."

        except Exception as err:
            return False, f"Error en transferencia SCP: {str(err)}"

    def upload_file(self, local_path: str, remote_path: str) -> Tuple[bool, str]:
        """Sube un archivo o carpeta local hacia el cluster de cómputo usando SCP.

        Args:
            local_path (str): Ruta del archivo o carpeta local a transferir.
            remote_path (str): Ruta remota de destino en el cluster.

        Returns:
            Tuple[bool, str]: Tupla con (exito: bool, mensaje: str).
        """
        if self.mock_mode:
            return True, f"Carga simulada de {local_path} a {remote_path} exitosa."

        import pexpect

        user_host = f"{self.username}@{self.host}"
        scp_cmd = f"scp -r -o StrictHostKeyChecking=no -P {self.port} {local_path} {user_host}:{remote_path}"

        try:
            child = pexpect.spawn(scp_cmd, encoding="utf-8", timeout=300)
            index = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            
            if index == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo cargado en {remote_path}"
                else:
                    return False, f"Error en subida SCP: {child.before}"
            elif index == 1:
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo cargado en {remote_path}"
                else:
                    return False, f"Error SCP: {child.before}"
            else:
                child.close()
                return False, "Timeout durante la subida SCP."

        except Exception as err:
            return False, f"Error en subida SCP: {str(err)}"

    def _mock_command(self, command: str) -> Tuple[bool, str, str]:
        """Genera respuestas simuladas para desarrollo local sin conexión al cluster.

        Args:
            command (str): Comando simulado.

        Returns:
            Tuple[bool, str, str]: Salida simulada (exito, stdout, stderr).
        """
        if "sinfo" in command:
            mock_sinfo = (
                "PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST\n"
                "gpu*      up   infinite      1   idle  gpu2\n"
                "XL        up   infinite      1   alloc toko04\n"
                "cpu       up   infinite      2   idle  node01,node02\n"
            )
            return True, mock_sinfo, ""
        elif "squeue" in command:
            mock_squeue = (
                "JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n"
                " 1042        XL JKR_sims fbazzoni  R   02:15:00      1 toko04\n"
            )
            return True, mock_squeue, ""
        elif "CARBOXI" in command:
            return True, "Simulación de análisis CARBOXI finalizada correctamente.\nPython -m CARBOXI.merge_all ok.", ""
        elif "sbatch" in command:
            return True, "Submitted batch job 1043", ""
        else:
            return True, f"Mock output para comando: {command}", ""
