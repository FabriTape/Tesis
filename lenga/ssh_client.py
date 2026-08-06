"""
Gestor de conexiones SSH y transferencias SCP para LENGA.
Soporta autenticación mediante usuario/correo y contraseña usando pexpect / subprocess.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, Optional, Tuple


class SSHClientManager:
    """Administra conexiones SSH y transferencias SCP con el cluster de cómputo."""

    def __init__(self, host: str = "", port: int = 22, username: str = "", password: str = ""):
        self.host = host.strip()
        self.port = port
        self.username = username.strip()
        self.password = password
        self.is_connected = False
        self.mock_mode = False

    def configure(self, host: str, port: int, username: str, password: str, mock_mode: bool = False):
        """Actualiza las credenciales de conexión SSH."""
        self.host = host.strip()
        self.port = port
        self.username = username.strip()
        self.password = password
        self.mock_mode = mock_mode
        self.is_connected = False

    def test_connection(self) -> Tuple[bool, str]:
        """Verifica la conectividad SSH con las credenciales actuales."""
        if self.mock_mode:
            self.is_connected = True
            return True, "Conexión SSH exitosa (Modo Simulación / Mock)"

        if not self.host or not self.username:
            return False, "Host y Usuario/Correo son requeridos."

        success, stdout, stderr = self.run_command("echo 'SSH OK'")
        if success and "SSH OK" in stdout:
            self.is_connected = True
            return True, "Conexión SSH establecida correctamente."
        else:
            self.is_connected = False
            error_msg = stderr.strip() if stderr else "No se pudo autenticar o conectar al servidor SSH."
            return False, error_msg

    def run_command(self, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
        """
        Ejecuta un comando remoto vía SSH.
        Retorna (exito, stdout, stderr).
        """
        if self.mock_mode:
            return self._mock_command(command)

        import pexpect

        user_host = f"{self.username}@{self.host}"
        ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {self.port} {user_host} {subprocess.list2cmdline([command])}"

        try:
            child = pexpect.spawn(ssh_cmd, encoding="utf-8", timeout=timeout)
            i = child.expect([
                r"[pP]assword:",
                r"Are you sure you want to continue connecting",
                pexpect.EOF,
                pexpect.TIMEOUT
            ])
            
            if i == 1:
                child.sendline("yes")
                i = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])

            if i == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                output = child.before
                child.close()
                if child.exitstatus == 0:
                    return True, output, ""
                else:
                    return False, output, f"Exit code {child.exitstatus}"
            elif i == 2:
                output = child.before
                child.close()
                if child.exitstatus == 0:
                    return True, output, ""
                else:
                    return False, output, f"Exit status {child.exitstatus}"
            else:
                child.close()
                return False, "", "Tiempo de espera agotado (Timeout en SSH)"

        except Exception as e:
            return False, "", str(e)

    def download_file(
        self, remote_path: str, local_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """Descarga un archivo o carpeta remota usando SCP."""
        if self.mock_mode:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if progress_callback:
                progress_callback("Descarga simulada completada.")
            return True, f"Descarga simulada de {remote_path} a {local_path} exitosa."

        import pexpect

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        user_host = f"{self.username}@{self.host}"
        scp_cmd = f"scp -r -o StrictHostKeyChecking=no -P {self.port} {user_host}:{remote_path} {local_path}"

        try:
            if progress_callback:
                progress_callback(f"Iniciando descarga SCP de {remote_path}...")
                
            child = pexpect.spawn(scp_cmd, encoding="utf-8", timeout=300)
            i = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            
            if i == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                child.close()
                if child.exitstatus == 0:
                    if progress_callback:
                        progress_callback("Descarga SCP finalizada con éxito.")
                    return True, f"Archivo descargado en {local_path}"
                else:
                    return False, f"Error en SCP (código {child.exitstatus}): {child.before}"
            elif i == 1:
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo descargado en {local_path}"
                else:
                    return False, f"Error SCP: {child.before}"
            else:
                child.close()
                return False, "Timeout durante la descarga SCP."

        except Exception as e:
            return False, f"Error en transferencia SCP: {str(e)}"

    def upload_file(self, local_path: str, remote_path: str) -> Tuple[bool, str]:
        """Sube un archivo o carpeta local hacia el cluster usando SCP."""
        if self.mock_mode:
            return True, f"Carga simulada de {local_path} a {remote_path} exitosa."

        import pexpect

        user_host = f"{self.username}@{self.host}"
        scp_cmd = f"scp -r -o StrictHostKeyChecking=no -P {self.port} {local_path} {user_host}:{remote_path}"

        try:
            child = pexpect.spawn(scp_cmd, encoding="utf-8", timeout=300)
            i = child.expect([r"[pP]assword:", pexpect.EOF, pexpect.TIMEOUT])
            
            if i == 0:
                child.sendline(self.password)
                child.expect(pexpect.EOF)
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo cargado en {remote_path}"
                else:
                    return False, f"Error en subida SCP: {child.before}"
            elif i == 1:
                child.close()
                if child.exitstatus == 0:
                    return True, f"Archivo cargado en {remote_path}"
                else:
                    return False, f"Error SCP: {child.before}"
            else:
                child.close()
                return False, "Timeout durante la subida SCP."

        except Exception as e:
            return False, f"Error en subida SCP: {str(e)}"

    def _mock_command(self, command: str) -> Tuple[bool, str, str]:
        """Respuestas simuladas para desarrollo local sin conexión real a cluster."""
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
