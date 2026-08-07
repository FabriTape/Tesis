"""Orquestador del Análisis Remoto CARBOXI y Consolidación CSV para LENGA.

Este módulo controla la ejecución remota de los scripts de análisis de erosión
y dinámica de ondas del módulo `CARBOXI`, así como el merge final mediante
`python -m CARBOXI.merge_all`.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
from lenga.ssh_client import SSHClientManager


class CarboxiRunner:
    """Orquesta la ejecución remota de CARBOXI y la consolidación de archivos CSV.

    Args:
        ssh_client (SSHClientManager): Administrador de la conexión SSH con el cluster.
    """

    def __init__(self, ssh_client: SSHClientManager) -> None:
        """Inicializa el ejecutador de CARBOXI.

        Args:
            ssh_client (SSHClientManager): Instancia del gestor SSH.
        """
        self.ssh = ssh_client

    def run_analysis_pipeline(
        self,
        remote_base_dir: str = "~",
        target_node: str = "gpu2",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """Ejecuta el pipeline de análisis remoto CARBOXI y consolidación final.

        Pasos del flujo:
        1. Enviar el trabajo SLURM slurm_sumbit_gpu.sh al nodo especificado.
        2. Ejecutar python -m CARBOXI.merge_all sobre el directorio de resultados.

        Args:
            remote_base_dir (str): Directorio raíz del usuario en el cluster (default: '~').
            target_node (str): Nodo objetivo para la ejecución (default: 'gpu2').
            progress_callback (Optional[Callable[[str], None]]): Callback para reportar logs en la GUI.

        Returns:
            Tuple[bool, str]: Tupla con (exito: bool, log_resultado: str).
        """
        if progress_callback:
            progress_callback("Iniciando análisis CARBOXI en el cluster remoto...")

        # 1. Lanzar el script de análisis SLURM en el cluster (slurm_sumbit_gpu.sh)
        cmd_submit = f"cd {remote_base_dir} && sbatch --nodelist={target_node} slurm_sumbit_gpu.sh"
        if self.ssh.mock_mode:
            cmd_submit = "echo 'Submitted batch job 1045 (CARBOXI Analysis Mock)'"

        success_sb, stdout_sb, stderr_sb = self.ssh.run_command(cmd_submit)
        if not success_sb:
            return False, f"Fallo al lanzar el script de análisis SLURM: {stderr_sb}"

        if progress_callback:
            progress_callback(f"Trabajo de análisis CARBOXI enviado. {stdout_sb.strip()}")

        # 2. Ejecutar la consolidación manual merge_all
        cmd_merge = (
            f"source {remote_base_dir}/.venv_gpu/bin/activate 2>/dev/null || true; "
            f"python -m CARBOXI.merge_all {remote_base_dir}/resultados"
        )
        if self.ssh.mock_mode:
            cmd_merge = "echo 'CARBOXI.merge_all completado en modo mock.'"

        if progress_callback:
            progress_callback("Verificando consolidación de datos (merge_all)...")

        success_m, stdout_m, stderr_m = self.ssh.run_command(cmd_merge)
        if success_m:
            if progress_callback:
                progress_callback("Consolidación de datos completada exitosamente.")
            return True, f"Análisis y merge completados:\n{stdout_m.strip()}"
        else:
            return False, f"Error durante CARBOXI.merge_all: {stderr_m.strip()}"

    def trigger_merge_only(self, remote_base_dir: str = "~") -> Tuple[bool, str]:
        """Ejecuta únicamente la consolidación merge_all sobre los datos existentes.

        Args:
            remote_base_dir (str): Directorio raíz remoto del proyecto (default: '~').

        Returns:
            Tuple[bool, str]: Tupla con (exito: bool, resultado: str).
        """
        cmd_merge = (
            f"source {remote_base_dir}/.venv_gpu/bin/activate 2>/dev/null || true; "
            f"python -m CARBOXI.merge_all {remote_base_dir}/resultados"
        )
        if self.ssh.mock_mode:
            return True, "Merge_all simulado exitoso."

        success, stdout, stderr = self.ssh.run_command(cmd_merge)
        if success:
            return True, f"Merge completado exitosamente:\n{stdout.strip()}"
        else:
            return False, f"Error en merge: {stderr.strip()}"
