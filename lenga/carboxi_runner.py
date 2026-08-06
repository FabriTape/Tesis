"""
Orquestador del Análisis Remoto CARBOXI y Consolidación de CSV.
Ejecuta el análisis post-simulación en el cluster y genera el merge final.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
from lenga.ssh_client import SSHClientManager


class CarboxiRunner:
    """Ejecuta el pipeline de post-procesamiento CARBOXI y merge_all en el cluster."""

    def __init__(self, ssh_client: SSHClientManager):
        self.ssh = ssh_client

    def run_analysis_pipeline(
        self,
        remote_base_dir: str = "/home/fbazzoni",
        target_node: str = "gpu2",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Ejecuta la etapa 2 del flujo:
        1. Envia el job SLURM slurm_sumbit_gpu.sh o ejecuta python -m CARBOXI.
        2. Al finalizar, ejecuta python -m CARBOXI.merge_all para consolidar en un .csv único.
        """
        if progress_callback:
            progress_callback("Iniciando análisis CARBOXI en el cluster...")

        # 1. Enviar slurm_sumbit_gpu.sh o ejecutar comando remoto
        cmd_submit = f"cd {remote_base_dir} && sbatch --nodelist={target_node} slurm_sumbit_gpu.sh"
        if self.ssh.mock_mode:
            cmd_submit = "echo 'Submitted batch job 1045 (CARBOXI Analysis Mock)'"

        success_sb, stdout_sb, stderr_sb = self.ssh.run_command(cmd_submit)
        if not success_sb:
            return False, f"Fallo al lanzar el script de análisis SLURM: {stderr_sb}"

        if progress_callback:
            progress_callback(f"Trabajo de análisis CARBOXI enviado. {stdout_sb.strip()}")

        # 2. En modo sbatch, el script slurm_sumbit_gpu.sh ejecuta automaticamente CARBOXI.merge_all al terminar.
        # Si se desea forzar el merge manual inmediatamente:
        cmd_merge = f"source {remote_base_dir}/.venv_gpu/bin/activate 2>/dev/null || true; python -m CARBOXI.merge_all {remote_base_dir}/resultados"
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

    def trigger_merge_only(self, remote_base_dir: str = "/home/fbazzoni") -> Tuple[bool, str]:
        """Ejecuta únicamente la consolidación merge_all sobre los resultados existentes."""
        cmd_merge = f"source {remote_base_dir}/.venv_gpu/bin/activate 2>/dev/null || true; python -m CARBOXI.merge_all {remote_base_dir}/resultados"
        if self.ssh.mock_mode:
            return True, "Merge_all simulado exitoso."

        success, stdout, stderr = self.ssh.run_command(cmd_merge)
        if success:
            return True, f"Merge completado:\n{stdout.strip()}"
        else:
            return False, f"Error en merge: {stderr.strip()}"
