"""
Gestor de Nodos y Generador de Trabajos SLURM para LENGA.
Parsea comandos de sinfo/squeue, valida la regla de máximo 2 nodos por envío
y construye las configuraciones de experimentos en TOML / SLURM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lenga.ssh_client import SSHClientManager


@dataclass
class ClusterNode:
    """Información de estado de un nodo del cluster."""
    name: str
    partition: str
    state_raw: str
    state_display: str  # 'Libre', 'Ocupado', 'Mixto', 'No disponible'
    cpus: str = "N/A"
    gpus: str = "N/A"
    active_jobs: int = 0
    selected: bool = False

    @property
    def is_available(self) -> bool:
        return self.state_display in ("Libre", "Mixto")


@dataclass
class SimulationConfig:
    """Parámetros de configuración para el envío de simulaciones LAMMPS."""
    # Parámetros del experimento
    fills: List[float] = field(default_factory=lambda: [0.15])
    vels: List[float] = field(default_factory=lambda: [-50.0])
    angles: List[float] = field(default_factory=lambda: [90.0])
    moves: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    orientations: List[str] = field(default_factory=lambda: ["90_0_1_0"])
    np_proyectil: int = 1
    nsteps: int = 2000000

    # Propiedades del material (Sílice)
    young_modulus: float = 70.0e9
    poisson: float = 0.17
    surface_energy: float = 0.025
    density: float = 2000.0

    # Configuración de simulación
    ndumps: int = 25000
    nthermo: int = 5000
    timestep: float = 5e-11
    grain_radius_um: float = 0.76

    # SLURM / Nodos
    selected_nodes: List[str] = field(default_factory=lambda: ["gpu2"])
    partition: str = "gpu"
    tasks_per_node: int = 8
    remote_base_dir: str = "/home/fbazzoni"

    def validate_node_selection(self) -> Tuple[bool, str]:
        """Valida que la cantidad de nodos seleccionados sea máximo 2."""
        if not self.selected_nodes:
            return False, "Debe seleccionar al menos 1 nodo para enviar simulaciones."
        if len(self.selected_nodes) > 2:
            return False, f"Límite excedido: Seleccionó {len(self.selected_nodes)} nodos. Máximo permitido: 2 nodos."
        return True, f"Nodos seleccionados ({len(self.selected_nodes)}/2): {', '.join(self.selected_nodes)}"

    def to_toml_string(self) -> str:
        """Genera el contenido del archivo config.toml para enqueue_sims.py."""
        nodes_str = ", ".join([f'"{n}"' for n in self.selected_nodes])
        fills_str = ", ".join([str(f) for f in self.fills])
        vels_str = ", ".join([str(v) for v in self.vels])
        angles_str = ", ".join([str(a) for a in self.angles])
        moves_str = ", ".join([str(m) for m in self.moves])
        orient_str = ", ".join([f'"{o}"' for o in self.orientations])

        return f"""# Generado automáticamente por LENGA GUI
[simulation]
ndumps   = {self.ndumps}
nthermo  = {self.nthermo}
exe      = "lmp_g++_openmpi-4.0.2_stableSep2021"
timestep = {self.timestep}

[sweep]
grain_radius_um = {self.grain_radius_um}
xrotate         = 1
yrotate         = 0
zrotate         = 0
nsteps          = [{self.nsteps}]
guardar_restart = false
max_horas_job   = 72
grid_por_ff      = {{ "0.15" = [2, 2, 1], "0.35" = [2, 4, 1] }}
est_horas_por_ff = {{ "0.15" = 5.5, "0.35" = 11.0 }}

[material]
young_modulus  = {self.young_modulus}
poisson        = {self.poisson}
surface_energy = {self.surface_energy}
density        = {self.density}

[slurm]
partition = "{self.partition}"
nodelist  = "{','.join(self.selected_nodes)}"
tasks     = {self.tasks_per_node}
nodes     = {len(self.selected_nodes)}

[[experiment]]
fills        = [{fills_str}]
vels         = [{vels_str}]
angles       = [{angles_str}]
moves        = [{moves_str}]
orientations = [{orient_str}]
np           = {self.np_proyectil}
nsteps       = [{self.nsteps}]
"""


class ClusterManager:
    """Administra la consulta de recursos y envío de simulaciones al cluster."""

    def __init__(self, ssh_client: SSHClientManager):
        self.ssh = ssh_client

    def fetch_nodes_status(self) -> Tuple[bool, List[ClusterNode], str]:
        """Consulta sinfo y squeue para obtener la lista de nodos y su estado."""
        success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command("sinfo --format='%N|%P|%T|%c|%G'")
        if not success_sinfo:
            # Fallback a sinfo estándar
            success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command("sinfo")
            if not success_sinfo:
                return False, [], f"Error al ejecutar sinfo: {err_sinfo}"

        success_sq, out_sq, _ = self.ssh.run_command("squeue --format='%i|%P|%j|%u|%t|%M|%D|%R'")
        active_jobs_per_node = self._parse_squeue_jobs(out_sq if success_sq else "")

        nodes = self._parse_sinfo_output(out_sinfo, active_jobs_per_node)
        return True, nodes, "Estado de nodos actualizado."

    def _parse_squeue_jobs(self, squeue_out: str) -> Dict[str, int]:
        jobs_count: Dict[str, int] = {}
        lines = squeue_out.strip().splitlines()
        for line in lines[1:]:
            parts = line.split("|") if "|" in line else line.split()
            if len(parts) >= 8:
                node_list = parts[7].strip()
                for node in node_list.split(","):
                    node = node.strip()
                    if node and not node.startswith("("):
                        jobs_count[node] = jobs_count.get(node, 0) + 1
        return jobs_count

    def _parse_sinfo_output(self, sinfo_out: str, active_jobs: Dict[str, int]) -> List[ClusterNode]:
        nodes_dict: Dict[str, ClusterNode] = {}
        lines = sinfo_out.strip().splitlines()
        if not lines:
            return []

        header = lines[0]
        has_pipe = "|" in header

        for line in lines[1:]:
            if not line.strip():
                continue
            if has_pipe:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    node_names, partition, state = parts[0], parts[1], parts[2]
                    cpus = parts[3] if len(parts) > 3 else "N/A"
                    gpus = parts[4] if len(parts) > 4 else "N/A"
                else:
                    continue
            else:
                parts = line.split()
                if len(parts) >= 5:
                    partition, _, _, _, state = parts[0], parts[1], parts[2], parts[3], parts[4]
                    node_names = parts[5] if len(parts) > 5 else parts[0]
                    cpus, gpus = "N/A", "N/A"
                else:
                    continue

            partition = partition.replace("*", "")
            state_lower = state.lower()

            if "idle" in state_lower or "idle*" in state_lower:
                display = "Libre"
            elif "alloc" in state_lower or "busy" in state_lower:
                display = "Ocupado"
            elif "mix" in state_lower:
                display = "Mixto"
            else:
                display = "No disponible"

            # Expandir nombres de nodo simples o listas
            for n_name in self._expand_nodelist(node_names):
                if n_name not in nodes_dict:
                    nodes_dict[n_name] = ClusterNode(
                        name=n_name,
                        partition=partition,
                        state_raw=state,
                        state_display=display,
                        cpus=cpus,
                        gpus=gpus,
                        active_jobs=active_jobs.get(n_name, 0)
                    )

        return list(nodes_dict.values())

    def _expand_nodelist(self, nodelist_str: str) -> List[str]:
        """Expande sintaxis compacta de SLURM como node[01-03] a ['node01', 'node02', 'node03']."""
        if "[" not in nodelist_str:
            return [n.strip() for n in nodelist_str.split(",") if n.strip()]

        match = re.match(r"([a-zA-Z0-9_-]+)\[([0-9,-]+)\]", nodelist_str)
        if not match:
            return [nodelist_str]

        prefix = match.group(1)
        ranges = match.group(2).split(",")
        result = []
        for r in ranges:
            if "-" in r:
                start, end = r.split("-")
                width = len(start)
                for i in range(int(start), int(end) + 1):
                    result.append(f"{prefix}{str(i).zfill(width)}")
            else:
                result.append(f"{prefix}{r}")
        return result

    def submit_simulations(self, config: SimulationConfig) -> Tuple[bool, str]:
        """Valida y envía el job de simulación al cluster."""
        valid, msg = config.validate_node_selection()
        if not valid:
            return False, msg

        toml_content = config.to_toml_string()
        remote_dir = f"{config.remote_base_dir}/lenga_jobs"
        remote_toml_path = f"{remote_dir}/config_lenga.toml"

        # Crear carpeta remota y subir config.toml
        self.ssh.run_command(f"mkdir -p {remote_dir}")
        
        # Guardar archivo local temporal para subirlo
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
            tf.write(toml_content)
            temp_local = tf.name

        try:
            success, up_msg = self.ssh.upload_file(temp_local, remote_toml_path)
            if not success:
                return False, f"Fallo al cargar config.toml: {up_msg}"
        finally:
            if os.path.exists(temp_local):
                os.remove(temp_local)

        # Enviar script a la cola SLURM
        sbatch_cmd = f"cd {config.remote_base_dir}/cluster/scripts && sbatch --nodelist={','.join(config.selected_nodes)} slurm_submit_JKR.sh"
        if self.ssh.mock_mode:
            sbatch_cmd = f"echo 'Enviado job mock a nodos {config.selected_nodes}'"

        success_sb, stdout_sb, stderr_sb = self.ssh.run_command(sbatch_cmd)
        if success_sb:
            return True, f"Simulaciones lanzadas con éxito en nodos {config.selected_nodes}.\nRespuesta: {stdout_sb.strip()}"
        else:
            return False, f"Error al enviar sbatch: {stderr_sb.strip()}"
