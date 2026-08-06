"""Gestor de Nodos y Generador de Trabajos SLURM para LENGA.

Este módulo consulta y parsea la información de recursos del cluster mediante
los comandos SLURM `sinfo` y `squeue`, valida la regla de selección de máximo 2 nodos
y construye archivos de configuración TOML / SLURM para el envío de simulaciones.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lenga.ssh_client import SSHClientManager


@dataclass
class ClusterNode:
    """Representa la información y estado operativo de un nodo del cluster.

    Attributes:
        name (str): Nombre único del nodo (ej. 'gpu2', 'toko04').
        partition (str): Partición a la que pertenece ('gpu', 'XL', 'cpu').
        state_raw (str): Estado bruto devuelto por SLURM ('idle', 'alloc', 'mix').
        state_display (str): Estado formateado para la GUI ('Libre', 'Ocupado', 'Mixto', 'No disponible').
        cpus (str): Cantidad o especificación de CPUs.
        gpus (str): Especificación de GPUs.
        active_jobs (int): Número de trabajos actualmente ejecutándose en el nodo.
        selected (bool): Indica si el nodo fue seleccionado por el usuario.
    """
    name: str
    partition: str
    state_raw: str
    state_display: str
    cpus: str = "N/A"
    gpus: str = "N/A"
    active_jobs: int = 0
    selected: bool = False

    @property
    def is_available(self) -> bool:
        """Indica si el nodo está disponible para recibir nuevos trabajos.

        Returns:
            bool: True si el estado es 'Libre' o 'Mixto'.
        """
        return self.state_display in ("Libre", "Mixto")


@dataclass
class SimulationConfig:
    """Estructura de datos para parametrizar experimentos granulares en LAMMPS.

    Attributes:
        fills (List[float]): Lista de fracciones de llenado del lecho (ej. [0.15, 0.35]).
        vels (List[float]): Lista de velocidades de impacto [m/s].
        angles (List[float]): Lista de ángulos de impacto en grados.
        moves (List[int]): Lista de desplazamientos discretos del grano.
        orientations (List[str]): Orientaciones de rotación del lecho.
        np_proyectil (int): Cantidad de partículas en el proyectil.
        nsteps (int): Total de pasos (timesteps) a simular.
        young_modulus (float): Módulo de Young del material [Pa].
        poisson (float): Coeficiente de Poisson.
        surface_energy (float): Energía superficial modelo JKR [J/m²].
        density (float): Densidad del material [kg/m³].
        ndumps (int): Frecuencia de escritura de dumps [timesteps].
        nthermo (int): Frecuencia de escritura de thermo [timesteps].
        timestep (float): Paso de integración temporal [s].
        grain_radius_um (float): Radio del grano [μm].
        selected_nodes (List[str]): Nodos del cluster seleccionados por el usuario.
        partition (str): Partición SLURM elegida.
        tasks_per_node (int): Tareas concurrentes por nodo.
        remote_base_dir (str): Directorio raíz en el servidor remoto.
    """
    fills: List[float] = field(default_factory=lambda: [0.15])
    vels: List[float] = field(default_factory=lambda: [-50.0])
    angles: List[float] = field(default_factory=lambda: [90.0])
    moves: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    orientations: List[str] = field(default_factory=lambda: ["90_0_1_0"])
    np_proyectil: int = 1
    nsteps: int = 2000000

    # Propiedades mecánicas y físicas del material (Sílice)
    young_modulus: float = 70.0e9
    poisson: float = 0.17
    surface_energy: float = 0.025
    density: float = 2000.0

    # Configuración base de simulación
    ndumps: int = 25000
    nthermo: int = 5000
    timestep: float = 5e-11
    grain_radius_um: float = 0.76

    # SLURM / Nodos destino
    selected_nodes: List[str] = field(default_factory=lambda: ["gpu2"])
    partition: str = "gpu"
    tasks_per_node: int = 8
    remote_base_dir: str = "/home/fbazzoni"

    def validate_node_selection(self) -> Tuple[bool, str]:
        """Valida que la cantidad de nodos elegidos cumpla con el límite (Máximo 2 nodos).

        Returns:
            Tuple[bool, str]: Tupla con (es_valido: bool, mensaje: str).
        """
        if not self.selected_nodes:
            return False, "Debe seleccionar al menos 1 nodo para enviar simulaciones."
        if len(self.selected_nodes) > 2:
            return (
                False,
                f"Límite excedido: Seleccionó {len(self.selected_nodes)} nodos. "
                f"El máximo permitido es 2 nodos."
            )
        return True, f"Nodos seleccionados ({len(self.selected_nodes)}/2): {', '.join(self.selected_nodes)}"

    def to_toml_string(self) -> str:
        """Genera el contenido del archivo TOML de configuración para encolar trabajos.

        Returns:
            str: Cadena formateada en TOML compatible con enqueue_sims.py.
        """
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
    """Administra la consulta de recursos y envío de simulaciones al cluster.

    Args:
        ssh_client (SSHClientManager): Cliente SSH configurado para enviar comandos.
    """

    def __init__(self, ssh_client: SSHClientManager) -> None:
        """Inicializa el administrador del cluster.

        Args:
            ssh_client (SSHClientManager): Instancia del administrador de conexiones SSH.
        """
        self.ssh = ssh_client

    def fetch_nodes_status(self) -> Tuple[bool, List[ClusterNode], str]:
        """Consulta comandos sinfo y squeue para obtener la lista de nodos y su estado.

        Returns:
            Tuple[bool, List[ClusterNode], str]: Tupla con (exito, lista_nodos, mensaje).
        """
        # 1. Ejecutar sinfo para obtener lista de nodos y particiones
        success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command("sinfo --format='%N|%P|%T|%c|%G'")
        if not success_sinfo:
            success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command("sinfo")
            if not success_sinfo:
                return False, [], f"Error al ejecutar sinfo en el cluster: {err_sinfo}"

        # 2. Ejecutar squeue para contar trabajos activos por nodo
        success_sq, out_sq, _ = self.ssh.run_command("squeue --format='%i|%P|%j|%u|%t|%M|%D|%R'")
        active_jobs_per_node = self._parse_squeue_jobs(out_sq if success_sq else "")

        # 3. Parsear resultado de sinfo y estructurar lista de ClusterNode
        nodes = self._parse_sinfo_output(out_sinfo, active_jobs_per_node)
        return True, nodes, "Estado de nodos del cluster actualizado correctamente."

    def _parse_squeue_jobs(self, squeue_out: str) -> Dict[str, int]:
        """Parsea la salida del comando squeue para contabilizar trabajos activos por nodo.

        Args:
            squeue_out (str): Salida del comando squeue.

        Returns:
            Dict[str, int]: Diccionario con nombre_nodo -> cantidad_trabajos.
        """
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
        """Parsea la salida de sinfo y construye objetos ClusterNode.

        Args:
            sinfo_out (str): Salida en texto plano de sinfo.
            active_jobs (Dict[str, int]): Conteo de trabajos por nodo.

        Returns:
            List[ClusterNode]: Lista de objetos de estado de nodos.
        """
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

            # Expandir nombres de nodo si vienen en rango (ej. node[01-03])
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
        """Expande la sintaxis compacta de rangos de nodos de SLURM.

        Args:
            nodelist_str (str): Cadena de lista de nodos (ej. 'node[01-03]').

        Returns:
            List[str]: Lista con los nombres individuales de cada nodo.
        """
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
        """Valida las restricciones y envía el trabajo de simulación al cluster.

        Args:
            config (SimulationConfig): Configuración del experimento.

        Returns:
            Tuple[bool, str]: Tupla con (exito: bool, respuesta: str).
        """
        # 1. Validar la restricción estricta de máximo 2 nodos seleccionados
        valid, msg = config.validate_node_selection()
        if not valid:
            return False, msg

        # 2. Generar el contenido del archivo TOML
        toml_content = config.to_toml_string()
        remote_dir = f"{config.remote_base_dir}/lenga_jobs"
        remote_toml_path = f"{remote_dir}/config_lenga.toml"

        self.ssh.run_command(f"mkdir -p {remote_dir}")
        
        # 3. Guardar archivo temporal local y subir vía SCP
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
            tf.write(toml_content)
            temp_local = tf.name

        try:
            success, up_msg = self.ssh.upload_file(temp_local, remote_toml_path)
            if not success:
                return False, f"Fallo al cargar config.toml en el servidor remoto: {up_msg}"
        finally:
            if os.path.exists(temp_local):
                os.remove(temp_local)

        # 4. Enviar el script a la cola de trabajos SLURM (sbatch)
        sbatch_cmd = (
            f"cd {config.remote_base_dir}/cluster/scripts && "
            f"sbatch --nodelist={','.join(config.selected_nodes)} slurm_submit_JKR.sh"
        )
        if self.ssh.mock_mode:
            sbatch_cmd = f"echo 'Enviado job mock a nodos {config.selected_nodes}'"

        success_sb, stdout_sb, stderr_sb = self.ssh.run_command(sbatch_cmd)
        if success_sb:
            return True, f"Simulaciones lanzadas con éxito en nodos {config.selected_nodes}.\n{stdout_sb.strip()}"
        else:
            return False, f"Error al ejecutar sbatch en el cluster: {stderr_sb.strip()}"
