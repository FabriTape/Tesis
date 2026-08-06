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
        self.last_sinfo_raw: str = ""
        self.last_squeue_raw: str = ""

    def fetch_nodes_status(self) -> Tuple[bool, List[ClusterNode], str]:
        """Consulta comandos sinfo y squeue para obtener la lista completa de nodos y su estado.

        Returns:
            Tuple[bool, List[ClusterNode], str]: Tupla con (exito, lista_nodos, mensaje).
        """
        debug = os.environ.get("LENGA_DEBUG") == "1"

        def _log(attempt: str, ok: bool, out: str, err: str = "") -> None:
            if debug:
                print(f"[LENGA-DEBUG] intento: {attempt} | success={ok}")
                print(f"[LENGA-DEBUG]   stdout:\n{out[:3000]}")
                if err:
                    print(f"[LENGA-DEBUG]   stderr: {err[:500]}")

        # 1. Probar sinfo con flag -N (orientado a nodos individuales) y formato explícito pipe.
        #    -a incluye TODAS las particiones (incluso down/reservadas).
        sinfo_cmd = 'sinfo -a -N -o "%N|%P|%T|%c|%C|%G"'
        success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command(sinfo_cmd)
        _log(sinfo_cmd, success_sinfo, out_sinfo, err_sinfo)

        if not success_sinfo or not out_sinfo.strip() or "INVALID" in out_sinfo.upper():
            sinfo_cmd = 'sinfo -a -N -o "%N|%P|%T|%c|%G"'
            success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command(sinfo_cmd)
            _log(sinfo_cmd, success_sinfo, out_sinfo, err_sinfo)

        if not success_sinfo or not out_sinfo.strip():
            sinfo_cmd = "sinfo -a -N"
            success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command(sinfo_cmd)
            _log(sinfo_cmd, success_sinfo, out_sinfo, err_sinfo)

        if not success_sinfo or not out_sinfo.strip():
            sinfo_cmd = "sinfo -a"
            success_sinfo, out_sinfo, err_sinfo = self.ssh.run_command(sinfo_cmd)
            _log(sinfo_cmd, success_sinfo, out_sinfo, err_sinfo)
            if not success_sinfo:
                return False, [], f"Error al ejecutar sinfo en el cluster: {err_sinfo}"

        self.last_sinfo_raw = out_sinfo

        # 2. Ejecutar squeue para contar trabajos activos por nodo
        success_sq, out_sq, _ = self.ssh.run_command("squeue --format='%i|%P|%j|%u|%t|%M|%D|%R'")
        if not success_sq or not out_sq.strip():
            success_sq, out_sq, _ = self.ssh.run_command("squeue")
        _log("squeue", success_sq, out_sq if success_sq else "")

        self.last_squeue_raw = out_sq if success_sq else ""
        active_jobs_per_node = self._parse_squeue_jobs(self.last_squeue_raw)

        # 3. Parsear resultado de sinfo y estructurar lista de ClusterNode
        nodes = self._parse_sinfo_output(out_sinfo, active_jobs_per_node)

        if debug:
            print(f"[LENGA-DEBUG] nodos parseados: {len(nodes)}")
            for n in nodes:
                print(
                    f"[LENGA-DEBUG]   {n.name} | {n.partition} | {n.state_display} "
                    f"| cpus={n.cpus} | gpus={n.gpus} | jobs={n.active_jobs}"
                )
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
                    node = re.sub(r"\([^)]*\)", "", node).strip()
                    if node:
                        jobs_count[node] = jobs_count.get(node, 0) + 1
        return jobs_count

    def _parse_sinfo_output(self, sinfo_out: str, active_jobs: Dict[str, int]) -> List[ClusterNode]:
        """Parsea la salida de sinfo y construye objetos ClusterNode.

        Soporta los formatos:
        - ``"%N|%P|%T|%c|%C|%G"`` / ``"%N|%P|%T|%c|%G"`` (separador ``|``).
        - ``sinfo -N`` por defecto (cabecera inicia con ``NODELIST``).
        - ``sinfo`` por defecto (cabecera inicia con ``PARTITION``).

        Args:
            sinfo_out (str): Salida en texto plano de sinfo.
            active_jobs (Dict[str, int]): Conteo de trabajos por nodo.

        Returns:
            List[ClusterNode]: Lista de objetos de estado de nodos.
        """
        lines = [ln.rstrip() for ln in sinfo_out.strip().splitlines()]
        if not lines:
            return []

        header = lines[0].lower()
        data_lines = [ln for ln in lines[1:] if ln.strip()]
        pipe_mode = any("|" in ln for ln in data_lines)
        node_mode = pipe_mode or header.startswith("nodelist") or header.startswith("hostnames")

        nodes_dict: Dict[str, ClusterNode] = {}

        for line in data_lines:
            if line.startswith("NODELIST") or line.startswith("HOSTNAMES"):
                continue

            if pipe_mode:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue
                node_names, partition, state = parts[0], parts[1], parts[2]
                cpus_raw = parts[3] if len(parts) > 3 and parts[3] not in ("", "N/A") else "N/A"
                cpus_detail = parts[4] if len(parts) > 4 else "N/A"
                gpus = parts[5] if len(parts) > 5 else "N/A"
                cpus = self._fmt_cpus(cpus_raw, cpus_detail)
            else:
                parts = line.split()
                if len(parts) < 4:
                    continue
                if node_mode:
                    # sinfo -N (default): NODELIST NODES PARTITION STATE CPUS S:C:T ...
                    node_names, partition, state = parts[0], parts[2], parts[3]
                    cpus = parts[4] if len(parts) > 4 else "N/A"
                else:
                    # sinfo (default): PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
                    partition, state = parts[0], parts[4]
                    node_names = " ".join(parts[5:]) if len(parts) > 5 else ""
                    cpus = "N/A"
                gpus = "N/A"

            partition = partition.replace("*", "").strip()
            node_names = re.sub(r"\([^)]*\)", "", node_names).strip()
            display = self._state_display(state)

            for n_name in self._expand_nodelist(node_names):
                if not n_name:
                    continue
                if n_name not in nodes_dict:
                    nodes_dict[n_name] = ClusterNode(
                        name=n_name,
                        partition=partition,
                        state_raw=state,
                        state_display=display,
                        cpus=cpus,
                        gpus=self._fmt_gpus(gpus),
                        active_jobs=active_jobs.get(n_name, 0)
                    )

        return list(nodes_dict.values())

    @staticmethod
    def _state_display(state: str) -> str:
        """Convierte el estado crudo de SLURM en etiquetas legibles para la GUI."""
        s = state.lower()
        if "idle" in s:
            return "Libre"
        if "alloc" in s or "busy" in s:
            return "Ocupado"
        if "mix" in s:
            return "Mixto"
        if "down" in s or "drain" in s:
            return "No disponible"
        return state.strip("*").capitalize() or state

    @staticmethod
    def _fmt_cpus(cpus_raw: str, cpus_detail: str) -> str:
        """Formatea la columna de CPUs: total (libres) si hay detalle ``%C``."""
        if cpus_detail not in ("", "N/A") and "/" in cpus_detail:
            sub = cpus_detail.split("/")
            if len(sub) == 4:
                try:
                    return f"{int(sub[3])} ({sub[1]} libres)"
                except ValueError:
                    pass
        return cpus_raw if cpus_raw not in ("", "N/A") else "N/A"

    @staticmethod
    def _fmt_gpus(gpus: str) -> str:
        """Normaliza la columna de GPUs, convirtiendo '(null)' en 'N/A'."""
        if gpus in ("(null)", "null", "None", "N/A", ""):
            return "N/A"
        return gpus

    def _expand_nodelist(self, nodelist_str: str) -> List[str]:
        """Expande la sintaxis compacta de rangos de nodos de SLURM.

        Soporta listas mixtas como ``gpu2,toko[01-02]`` y rangos con varios
        segmentos como ``node[01-03,05]``.

        Args:
            nodelist_str (str): Cadena de lista de nodos (ej. 'node[01-03]').

        Returns:
            List[str]: Lista con los nombres individuales de cada nodo.
        """
        result: List[str] = []
        parts = self._split_respecting_brackets(nodelist_str)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            match = re.match(r"^([a-zA-Z0-9_-]+)\[([0-9,-]+)\]$", part)
            if not match:
                result.append(part)
                continue
            prefix = match.group(1)
            for r in match.group(2).split(","):
                if "-" in r:
                    start, end = r.split("-")
                    width = len(start)
                    for i in range(int(start), int(end) + 1):
                        result.append(f"{prefix}{str(i).zfill(width)}")
                else:
                    result.append(f"{prefix}{r}")
        return result

    @staticmethod
    def _split_respecting_brackets(nodelist_str: str) -> List[str]:
        """Divide una nodelist por comas, ignorando las comas dentro de '[...]'."""
        parts: List[str] = []
        current = ""
        depth = 0
        for ch in nodelist_str:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        return parts

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
