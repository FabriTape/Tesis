"""
Controlador para retomar simulaciones incompletas de LAMMPS.
Escanea la carpeta 'Buffer/sims_SI...', identifica carpetas de simulación
sin 'post-run_final.data' pero con archivos restart, y las reanuda.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(
        description="Controlador para retomar simulaciones incompletas de LAMMPS."
    )
    parser.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Muestra las acciones que se tomarían sin ejecutar",
        default=False,
    )
    args = parser.parse_args()

    # Asume que se ejecuta desde la raíz del proyecto
    base_dir = Path.cwd()
    config_path = base_dir / "config.toml"
    input_path = base_dir / "resume_collision.input"
    buffer_dir = base_dir / "Buffer"

    # Fallback por si el config está en scripts/ directamente
    if not config_path.exists():
        config_path = base_dir / "config.toml"

    if not config_path.exists():
        print(f"Error: No se encontró {config_path}")
        sys.exit(1)

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Cargar variables del config.toml
    sim_config = config["simulation"]
    mat_config = config["material"]

    cores = sim_config["cores"]
    exe = sim_config["exe"]
    proc_x, proc_y, proc_z = sim_config["processors"]
    y_mod = mat_config["young_modulus"]
    poisson = mat_config["poisson"]
    surf_ener = mat_config["surface_energy"]
    ndumps = sim_config["ndumps"]
    nthermo = sim_config["nthermo"]
    default_nsteps = config["sweep"]["total_steps"][0]

    exe_path = base_dir / exe

    print("Iniciando búsqueda de simulaciones incompletas en Buffer/ ...")
    if not buffer_dir.exists():
        print("No existe el directorio Buffer.")
        return

    # 1. Iterar sobre las carpetas de los lotes (sims_SI_erosion_bed_...)
    for batch_dir in buffer_dir.glob("sims_SI_erosion_*"):
        if not batch_dir.is_dir():
            continue

        # 2. Iterar sobre las carpetas de las simulaciones individuales adentro del lote
        for sim_dir in batch_dir.glob("erosion_bed_SI_*"):
            if not sim_dir.is_dir():
                continue

            if (sim_dir / "post-run_final.data").exists():
                continue  # Esta simulación terminó correctamente

            # Buscar cuál archivo restart es el más reciente
            res_a = sim_dir / "res_A.rst"
            res_b = sim_dir / "res_B.rst"
            restart_file = None

            if res_a.exists() and res_b.exists():
                if os.path.getmtime(res_a) > os.path.getmtime(res_b):
                    restart_file = "res_A.rst"
                else:
                    restart_file = "res_B.rst"
            elif res_a.exists():
                restart_file = "res_A.rst"
            elif res_b.exists():
                restart_file = "res_B.rst"

            if restart_file:
                print(f"\n[+] Retomando: {sim_dir.name}\n    Desde: {restart_file}")

                # Intentar deducir los nsteps originales del salida.log
                local_nsteps = default_nsteps
                salida_log = sim_dir / "salida.log"
                if salida_log.exists():
                    with open(salida_log, "r") as f_log:
                        for line in f_log:
                            if "nsteps" in line and "velocity" in line:
                                parts = line.split()
                                if "nsteps" in parts:
                                    idx = parts.index("nsteps")
                                    local_nsteps = parts[idx + 1]
                                break

                cmd = [
                    "mpirun",
                    "--allow-run-as-root",
                    "-np",
                    str(cores),
                    str(exe_path),
                    "-in",
                    str(input_path),
                    "-v",
                    "restart_file",
                    restart_file,
                    "-v",
                    "young_modulus",
                    str(y_mod),
                    "-v",
                    "poisson",
                    str(poisson),
                    "-v",
                    "surface_energy",
                    str(surf_ener),
                    "-v",
                    "nsteps",
                    str(local_nsteps),
                    "-v",
                    "ndumps",
                    str(ndumps),
                    "-v",
                    "nthermo",
                    str(nthermo),
                    "-v",
                    "proc_x",
                    str(proc_x),
                    "-v",
                    "proc_y",
                    str(proc_y),
                    "-v",
                    "proc_z",
                    str(proc_z),
                ]
                if args.dry:
                    print(f"    -> Ejecutando: {cmd}")
                    print(f"Simulacion a ejecutar: {sim_dir.name}")
                else:
                    # Guardamos el output en resume.log
                    with open(sim_dir / "resume.log", "w") as log_file:
                        subprocess.run(
                            cmd, cwd=sim_dir, stdout=log_file, stderr=subprocess.STDOUT
                        )

                    print(f"    -> Finalizado: {sim_dir.name}")


if __name__ == "__main__":
    main()
