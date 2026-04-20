"""Módulo de ejecución de simulaciones LAMMPS en cluster.

CLI para gestionar jobs en SLURM: bpm run, bpm dry-run, bpm status.

Física: Ejecuta simulaciones de erosión de cama por impacto de proyectil
usando LAMMPS con potencial JKR para modelar contacto adhesivo.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(THIS_DIR))

from config import load_execution_config, validate_config
from executor import (
    check_jobs_status,
    create_sim_dir,
    prepare_sim_input,
    run_mpirun,
    submit_job,
)
from scheduler import JobSpec, count_jobs, generate_jobs, select_ntasks, select_node


def create_parser() -> ArgumentParser:
    """Crea el parser de argumentos para el CLI."""
    parser = ArgumentParser(
        prog="bpm",
        description="Ejecutor de simulaciones LAMMPS en cluster SLURM.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Ejecutar simulaciones")
    run_parser.add_argument(
        "--config",
        type=Path,
        help="Ruta al archivo config.toml",
    )
    run_parser.add_argument(
        "--wait",
        action="store_true",
        help="Esperar a que terminen los jobs",
    )
    run_parser.add_argument(
        "--node",
        type=str,
        help="Nodo específico (default: round-robin)",
    )

    dry_parser = sub.add_parser("dry-run", help="Mostrar jobs sin ejecutar")
    dry_parser.add_argument(
        "--config",
        type=Path,
        help="Ruta al archivo config.toml",
    )
    dry_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles de cada job",
    )

    status_parser = sub.add_parser("status", help="Ver estado de jobs")
    status_parser.add_argument(
        "job_ids",
        type=int,
        nargs="+",
        help="IDs de jobs en SLURM",
    )

    return parser


def cmd_run(args: Namespace) -> int:
    """Ejecuta el comando 'bpm run'."""
    cfg_path = args.config
    cfg = load_execution_config(cfg_path)

    errors = validate_config(cfg)
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    base_dir = Path(__file__).parent.parent

    template_slurm = base_dir / "execution" / "templates" / "slurm.sh.template"
    template_lammps = base_dir / "execution" / "templates" / "lammps.input"

    if not template_slurm.exists() or not template_lammps.exists():
        print("Error: Templates no encontrados", file=sys.stderr)
        return 1

    job_count = count_jobs(cfg)
    print(f"Generando {job_count} jobs...")

    if args.wait:
        print("Modo: submit + wait")
    else:
        print("Modo: submit only")

    submitted: list[int] = []
    for idx, job_spec in enumerate(generate_jobs(cfg)):
        node = args.node or select_node(cfg, idx)
        ntasks = select_ntasks(cfg)

        sim_dir = create_sim_dir(base_dir, job_spec.sim_dir)
        output_file = sim_dir / "output_trayectoria.dump"

        if output_file.exists():
            print(f"Skipped (ya existe): {job_spec.job_id}")
            continue

        script = render_slurm_script(
            template_slurm.read_text(),
            job_spec,
            node,
            ntasks,
            cfg.cluster.partition,
        )

        if args.wait:
            sim_dir = create_sim_dir(base_dir, job_spec.sim_dir)
            prepare_sim_input(sim_dir, template_lammps, job_spec, cfg)

            result = run_mpirun(sim_dir, cfg, job_spec, base_dir / cfg.simulation.exe)

            if result.returncode != 0:
                print(f"Job {job_spec.job_id} falló: {result.stderr}", file=sys.stderr)
                continue

            print(f"Job {job_spec.job_id} completado")
        else:
            try:
                slurm_id = submit_job(job_spec, cfg, base_dir, script)
                submitted.append(slurm_id)
                print(f"Enviado: {job_spec.job_id} -> SLURM {slurm_id}")
            except RuntimeError as e:
                print(f"Error submitting {job_spec.job_id}: {e}", file=sys.stderr)

    if submitted:
        print(f"\nEnviados {len(submitted)} jobs")
        print(f"IDs: {', '.join(map(str, submitted))}")

    return 0


def cmd_dry_run(args: Namespace) -> int:
    """Ejecuta el comando 'bpm dry-run'."""
    cfg_path = args.config
    cfg = load_execution_config(cfg_path)

    errors = validate_config(cfg)
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    job_count = count_jobs(cfg)
    print(f"Total de jobs: {job_count}")

    if args.verbose:
        for idx, job_spec in enumerate(generate_jobs(cfg)):
            print(f"  [{idx+1}] {job_spec.job_id}")
            print(f"       fill={job_spec.fill} vel={job_spec.vel} angle={job_spec.angle}")
            print(f"       xmove={job_spec.xmove} ymove={job_spec.ymove}")
    else:
        for job_spec in generate_jobs(cfg):
            print(job_spec.job_id)

    return 0


def cmd_status(args: Namespace) -> int:
    """Ejecuta el comando 'bpm status'."""
    job_ids = args.job_ids

    for sid, status in check_jobs_status(job_ids):
        state = status.state
        exit_code = status.exit_code
        code_str = f" (exit {exit_code})" if exit_code is not None else ""
        print(f"{sid}: {state}{code_str}")

    return 0


def render_slurm_script(
    template: str,
    job_spec: JobSpec,
    node: str,
    ntasks: int,
    partition: str,
) -> str:
    """Renderiza el template SLURM con los valores del job."""
    replacements = {
        "${JOB_ID}": job_spec.job_id,
        "${NTASKS}": str(ntasks),
        "${NODE}": node,
        "${PARTITION}": partition,
    }

    content = template
    for key, val in replacements.items():
        content = content.replace(key, val)

    return content


def main() -> int:
    """Entry point principal del CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    if args.command == "dry-run":
        return cmd_dry_run(args)
    if args.command == "status":
        return cmd_status(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
