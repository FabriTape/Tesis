#!/usr/bin/env python3
"""CLI entry point para execution: bpm run, bpm dry-run, bpm status.

Comandos disponibles:
- run: Ejecutar simulaciones en cluster SLURM.
- dry-run: Mostrar jobs sin ejecutar.
- status: Ver estado de jobs en SLURM.
- check: Verificar integridad post-simulación.
- analyze: Merge + stats + visualización.
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from matplotlib.figure import Figure

_path = Path(__file__).resolve()
exec_dir = _path.parent
pkg_dir = exec_dir.parent

sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(exec_dir))

from config import load_execution_config, validate_config
from executor import check_jobs_status, create_sim_dir, prepare_sim_input, run_mpirun, submit_job
from scheduler import JobSpec, count_jobs, generate_jobs, select_ntasks, select_node

from io.merge import merge_with_no_erosion
import viz
import polars as pl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def create_parser() -> ArgumentParser:
    """Crea el parser de argumentos para el CLI."""
    parser = ArgumentParser(
        prog="bpm",
        description="Ejecutor de simulaciones LAMMPS en cluster SLURM.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Ejecutar simulaciones")
    run_parser.add_argument("--config", type=Path, help="Ruta al archivo config.toml")
    run_parser.add_argument("--wait", action="store_true", help="Esperar a que terminen los jobs")
    run_parser.add_argument("--node", type=str, help="Nodo específico (default: round-robin)")

    dry_parser = sub.add_parser("dry-run", help="Mostrar jobs sin ejecutar")
    dry_parser.add_argument("--config", type=Path, help="Ruta al archivo config.toml")
    dry_parser.add_argument("--verbose", action="store_true", help="Mostrar detalles de cada job")

    status_parser = sub.add_parser("status", help="Ver estado de jobs")
    status_parser.add_argument("job_ids", type=int, nargs="+", help="IDs de jobs en SLURM")

    check_parser = sub.add_parser(
        "check",
        help="Verificar integridad post-simulación",
    )
    check_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directorio con resultados de simulaciones",
    )

    analyze_parser = sub.add_parser(
        "analyze",
        help="Merge + stats + visualización",
    )
    analyze_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directorio con resultados de simulaciones",
    )

    return parser


def render_slurm_script(template: str, job_spec: JobSpec, node: str, ntasks: int, partition: str) -> str:
    """Renderiza el template SLURM."""
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


def cmd_run(args: Namespace) -> int:
    """Ejecuta el comando 'bpm run'."""
    cfg = load_execution_config(args.config)
    errors = validate_config(cfg)
    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    base_dir = pkg_dir
    template_slurm = exec_dir / "templates" / "slurm.sh.template"
    template_lammps = exec_dir / "templates" / "lammps.input"

    if not template_slurm.exists() or not template_lammps.exists():
        print("Error: Templates no encontrados", file=sys.stderr)
        return 1

    job_count = count_jobs(cfg)
    print(f"Generando {job_count} jobs...")

    submitted = []
    for idx, job_spec in enumerate(generate_jobs(cfg)):
        node = args.node or select_node(cfg, idx)
        ntasks = select_ntasks(cfg)
        script = render_slurm_script(template_slurm.read_text(), job_spec, node, ntasks, cfg.cluster.partition)

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
                slurm_id = submit_job(job_spec, cfg, base_dir, script, args.config)
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
    cfg = load_execution_config(args.config)
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
    for sid, status in check_jobs_status(args.job_ids):
        state = status.state
        exit_code = status.exit_code
        code_str = f" (exit {exit_code})" if exit_code is not None else ""
        print(f"{sid}: {state}{code_str}")
    return 0


def _check_single_simulation(sim_dir: Path) -> tuple[bool, str]:
    """Verifica integridad de una simulación individual.

    Args:
        sim_dir: Directorio de la simulación.

    Returns:
        Tupla (éxito, mensaje).
    """
    name = sim_dir.name

    output_dump = sim_dir / "output_trayectoria.dump"
    output_gz = list(sim_dir.glob("output.*.gz"))
    has_output = output_dump.exists() or len(output_gz) > 0

    log_file = sim_dir / "log.lammps"
    has_log = log_file.exists()

    wall_time_ok = False
    if has_log:
        try:
            content = log_file.read_text()
            wall_time_ok = "Total wall time" in content
        except Exception:
            pass

    issues: list[str] = []
    if not has_output:
        issues.append("sin output")
    if not has_log:
        issues.append("sin log")
    if not wall_time_ok:
        issues.append("sin wall time")

    if issues:
        return False, f"{name}: {', '.join(issues)}"
    return True, f"{name}: OK"


def cmd_check(args: Namespace) -> int:
    """Ejecuta el comando 'bpm check'.

    Verifica integridad post-simulación:
    - Existe directorio de cada simulación.
    - Existe output_trayectoria.dump o output.*.gz.
    - El log tiene 'Total wall time'.
    - Genera archivo sims_a_repetir.txt con las fallidas.
    """
    results = args.results.resolve()

    if not results.is_dir():
        print(f"Error: Resultados no encontrado: {results}", file=sys.stderr)
        return 1

    sim_dirs = sorted(
        d for d in results.iterdir() if d.is_dir() and d.name.startswith("erosion_bed_SI_")
    )

    if not sim_dirs:
        print(f"Warning: No hay dirs erosion_bed_SI_* en {results}", file=sys.stderr)
        return 0

    failed: list[str] = []
    passed: list[str] = []

    for sim_dir in sim_dirs:
        success, msg = _check_single_simulation(sim_dir)
        if success:
            passed.append(msg)
        else:
            failed.append(msg)

    print("\n=== Verificación de integridad ===")
    print(f"Total: {len(sim_dirs)}")
    print(f"OK: {len(passed)}")
    print(f"Fallidas: {len(failed)}")

    if passed:
        print("\nSimulaciones OK:")
        for p in passed:
            print(f"  {p}")

    if failed:
        print("\nSimulaciones a repetir:")
        for f in failed:
            print(f"  {f}")

        output_file = results / "sims_a_repetir.txt"
        with output_file.open("w") as fp:
            for f in failed:
                fp.write(f"{f.split(':')[0]}\n")
        print(f"\nGuardado en: {output_file}")

    return 0


def cmd_analyze(args: Namespace) -> int:
    """Ejecuta el comando 'bpm analyze'.

    Realiza análisis completo:
    - Merge de parquet en merge_resultados.csv.
    - Genera estadísticas en stats/.
    - Genera figuras en viz/.
    """
    results = args.results.resolve()

    if not results.is_dir():
        print(f"Error: Resultados no encontrado: {results}", file=sys.stderr)
        return 1

    print("\n=== Análisis de resultados ===")
    print(f"Directorio: {results}")

    parquet_files = list(results.glob("*.parquet"))
    if not parquet_files:
        print("Warning: No hay archivos .parquet", file=sys.stderr)

    print("\n[1/3] Merge de parquet...")
    merged = merge_with_no_erosion(results)

    if merged.is_empty():
        print("Error: No se pudieron fusionar datos", file=sys.stderr)
        return 1

    merge_path = results / "merge_resultados.csv"
    merged.write_csv(merge_path)
    print(f"  Guardado: {merge_path} ({len(merged)} rows)")

    print("\n[2/3] Generando estadísticas...")
    stats_dir = results / "stats"
    stats_dir.mkdir(exist_ok=True)

    grouped = merged.group_by("ff").agg(pl.col("id").n_unique().alias("n_sims"))
    for row in grouped.iter_rows(named=True):
        ff = row["ff"]
        n_sims = row["n_sims"]
        print(f"  ff={ff}: {n_sims} simulaciones")

    print("\n[3/3] Generando visualizaciones...")
    viz_dir = results / "viz"
    viz_dir.mkdir(exist_ok=True)

    figures: list[Figure] = []

    unique_ff = merged["ff"].unique().to_list()
    loaded_ff = unique_ff[0] if unique_ff else None

    if loaded_ff is not None:
        try:
            fig = viz.plot_erosion_boxplot(merged, loaded_ff, viz_dir)
            if fig:
                figures.append(fig)
                print("  Boxplot guardado")
        except Exception as e:
            print(f"  Warning boxplot: {e}")

    unique_vel0 = merged["vel0"].unique().to_list()
    for vel in unique_vel0[:2]:
        try:
            fig = viz.plot_erosion_heatmap(merged, loaded_ff, vel, viz_dir)
            if fig and fig.get_axes():
                figures.append(fig)
                print(f"  Heatmap vel={vel} guardado")
        except Exception as e:
            print(f"  Warning heatmap: {e}")

    if figures:
        pdf_path = viz_dir / "resultados_erosion.pdf"
        viz.save_all_figures_pdf(figures, pdf_path)
        print(f"  PDF: {pdf_path}")

    print("\n=== Complete ===")
    print(f"  stats/: {stats_dir}")
    print(f"  viz/: {viz_dir}")
    return 0


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
    if args.command == "check":
        return cmd_check(args)
    if args.command == "analyze":
        return cmd_analyze(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
