#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlador de envío de simulaciones LAMMPS (idempotente).

Lee el formulario central ``config.toml`` (sección ``[[experiment]]``) y envía
los jobs de Slurm que faltan correr. Se ejecuta desde ``Programas/cluster/scripts/``
EN EL CLUSTER, donde existen ``Buffer/``, ``datas/``, ``sbatch`` y ``squeue``.

Flujo:
    1. Lee ``config.toml`` y expande el producto cartesiano de cada experimento
       (fills x vels x angles x moves^2 x orientations).
    2. Agrupa por ``ff`` (resuelve grid/cores/processors desde ``grid_por_ff``)
       y parte la lista en chunks cuya estimación de muro sea <= ``max_horas_job``.
    3. Descarta sims ya completos (``post-run_final.data`` en ``Buffer/``) y
       chunks cuyo job siga activo (manifiesto ``queue/jobs.tsv`` + ``squeue``).
    4. Para cada chunk pendiente: escribe el TOML mergeado
       (``configs/config_<jobid>.toml``) y la lista exacta de sims
       (``configs/sims_<jobid>.tsv``), y llama a ``sbatch`` con los parámetros
       del job (job-name, ntasks-per-node, nodelist, partition).

Idempotencia: un job cortado deja sims incompletas en disco; el runner las
re-ejecuta frescas. Este controlador solo evita enviar dos veces lo mismo
que siga activo.

Uso:
    python enqueue_sims.py [--dry-run] [--tasks N]
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


BASE_DIR = Path.cwd()
CONFIG_PATH = BASE_DIR / "config.toml"
BUFFER_DIR = BASE_DIR / "Buffer"
DATAS_DIR = BASE_DIR / "datas"
CONFIGS_DIR = BASE_DIR / "configs"
QUEUE_DIR = BASE_DIR / "queue"
LOGS_DIR = BASE_DIR / "logs"
MANIFEST_PATH = QUEUE_DIR / "jobs.tsv"
WRAPPER = BASE_DIR / "slurm_submit_JKR.sh"

SBATCH_RE = re.compile(r"batch job\s+(\d+)")

MANIFEST_HEADER = "# clave\tjobid\testado\texp\tff\tn_sims\test_horas\tcreado\n"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Lee el config.toml base."""
    with path.open("rb") as fh:
        return tomllib.load(fh)


def ff_key(ff: float | int | str) -> str:
    """Normaliza un valor de ff a la clave usada en grid_por_ff/est_horas_por_ff."""
    if isinstance(ff, str):
        return ff
    return str(ff)


def bc_mul(move: int | float, radius: float) -> str:
    """Multiplica move*radius con bc, para replicar EXACTO el nombre del simdir
    que genera el runner en bash (bc omite ceros: '.76', '0', '2.28')."""
    try:
        res = subprocess.run(
            ["bc"], input=f"{move} * {radius}\n", capture_output=True, text=True, timeout=10
        )
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return str(move * radius)


def job_active(jobid: str) -> bool:
    """True si el job sigue en cola o corriendo en Slurm."""
    if not jobid:
        return False
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", jobid], capture_output=True, text=True, timeout=30
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True  # sin squeue disponible: no reenviar lo ya enviado


def parse_jobid(sbatch_out: str) -> str:
    match = SBATCH_RE.search(sbatch_out)
    if not match:
        raise RuntimeError(f"No se pudo extraer el job id de: {sbatch_out}")
    return match.group(1)


def toml_value(value: object) -> str:
    """Serializa un valor simple a sintaxis TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[ " + ", ".join(toml_value(v) for v in value) + " ]"
    raise TypeError(f"Tipo no soportado en TOML: {type(value)!r} ({value!r})")


def write_job_config(jobid: str, params: dict) -> None:
    """Escribe el TOML mergeado del job (lo lee read_config.py → config_vars.sh)."""
    sections = [
        ("simulation", params["simulation"]),
        ("sweep", params["sweep"]),
        ("material", params["material"]),
        ("slurm", params["slurm"]),
    ]
    lines = [f"# Auto-generado por enqueue_sims.py para el job {jobid} — NO editar\n"]
    for name, values in sections:
        lines.append(f"[{name}]")
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    (CONFIGS_DIR / f"config_{jobid}.toml").write_text("\n".join(lines))


def write_sim_list(jobid: str, sims: list[dict]) -> None:
    """Escribe la lista exacta de sims del chunk: orientation fill vel angle xm ym nsteps."""
    lines = [
        "\t".join(
            [
                s["orientation"],
                str(s["fill"]),
                str(s["vel"]),
                str(s["angle"]),
                s["xm"],
                s["ym"],
                str(s["nsteps"]),
            ]
        )
        for s in sims
    ]
    (CONFIGS_DIR / f"sims_{jobid}.tsv").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Manifiesto queue/jobs.tsv
# ---------------------------------------------------------------------------
def load_manifest() -> dict[str, dict]:
    """Devuelve {clave: {jobid, estado, exp, ff, n_sims, est_horas, creado}}."""
    rows: dict[str, dict] = {}
    if not MANIFEST_PATH.exists():
        return rows
    for line in MANIFEST_PATH.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        clave, jobid, estado, exp, ff, n_sims, est_horas, creado = line.split("\t")
        rows[clave] = {
            "jobid": jobid,
            "estado": estado,
            "exp": exp,
            "ff": ff,
            "n_sims": int(n_sims),
            "est_horas": float(est_horas),
            "creado": creado,
        }
    return rows


def save_manifest(rows: dict[str, dict]) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    out = [MANIFEST_HEADER]
    for clave in sorted(rows):
        r = rows[clave]
        out.append(
            "\t".join(
                [
                    clave,
                    r["jobid"],
                    r["estado"],
                    r["exp"],
                    r["ff"],
                    str(r["n_sims"]),
                    str(r["est_horas"]),
                    r["creado"],
                ]
            )
        )
    MANIFEST_PATH.write_text("".join(out))


# ---------------------------------------------------------------------------
# Construcción de sims
# ---------------------------------------------------------------------------
def build_groups(config: dict) -> list[tuple[str, float, dict]]:
    """Expande cada [[experiment]] agrupado por ff.

    Returns:
        Lista de (exp_label, ff, group) donde group tiene:
            grid, cores, processors, est_horas, sims, params
    """
    sim_base = config["simulation"]
    sweep = config["sweep"]
    material = config["material"]
    grid_por_ff: dict = sweep.get("grid_por_ff", {})
    est_horas_por_ff: dict = sweep.get("est_horas_por_ff", {})
    radius = sweep["grain_radius_um"]

    groups: list[tuple[str, float, dict]] = []
    for idx, exp in enumerate(config["experiment"], start=1):
        fills: list = exp["fills"]
        vels: list = exp["vels"]
        angles: list = exp["angles"]
        moves: list = exp["moves"]
        orientations: list = exp.get("orientations", [""])
        np_ = exp.get("np", sim_base.get("np", 1))
        nsteps: list = exp.get("nsteps") or sweep.get("nsteps", [2000000])

        # Pares (xm, ym) en el mismo orden que el runner (moves x moves)
        move_pairs = [(bc_mul(mx, radius), bc_mul(my, radius)) for mx in moves for my in moves]

        for fill in fills:
            key = ff_key(fill)
            grid = grid_por_ff.get(key)
            if grid is None:
                raise ValueError(
                    f"[experiment {idx}] fill={fill}: falta grid_por_ff para '{key}'"
                )
            cores = math.prod(grid)
            est_horas = est_horas_por_ff.get(key, 5.0)

            sims: list[dict] = []
            for vi, vel in enumerate(vels):
                ns = nsteps[vi] if vi < len(nsteps) else nsteps[0]
                for angle in angles:
                    for xm, ym in move_pairs:
                        for orientation in orientations:
                            sims.append(
                                {
                                    "exp_idx": idx,
                                    "fill": fill,
                                    "vel": vel,
                                    "angle": angle,
                                    "xm": xm,
                                    "ym": ym,
                                    "nsteps": ns,
                                    "orientation": orientation,
                                    "simdir": f"erosion_bed_SI_ff_{fill}_np_{np_}"
                                    f"_vel_{vel}_angle_{angle}_move_{xm}_{ym}_0",
                                }
                            )

            params = {
                "simulation": {
                    "np": np_,
                    "cores": cores,
                    "processors": list(grid),
                    "ndumps": sim_base["ndumps"],
                    "nthermo": sim_base["nthermo"],
                    "exe": sim_base["exe"],
                    "timestep": sim_base["timestep"],
                },
                "sweep": {
                    "fills": [fill],
                    "vels": vels,
                    "angles": angles,
                    "moves": moves,
                    "nsteps": nsteps,
                    "orientations": orientations,
                    "grain_radius_um": radius,
                    "xrotate": sweep.get("xrotate", 1),
                    "yrotate": sweep.get("yrotate", 0),
                    "zrotate": sweep.get("zrotate", 0),
                    "guardar_restart": sweep.get("guardar_restart", False),
                },
                "material": {
                    "young_modulus": material["young_modulus"],
                    "poisson": material["poisson"],
                    "surface_energy": material["surface_energy"],
                    "density": material["density"],
                },
            }
            groups.append((f"exp{idx}", fill, {"sims": sims, "grid": grid, "cores": cores,
                                               "est_horas": est_horas, "params": params}))
    return groups


def batch_dir(group: dict) -> str:
    """Nombre del directorio de lote para una orientación (misma plantilla que el runner)."""
    p = group["params"]["sweep"]
    ff = str(p["fills"][0])
    vels = "_".join(str(v) for v in p["vels"])
    moves = "_".join(str(m) for m in p["moves"])
    angles = "_".join(str(a) for a in p["angles"])
    np_ = group["params"]["simulation"]["np"]
    return (
        f"sims_SI_erosion_bed_100um_"
        f"{{ori}}_ff_{ff}_vel_{vels}_move_{moves}_NP_{np_}_angle_{angles}"
    )


def is_completed(group: dict, sim: dict) -> bool:
    """True si el sim ya terminó (existe post-run_final.data)."""
    template = batch_dir(group)
    simdir = BUFFER_DIR / template.format(ori=sim["orientation"]) / sim["simdir"]
    return (simdir / "post-run_final.data").exists()


def validate_data_files(group: dict) -> None:
    """Advierte y quita orientaciones cuyo .data no existe para su ff."""
    if not DATAS_DIR.is_dir():
        print("  [aviso] no se encontró datas/, se omite validación de archivos .data")
        return
    p = group["params"]["sweep"]
    fills = p["fills"]
    keep = []
    for sim in group["sims"]:
        data = DATAS_DIR / f"box_fill_{sim['fill']}_SI_100um_{sim['orientation']}.data"
        if data.exists():
            keep.append(sim)
        else:
            print(
                f"  [aviso] no existe {data.name}; se omite la orientación "
                f"'{sim['orientation']}' para ff={sim['fill']}"
            )
    if fills and keep:
        group["sims"] = keep


# ---------------------------------------------------------------------------
# Envío
# ---------------------------------------------------------------------------
def submit_job(name: str, ntasks: int, slurm: dict, wrapper_arg: str) -> str:
    """Envía el job. Los archivos del chunk (config/sims) se escriben ANTES,
    con nombre clave (= wrapper_arg), para evitar la carrera de que el job
    arranque antes de que existan."""
    cmd = [
        "sbatch",
        f"--job-name={name}",
        f"--ntasks-per-node={ntasks}",
        f"--nodelist={slurm.get('nodelist', '')}",
        f"--partition={slurm.get('partition', '')}",
        str(WRAPPER),
        wrapper_arg,
    ]
    print("  sbatch:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    jobid = parse_jobid(out.stdout)
    print(f"  → job {jobid}")
    return jobid


def split_chunks(group: dict, max_horas: float, tasks: int) -> list[list[dict]]:
    """Parte la lista completa del (exp, ff) en chunks que estimen <= max_horas."""
    est = group["est_horas"]
    per_round = max(1, int(max_horas // est))  # rondas que entran en el muro
    chunk_size = max(1, tasks * per_round)
    sims = group["sims"]
    return [sims[i : i + chunk_size] for i in range(0, len(sims), chunk_size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlador de envío de sims LAMMPS")
    parser.add_argument("--dry-run", action="store_true", help="solo muestra el plan, no envía")
    parser.add_argument("--tasks", type=int, default=None, help="sims concurrentes por job (override)")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print(f"Error: no se halló {CONFIG_PATH}", file=sys.stderr)
        return 1

    config = load_config(CONFIG_PATH)
    sweep = config["sweep"]
    slurm = config.get("slurm", {})
    max_horas = float(sweep.get("max_horas_job", 72))
    tasks = args.tasks or int(slurm.get("tasks", 8))

    groups = build_groups(config)
    manifest = load_manifest()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    total_pend = 0
    for exp_label, ff, group in groups:
        validate_data_files(group)
        chunks = split_chunks(group, max_horas, tasks)
        print(f"\n== {exp_label} ff={ff} (cores={group['cores']}, est={group['est_horas']}h/sim) ==")
        for c, chunk in enumerate(chunks):
            clave = f"{exp_label}_ff_{ff_key(ff)}_chunk{c}"
            n_done = sum(1 for s in chunk if is_completed(group, s))
            n_pend = len(chunk) - n_done
            est_wall = math.ceil(max(1, n_pend) / tasks) * group["est_horas"]
            row = manifest.get(clave)

            if n_pend == 0:
                print(f"  chunk {c}: {len(chunk)} sims, todas completas — sin acción")
                continue

            if row and row["estado"] == "enviado" and job_active(row["jobid"]):
                print(
                    f"  chunk {c}: {len(chunk)} sims ({n_pend} pendientes) — ya enviado "
                    f"(job {row['jobid']} activo)"
                )
                continue

            print(
                f"  chunk {c}: {len(chunk)} sims ({n_pend} pendientes, est_wall≈{est_wall:.1f}h) — ENVIAR"
            )
            total_pend += n_pend
            if args.dry_run:
                continue

            ntasks = tasks * group["cores"]
            params = dict(group["params"])
            params["slurm"] = {
                "partition": slurm.get("partition", ""),
                "nodelist": slurm.get("nodelist", ""),
                "nodes": slurm.get("nodes", 1),
                "tasks": tasks,
                "ntasks": ntasks,
            }
            # Escribir antes de sbatch: el job usa config_<clave>.toml y
            # sims_<clave>.tsv apenas arranca.
            write_job_config(clave, params)
            write_sim_list(clave, chunk)
            jobid = submit_job(
                name=clave, ntasks=ntasks,
                slurm={"partition": params["slurm"]["partition"],
                       "nodelist": params["slurm"]["nodelist"]},
                wrapper_arg=clave,
            )
            manifest[clave] = {
                "jobid": jobid,
                "estado": "enviado",
                "exp": exp_label,
                "ff": str(ff),
                "n_sims": len(chunk),
                "est_horas": round(est_wall, 2),
                "creado": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            }

    if not args.dry_run:
        save_manifest(manifest)

    print(f"\nSims pendientes a enviar: {total_pend}")
    print("(dry-run)" if args.dry_run else "listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
