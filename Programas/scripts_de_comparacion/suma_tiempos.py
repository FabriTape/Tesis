#!/usr/bin/env python3
import os
import re
from datetime import datetime

NAS_DIR = "/home/fbazzoni/nas"
EXCLUDE = {"sims_25_moves"}

def find_final_output(subdir):
    gz_files = []
    for f in os.listdir(subdir):
        m = re.match(r'output\.(\d+)\.gz$', f)
        if m:
            gz_files.append((int(m.group(1)), f))
    if gz_files:
        gz_files.sort()
        _, name = gz_files[-1]
        return os.path.join(subdir, name)
    cand = os.path.join(subdir, "output_trayectoria.dump")
    if os.path.isfile(cand):
        return cand
    return None

def get_cores_from_log(subdir):
    log = os.path.join(subdir, "log.lammps")
    if not os.path.isfile(log):
        return None
    with open(log) as f:
        for line in f:
            m = re.search(r'(\d+)\s+by\s+(\d+)\s+by\s+(\d+)\s+MPI\s+processor\s+grid', line)
            if m:
                return int(m.group(1)) * int(m.group(2)) * int(m.group(3))
    return None

def get_cores(parent_name, subdir):
    # Try log.lammps first for accuracy
    c = get_cores_from_log(subdir)
    if c:
        return c
    # Fallback: extract from parent dir name
    m = re.search(r'cores_(\d+)', parent_name)
    if m:
        return int(m.group(1))
    return None

def main():
    all_sims = []
    skipped = 0

    for entry in sorted(os.listdir(NAS_DIR)):
        parent = os.path.join(NAS_DIR, entry)
        if not os.path.isdir(parent) or entry in EXCLUDE:
            continue
        subdirs = sorted([
            os.path.join(parent, d)
            for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d))
        ])
        if not subdirs:
            continue

        for sub in subdirs:
            pre_run = os.path.join(sub, "pre-run_with_projectile.data")
            if not os.path.isfile(pre_run):
                skipped += 1
                continue
            final = find_final_output(sub)
            if final is None:
                skipped += 1
                continue
            t_pre = os.path.getmtime(pre_run)
            t_final = os.path.getmtime(final)
            dt = t_final - t_pre
            if dt <= 0:
                skipped += 1
                continue
            cores = get_cores(entry, sub)
            if cores is None:
                skipped += 1
                continue

            core_h = dt / 3600 * cores
            all_sims.append((t_pre, t_final, dt, cores, core_h, os.path.relpath(sub, NAS_DIR)))

    all_sims.sort(key=lambda r: r[0])

    # Print per-sim detail
    total_core_h = 0.0
    for s in all_sims:
        t_pre, t_final, dt, cores, core_h, path = s
        total_core_h += core_h
        pre_str = datetime.fromtimestamp(t_pre).strftime("%Y-%m-%d %H:%M")
        print(f"{dt/3600:7.2f} h × {cores:2d} cores = {core_h:8.1f} core-h  {pre_str}  {path}")

    print()
    total_sim_h = sum(s[2] for s in all_sims) / 3600
    print(f"Simulaciones: {len(all_sims)}  |  Saltadas: {skipped}")
    print(f"Suma de duraciones individuales: {total_sim_h:.2f} h = {total_sim_h/24:.2f} d")
    print(f"Total core-horas:                {total_core_h:.2f} core-h = {total_core_h/24:.2f} core-d")
    print()
    print(f"Tiempo equivalente si siempre se usaran todos los nucleos disponibles:")
    print(f"  Con 32 nucleos: {total_core_h/32:.2f} h = {total_core_h/32/24:.2f} d")
    print(f"  Con 64 nucleos: {total_core_h/64:.2f} h = {total_core_h/64/24:.2f} d")
    print(f"  Con 128 nucleos (2 nodos): {total_core_h/128:.2f} h = {total_core_h/128/24:.2f} d")

if __name__ == "__main__":
    main()
