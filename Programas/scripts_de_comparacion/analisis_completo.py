#!/usr/bin/env python3
import os
import re

DIRS = [
    ("nas", "/home/fbazzoni/nas", {"sims_25_moves", "vigilante.py", "otros"}),
    ("sims_emmanuel", "/home/fbazzoni/sims_emmanuel", {"old_sims","sims_SI_erosion_bed_100um_ff_0.15-0.35_vels_all_move_0_NP_1_cores_32_v3","sims_SI_erosion_bed_100um_ff_0.15_vels_all_move_0_NP_1_cores_32","sims_SI_erosion_bed_100um_ff_0.35_vels_all_move_0_NP_1_cores_32"}),
    ("Buffer", "/home/fbazzoni/SIMULACIONES/Buffer", {"vigilante.py"}),
]

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

def get_cores(subdir, parent_name):
    log = os.path.join(subdir, "log.lammps")
    if os.path.isfile(log):
        with open(log) as f:
            for line in f:
                m = re.search(r'(\d+)\s+by\s+(\d+)\s+by\s+(\d+)\s+MPI\s+processor\s+grid', line)
                if m:
                    return int(m.group(1)) * int(m.group(2)) * int(m.group(3))
    m = re.search(r'cores_(\d+)', parent_name)
    if m:
        return int(m.group(1))
    return None

def get_ff(subdir_name):
    if '_ff_0.35_' in subdir_name or '_ff_0.35/' in subdir_name:
        return 0.35
    elif '_ff_0.15_' in subdir_name or '_ff_0.15/' in subdir_name:
        return 0.15
    return None

def process_dir(label, path, exclude):
    all_sims = []
    skipped = 0

    for entry in sorted(os.listdir(path)):
        parent = os.path.join(path, entry)
        if not os.path.isdir(parent) or entry in exclude:
            continue
        subdirs_list = sorted([
            os.path.join(parent, d)
            for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d))
        ])
        if not subdirs_list:
            continue

        for sub in subdirs_list:
            ff = get_ff(os.path.basename(sub))
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
            cores = get_cores(sub, entry)
            if cores is None:
                skipped += 1
                continue
            all_sims.append((t_pre, dt, cores, dt/3600*cores, ff, entry))

    if not all_sims:
        return [], 0, 0, 0, 0

    all_sims.sort(key=lambda r: r[0])

    # Sub-totals by NP and FF
    totals = {}
    for s in all_sims:
        dt, cores, core_h, ff = s[1], s[2], s[3], s[4]
        np = "NP_3" if "NP_3" in s[5] else "NP_1"
        key = (np, ff)
        if key not in totals:
            totals[key] = {"count": 0, "sim_h": 0.0, "core_h": 0.0, "core_h_norm": 0.0}
        totals[key]["count"] += 1
        totals[key]["sim_h"] += dt / 3600
        totals[key]["core_h"] += core_h
        # Normalize: ff=0.35 has 2x particles, so divide core-h by 2
        totals[key]["core_h_norm"] += core_h if ff == 0.15 else core_h / 2

    total_core_h = sum(s[3] for s in all_sims)
    total_core_h_norm = sum(s[3] if s[4] == 0.15 else s[3]/2 for s in all_sims)
    total_sim_h = sum(s[1] for s in all_sims) / 3600

    # Print table
    print(f"\n{'='*90}")
    print(f"  {label}: {path}")
    print(f"{'='*90}")
    print(f"  {'(NP, FF)':<12} {'Sims':>5} {'Sim-h':>8} {'Core-h':>10} {'Core-h norm.':>12}  Nota")
    print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*10} {'-'*12}  {'-'*30}")

    for (np, ff), info in sorted(totals.items()):
        nota = "x2 partículas" if ff == 0.35 else ""
        print(f"  ({np},{ff})     {info['count']:>5} {info['sim_h']:>8.1f} {info['core_h']:>10.0f} {info['core_h_norm']:>12.0f}  {nota}")

    print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL':<12} {len(all_sims):>5} {total_sim_h:>8.1f} {total_core_h:>10.0f} {total_core_h_norm:>12.0f}")
    print(f"\n  Saltadas: {skipped}")
    print(f"  Equiv. 64 cores (raw):      {total_core_h/64:.1f} h = {total_core_h/64/24:.2f} d")
    print(f"  Equiv. 64 cores (norm):     {total_core_h_norm/64:.1f} h = {total_core_h_norm/64/24:.2f} d")
    print(f"  Equiv. 128 cores (raw):     {total_core_h/128:.1f} h = {total_core_h/128/24:.2f} d")
    print(f"  Equiv. 128 cores (norm):    {total_core_h_norm/128:.1f} h = {total_core_h_norm/128/24:.2f} d")

    return all_sims, total_core_h, total_core_h_norm, len(all_sims), total_sim_h

grand_raw = 0
grand_norm = 0
grand_sims = 0
grand_sim_h = 0

for label, path, exclude in DIRS:
    sims, raw, norm, n, sim_h = process_dir(label, path, exclude)
    grand_raw += raw
    grand_norm += norm
    grand_sims += n
    grand_sim_h += sim_h

print(f"\n{f'='*90}")
print(f"  GRAN TOTAL")
print(f"{'='*90}")
print(f"  Simulaciones: {grand_sims}")
print(f"  Core-horas (raw):    {grand_raw:>10.0f}  =  {grand_raw/24:.0f} core-d")
print(f"  Core-horas (norm):   {grand_norm:>10.0f}  =  {grand_norm/24:.0f} core-d")
print(f"  Equiv. 64 cores raw:    {grand_raw/64/24:.2f} d")
print(f"  Equiv. 64 cores norm:   {grand_norm/64/24:.2f} d")
print(f"  Equiv. 128 cores raw:   {grand_raw/128/24:.2f} d")
print(f"  Equiv. 128 cores norm:  {grand_norm/128/24:.2f} d")
