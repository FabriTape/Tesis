#!/usr/bin/env python3
import sys, csv
import numpy as np
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, PythonScriptModifier


if len(sys.argv) < 2:
    print("Usage: ovitos fabric_tensor.py <file> [cutoff]")
    sys.exit(1)
input_path = sys.argv[1]
cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
output_csv = "fabric_tensor_output.csv"

pipeline = import_file(input_path, multiple_frames=True)

# CORRECCIÓN: quitar 'only_between_types' (no existe en algunas versiones)
create_bonds = CreateBondsModifier(cutoff=cutoff)
pipeline.modifiers.append(create_bonds)

def compute_fabric(frame, data):
    
    M = np.array([[1.0,0.0,0.0],
                  [0.0,1.0,0.0],
                  [0.0,0.0,1.0]])


    try:
        data.cell.matrix = M
        data.cell.origin = (0.0,0.0,0.0)
        return
    except Exception:
        pass


    try:
        data.cell_.matrix = M
        data.cell_.origin = (0.0,0.0,0.0)
        return
    except Exception:
        pass

    try:
        data.cell = type('CellStub', (), {'matrix': M, 'origin': (0.0,0.0,0.0)})()
    except Exception:
        pass

    positions = data.particles.positions.copy()
    bonds = data.bonds
    nbonds = bonds.count
    if nbonds == 0:
        data.attributes["fabric_exists"] = False
        return
    F = np.zeros((3,3), dtype=float); N = 0
    for i in range(nbonds):
        a = bonds.a[i]; b = bonds.b[i]
        ra = positions[a]; rb = positions[b]
        vec = np.array(rb) - np.array(ra)
        norm = np.linalg.norm(vec)
        if norm == 0: continue
        n = vec / norm
        F += np.outer(n, n); N += 1
    if N == 0:
        data.attributes["fabric_exists"] = False; return
    F /= N
    eigvals, eigvecs = np.linalg.eigh(F)
    idx = np.argsort(eigvals)[::-1]; eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    data.attributes["fabric_exists"] = True
    data.attributes.update({
        "fabric_Fxx": float(F[0,0]), "fabric_Fxy": float(F[0,1]), "fabric_Fxz": float(F[0,2]),
        "fabric_Fyy": float(F[1,1]), "fabric_Fyz": float(F[1,2]), "fabric_Fzz": float(F[2,2]),
        "fabric_eig1": float(eigvals[0]), "fabric_eig2": float(eigvals[1]), "fabric_eig3": float(eigvals[2]),
        "fabric_ev1x": float(eigvecs[0,0]), "fabric_ev1y": float(eigvecs[1,0]), "fabric_ev1z": float(eigvecs[2,0]),
        "fabric_ev2x": float(eigvecs[0,1]), "fabric_ev2y": float(eigvecs[1,1]), "fabric_ev2z": float(eigvecs[2,1]),
        "fabric_ev3x": float(eigvecs[0,2]), "fabric_ev3y": float(eigvecs[1,2]), "fabric_ev3z": float(eigvecs[2,2]),
    })

pipeline.modifiers.append(PythonScriptModifier(function=compute_fabric))

n_frames = pipeline.source.num_frames
header = ["frame","time","fabric_exists","Fxx","Fxy","Fxz","Fyy","Fyz","Fzz",
          "eig1","eig2","eig3","ev1x","ev1y","ev1z","ev2x","ev2y","ev2z","ev3x","ev3y","ev3z"]
rows = []
for frame in range(n_frames):
    data = pipeline.compute(frame)
    t = data.attributes.get("Timestep", frame)
    if not data.attributes.get("fabric_exists", False):
        rows.append([frame, t, False] + [""]*(len(header)-3))
    else:
        rows.append([
            frame, t, True,
            data.attributes["fabric_Fxx"], data.attributes["fabric_Fxy"], data.attributes["fabric_Fxz"],
            data.attributes["fabric_Fyy"], data.attributes["fabric_Fyz"], data.attributes["fabric_Fzz"],
            data.attributes["fabric_eig1"], data.attributes["fabric_eig2"], data.attributes["fabric_eig3"],
            data.attributes["fabric_ev1x"], data.attributes["fabric_ev1y"], data.attributes["fabric_ev1z"],
            data.attributes["fabric_ev2x"], data.attributes["fabric_ev2y"], data.attributes["fabric_ev2z"],
            data.attributes["fabric_ev3x"], data.attributes["fabric_ev3y"], data.attributes["fabric_ev3z"],
        ])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)

print("Results written to", output_csv)
