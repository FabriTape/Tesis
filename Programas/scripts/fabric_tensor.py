#!/usr/bin/env python3
# fabric_tensor_ovito.py
# Requiere OVITO 3.x (ovitos). Usage: ovitos fabric_tensor_ovito.py file.lammpstrj.gz

import sys
import math
import gzip
import csv
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import CreateBondsModifier, PythonScriptModifier

if len(sys.argv) < 2:
    print("Usage: ovitos fabric_tensor_ovito.py <lammps_dump.gz> [cutoff]")
    sys.exit(1)

input_path = sys.argv[1]
cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5  # ajustar según tu sistema
output_csv = "fabric_tensor_output.csv"

pipeline = import_file(input_path, multiple_frames=True)

# Crear enlaces por distancia con cutoff (para definir "contactos")
create_bonds = CreateBondsModifier(cutoff=cutoff, only_between_types=None)
pipeline.modifiers.append(create_bonds)

# Modificador Python para calcular tensor de fabric por frame
def compute_fabric(frame, data):
    positions = data.particles.positions.copy()
    n_particles = len(positions)
    bonds = data.bonds
    nbonds = bonds.count
    if nbonds == 0:
        # Añadir resultados vacíos en metadatos
        data.attributes["fabric_exists"] = False
        return

    F = np.zeros((3,3), dtype=float)
    N = 0

    # Si el archivo tiene una propiedad de fuerza en bonds, podríamos usarla como peso.
    # Aquí usamos igualdad de peso por defecto.
    for i in range(nbonds):
        a = bonds.a[i]
        b = bonds.b[i]
        ra = positions[a]
        rb = positions[b]
        vec = np.array(rb) - np.array(ra)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        n = vec / norm
        F += np.outer(n, n)
        N += 1

    if N == 0:
        data.attributes["fabric_exists"] = False
        return

    F /= N  # promedio
    # simetría y diagonalización
    eigvals, eigvecs = np.linalg.eigh(F)  # orden ascendente
    # Guardar en atributos para poder exportar si se desea
    data.attributes["fabric_exists"] = True
    data.attributes["fabric_Fxx"] = float(F[0,0]); data.attributes["fabric_Fxy"] = float(F[0,1]); data.attributes["fabric_Fxz"] = float(F[0,2])
    data.attributes["fabric_Fyy"] = float(F[1,1]); data.attributes["fabric_Fyz"] = float(F[1,2]); data.attributes["fabric_Fzz"] = float(F[2,2])
    # Autovalores y autovectores (guardar en orden descendente: mayor primero)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    data.attributes["fabric_eig1"] = float(eigvals[0]); data.attributes["fabric_eig2"] = float(eigvals[1]); data.attributes["fabric_eig3"] = float(eigvals[2])
    # vectores
    for j in range(3):
        vx, vy, vz = eigvecs[:, j]
        data.attributes[f"fabric_ev{j+1}x"] = float(vx)
        data.attributes[f"fabric_ev{j+1}y"] = float(vy)
        data.attributes[f"fabric_ev{j+1}z"] = float(vz)

pipeline.modifiers.append(PythonScriptModifier(function=compute_fabric))

# Recorrer frames y recopilar resultados
n_frames = pipeline.source.num_frames
rows = []
header = ["frame","time","fabric_exists",
          "Fxx","Fxy","Fxz","Fyy","Fyz","Fzz",
          "eig1","eig2","eig3",
          "ev1x","ev1y","ev1z","ev2x","ev2y","ev2z","ev3x","ev3y","ev3z"]

for frame in range(n_frames):
    data = pipeline.compute(frame)
    t = data.attributes.get("Timestep", frame)
    exists = data.attributes.get("fabric_exists", False)
    if not exists:
        row = [frame, t, False] + [""]*(len(header)-3)
    else:
        row = [
            frame, t, True,
            data.attributes["fabric_Fxx"], data.attributes["fabric_Fxy"], data.attributes["fabric_Fxz"],
            data.attributes["fabric_Fyy"], data.attributes["fabric_Fyz"], data.attributes["fabric_Fzz"],
            data.attributes["fabric_eig1"], data.attributes["fabric_eig2"], data.attributes["fabric_eig3"],
            data.attributes["fabric_ev1x"], data.attributes["fabric_ev1y"], data.attributes["fabric_ev1z"],
            data.attributes["fabric_ev2x"], data.attributes["fabric_ev2y"], data.attributes["fabric_ev2z"],
            data.attributes["fabric_ev3x"], data.attributes["fabric_ev3y"], data.attributes["fabric_ev3z"],
        ]
    rows.append(row)

# Escribir CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Finished. Results written to {output_csv}")

