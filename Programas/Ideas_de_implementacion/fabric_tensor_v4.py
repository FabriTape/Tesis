#!/usr/bin/env python3
import sys, csv
import numpy as np
from ovito.pipeline import Pipeline
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, ClusterAnalysisModifier, CreateBondsModifier, PythonScriptModifier
from ovito.data import DataCollection, SimulationCell
from ovito.pipeline import Pipeline, StaticSource

def build_ovito_pipeline(cutoff: float) -> Pipeline:
    """Inicializa el pipeline de OVITO con la celda y analizadores básicos."""
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property("Position", data=np.empty((0, 3)))

    cell = SimulationCell()
    # SimulationCell espera una matriz 3x4: tres vectores columna y la columna de origen
    M = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    cell.matrix = M
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    # añadir analizadores opcionales (según tu intención)
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    return pipeline

def main():
    if len(sys.argv) < 3:
        print("Usage: ovitos fabric_tensor_pipeline.py <input.dump.gz> <cutoff>")
        sys.exit(1)

    input_path = sys.argv[1]
    cutoff = float(sys.argv[2])
    output_csv = "fabric_tensor_output.csv"

    # Import file as pipeline with multiple frames enabled
    src_pipeline = import_file(input_path, multiple_frames=True)

    # Build a pipeline that contains our SimulationCell and modifiers
    user_pipeline = build_ovito_pipeline(cutoff)

    # Replace the empty static source with the imported file's source (preserving modifiers order)
    user_pipeline.source = src_pipeline.source

    # Ensure the simulation cell on the source is set BEFORE modifiers that use it
    M = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    try:
        user_pipeline.source.data.cell.matrix = M
        user_pipeline.source.data.cell.origin = (0.0, 0.0, 0.0)
    except Exception:
        from ovito.data import SimulationCell
        cell = SimulationCell()
        cell.matrix = M
        user_pipeline.source.data.cell = cell


    def compute_fabric(frame, data):
        positions = data.particles.positions.copy()
        bonds = data.bonds
        nbonds = bonds.count
        if nbonds == 0:
            data.attributes["fabric_exists"] = False
            return
        F = np.zeros((3,3), dtype=float); N = 0
        for i in range(nbonds):
            a = int(bonds.a[i]); b = int(bonds.b[i])
            ra = np.array(positions[a]); rb = np.array(positions[b])
            vec = rb - ra
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
    user_pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    user_pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))
    user_pipeline.modifiers.append(CreateBondsModifier(cutoff=cutoff))
    user_pipeline.modifiers.append(PythonScriptModifier(function=compute_fabric))

    user_pipeline.modifiers.append(PythonScriptModifier(function=compute_fabric))

    # Compute frames and export CSV
    n_frames = user_pipeline.source.num_frames
    header = ["frame","time","fabric_exists","Fxx","Fxy","Fxz","Fyy","Fyz","Fzz",
              "eig1","eig2","eig3","ev1x","ev1y","ev1z","ev2x","ev2y","ev2z","ev3x","ev3y","ev3z"]
    rows = []
    for frame in range(n_frames):
        data = user_pipeline.compute(frame)
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

if __name__ == "__main__":
    main()
