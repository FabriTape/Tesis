#!/usr/bin/env python3
import sys, csv
import numpy as np
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, PythonScriptModifier

def compute_fabric_tensor(frame, data):
    """Calcula el fabric tensor de segundo orden basado en los enlaces (contactos)."""
    bonds = getattr(data.particles, 'bonds', None)
    # 1. Verificar si existen enlaces
    if bonds is None or bonds.count == 0:
        data.attributes["fabric_exists"] = False
        return

    #Obtener posiciones y topología de enlaces
    # topology es un array (N, 2) con los índices de los átomos conectados
    positions = data.particles.positions
    topology = bonds.topology
    
    # 3. Calcular vectores de enlace (rb - ra) considerando PBC
    # Usamos delta_vector para que el vector cruce la cara de la celda si es necesario
    p1 = positions[topology[:, 0]]
    p2 = positions[topology[:, 1]]

    vectors = data.cell.delta_vector(p1, p2) # Aplica corrección de PBC
    
    # 4. Normalizar vectores a unitarios (n = vec / |vec|)
    norms = np.linalg.norm(vectors, axis=1)
    # Evitar división por cero si hubiera partículas solapadas
    valid = norms > 0
    unit_vectors = vectors[valid] / norms[valid, np.newaxis]
    
    N = unit_vectors.shape[0]
    if N == 0:
        data.attributes["fabric_exists"] = False
        return

    # 5. Calcular Fabric Tensor: F = (1/N) * sum(n_i * n_i^T)
    # Esto se hace eficientemente con np.einsum o producto matricial
    F = np.einsum('ij,ik->jk', unit_vectors, unit_vectors) / N
    
    # 6. Diagonalización (Autovalores y Autovectores)
    eigvals, eigvecs = np.linalg.eigh(F)
    # Ordenar de mayor a menor
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 7. Almacenar resultados en atributos
    data.attributes.update({
        "fabric_exists": True,
        "fabric_Fxx": F[0,0], "fabric_Fxy": F[0,1], "fabric_Fxz": F[0,2],
        "fabric_Fyy": F[1,1], "fabric_Fyz": F[1,2], "fabric_Fzz": F[2,2],
        "fabric_eig1": eigvals[0], "fabric_eig2": eigvals[1], "fabric_eig3": eigvals[2],
        "fabric_ev1": eigvecs[:, 0], "fabric_ev2": eigvecs[:, 1], "fabric_ev3": eigvecs[:, 2]
    })
def fix_degenerate_cell(frame, data):
    """
    Crea una celda de simulación si la actual es inexistente o tiene volumen cero.
    """
    # Verificamos si la celda es 'degenerada' (volumen <= 0)
    if data.cell is None or data.cell.volume == 0:
        pos = data.particles.positions
        if pos.count > 0:
            # Calculamos los límites de las partículas
            low = np.min(pos, axis=0)
            high = np.max(pos, axis=0)
            
            # Añadimos un pequeño margen del 10% para evitar partículas en los bordes
            size = (high - low) * 1.1
            if np.any(size == 0): size += 1e-9 # Evitar dimensiones cero
            
            # Creamos una celda ortogonal (matriz diagonal)
            # El origen se desplaza para centrar la caja
            matrix = np.diag(size)
            origin = low - size * 0.05
            
            # Asignamos la nueva celda al dataset
            data.create_cell(matrix, origin=origin, pbc=(False, False, False))

def main():
    if len(sys.argv) < 3:
        print("Uso: ovitos fabric_tensor_optimized.py <input.dump> <cutoff>")
        sys.exit(1)

    input_path = sys.argv[1]
    cutoff = float(sys.argv[2])
    output_csv = "fabric_tensor_output.csv"

    # Cargar pipeline
    pipeline = import_file(input_path)
    from ovito.modifiers import AffineTransformationModifier

    # 1. Transformar a Micrómetros (x 1,000,000)
    transform = AffineTransformationModifier(
        transformation = [
            [1e6, 0,   0,   0],
            [0,   1e6, 0,   0],
            [0,   0,   1e6, 0]
        ]
    )
    pipeline.modifiers.append(transform)

    # 2. Definir la celda ahora usando valores enteros (Micrómetros)
    def set_correct_cell(frame, data):
        matrix = [
            [100.0, 0,     0,     0],
            [0,     100.0, 0,     0],
            [0,     0,     240.0, 0]
        ]
        data.create_cell(matrix, pbc=(True, True, False))
    
    pipeline.modifiers.append(PythonScriptModifier(function=set_correct_cell))

    # 3. Escalar tu input del cutoff a micrómetros también
    cutoff_um = cutoff * 1e6
    pipeline.modifiers.append(CreateBondsModifier(cutoff=cutoff_um))    
    # Agregar nuestro script de cálculo
    pipeline.modifiers.append(PythonScriptModifier(function=compute_fabric_tensor))

    # Preparar exportación
    header = ["frame", "timestep", "Fxx", "Fxy", "Fxz", "Fyy", "Fyz", "Fzz", 
              "eig1", "eig2", "eig3"]
    rows = []

    print(f"Procesando {pipeline.source.num_frames} frames...")

    for frame in range(pipeline.source.num_frames):
        data = pipeline.compute(frame)
        ts = data.attributes.get("Timestep", frame)
        
        if data.attributes.get("fabric_exists"):
            rows.append([
                frame, ts,
                data.attributes["fabric_Fxx"], data.attributes["fabric_Fxy"], data.attributes["fabric_Fxz"],
                data.attributes["fabric_Fyy"], data.attributes["fabric_Fyz"], data.attributes["fabric_Fzz"],
                data.attributes["fabric_eig1"], data.attributes["fabric_eig2"], data.attributes["fabric_eig3"]
            ])
        else:
            rows.append([frame, ts] + [0.0]*9)

    # Guardar CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"¡Listo! Resultados guardados en: {output_csv}")

if __name__ == "__main__":
    main()
