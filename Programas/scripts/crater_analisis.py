#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier

def main():
    if len(sys.argv) < 3:
        print("Uso: ovitos crater_volume_grid.py <ruta/output.*.gz> <cutoff_contacto>")
        sys.exit(1)

    input_path = sys.argv[1]
    cutoff = float(sys.argv[2])
    
    # Tamaño de la cuadrícula en micrómetros (aprox. 2 veces el radio de tus partículas)
    grid_size_um = 3.0  

    print(f"Cargando simulación: {input_path}")
    pipeline = import_file(input_path)
    from ovito.modifiers import AffineTransformationModifier, PythonScriptModifier

    # 1. Escalar a Micrómetros (x 1,000,000)
    transform = AffineTransformationModifier(
         transformation = [
            [1e6, 0,   0,   0],
            [0,   1e6, 0,   0],
            [0,   0,   1e6, 0]
        ]
    )
    pipeline.modifiers.append(transform)

    # 2. Definir la celda con dimensiones enteras (Micrómetros)
    def set_correct_cell(frame, data):
        matrix = [
            [100.0, 0,     0,     0],
            [0,     100.0, 0,     0],
            [0,     0,     240.0, 0]
        ]
        data.create_cell(matrix, pbc=(True, True, False))

    pipeline.modifiers.append(PythonScriptModifier(function=set_correct_cell))
   

    # Modificador para saber qué partículas están "volando" y excluirlas de la superficie
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))

    num_frames = pipeline.source.num_frames
    bin_area = grid_size_um ** 2
    results = []

    print("Calculando topografía inicial (Frame 0)...")
    data0 = pipeline.compute(0)
    pos0 = data0.particles.positions

    df0 = pd.DataFrame(pos0, columns=['x', 'y', 'z'])
    df0 = df0.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'z'])
    df0['coord'] = data0.particles['Coordination']
    if data0.particles.particle_types is not None:
        df0['type'] = np.array(data0.particles.particle_types)

    # AHORA que tenemos todas las columnas, aplicamos los filtros
    df0 = df0[df0['coord'] >= 3]
    if 'type' in df0.columns:
        df0 = df0[df0['type'] == 1]
    # Filtrar solo el lecho consolidado (ignoramos partículas volando con < 3 vecinos)
    # Asignar cada partícula a una celda de la cuadrícula
    df0['bin_x'] = (df0['x'] // grid_size_um).astype(int)
    df0['bin_y'] = (df0['y'] // grid_size_um).astype(int)

    # Encontrar la altura máxima (la superficie) en cada celda
    H0 = df0.groupby(['bin_x', 'bin_y'])['z'].max().reset_index()
    H0.rename(columns={'z': 'z_initial'}, inplace=True)

    print("Procesando evolución del volumen del cráter...")
    for frame in range(num_frames):
        data = pipeline.compute(frame)
        ts = data.attributes.get("Timestep", frame)

        pos = data.particles.positions
        df = pd.DataFrame(pos, columns=['x', 'y', 'z'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'z'])
        # Volvemos a ignorar las partículas eyectadas
        df['coord'] = data.particles['Coordination']
        if data.particles.particle_types is not None:
            df['type'] = np.array(data.particles.particle_types)

        # AHORA filtramos
        df = df[df['coord'] >= 3]
        if 'type' in df.columns:
            df = df[df['type'] == 1]
	
        df['bin_x'] = (df['x'] // grid_size_um).astype(int)
        df['bin_y'] = (df['y'] // grid_size_um).astype(int)

        # Superficie actual
        Ht = df.groupby(['bin_x', 'bin_y'])['z'].max().reset_index()
        Ht.rename(columns={'z': 'z_current'}, inplace=True)

        # Unir la topografía inicial con la actual
        merged = pd.merge(H0, Ht, on=['bin_x', 'bin_y'], how='left')

        # Si una celda quedó sin partículas sólidas (el cráter llegó al fondo), asumimos altura 0
        merged['z_current'] = merged['z_current'].fillna(0)
        
        # Calcular la diferencia de altura (dz positivo significa que se hizo un pozo)
        merged['dz'] = merged['z_initial'] - merged['z_current']

        # Volumen del cráter (solo las zonas que bajaron respecto al nivel 0)
        crater_vol = (merged[merged['dz'] > 0]['dz'] * bin_area).sum()
        
        # Volumen del material apilado en los bordes (Rim uplift)
        uplift_vol = (abs(merged[merged['dz'] < 0]['dz']) * bin_area).sum()

        results.append({
            'frame': frame,
            'timestep': ts,
            'crater_volume_um3': crater_vol,
            'uplift_volume_um3': uplift_vol
        })

        if frame % 10 == 0 or frame == num_frames - 1:
            print(f"Frame {frame}/{num_frames-1} -> Volumen Excavado: {crater_vol:.1f} um^3")

    # Guardar datos
    out_df = pd.DataFrame(results)
    out_csv = "crater_volume_output.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n¡Listo! CSV guardado: {out_csv}")

    # Generar gráfico
    plt.figure(figsize=(9, 6))
    plt.plot(out_df['timestep'], out_df['crater_volume_um3'], marker='o', color='firebrick', label='Volumen del Cráter (Excavado)')
    plt.plot(out_df['timestep'], out_df['uplift_volume_um3'], marker='^', color='royalblue', label='Volumen Apilado (Borde/Rim)')
    plt.xlabel('Timestep')
    plt.ylabel(r'Volumen ($\mu m^3$)')
    plt.title('Evolución del Volumen del Cráter y Bordes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('crater_volume_plot.png', dpi=300)
    print("Gráfico generado: crater_volume_plot.png")

if __name__ == "__main__":
    main()
