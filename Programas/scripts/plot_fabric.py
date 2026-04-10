#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_simulations():
    # 1. Definir el directorio donde buscar (ajusta si es necesario)
    base_dir = Path("./")
    
    # 2. Buscar TODOS los archivos de salida generados por tu script de OVITO
    csv_files = list(base_dir.rglob("fabric_tensor_output.csv"))
    
    if not csv_files:
        print(f"No se encontraron archivos 'fabric_tensor_output.csv' en {base_dir}")
        return
        
    print(f"Se encontraron {len(csv_files)} simulaciones. Generando gráficos...")

    for csv_path in csv_files:
        # Cargar datos
        df = pd.read_csv(csv_path)
        
        # Verificar que el CSV no esté vacío
        if df.empty:
            print(f"El archivo {csv_path} está vacío. Saltando...")
            continue
            
        # Calcular Deviatoric Fabric (F_dev = eig1 - eig3)
        df['F_dev'] = df['eig1'] - df['eig3']
        
        # Crear figura con 2 subgráficos compartiendo el eje X (tiempo)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # El nombre de la simulación suele ser el nombre de la carpeta padre
        sim_name = csv_path.parent.name
        
        # --- Gráfico 1: Deviatoric Fabric (La diferencia) ---
        ax1.plot(df['timestep'], df['F_dev'], marker='o', markersize=4, linestyle='-', color='black', label='$F_{dev} = \lambda_1 - \lambda_3$')
        ax1.set_ylabel('Deviatoric Fabric ($F_{dev}$)', fontsize=12)
        ax1.set_title(f'Evolución de la Anisotropía\n{sim_name}', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # --- Gráfico 2: Los 3 autovalores (3e) ---
        ax2.plot(df['timestep'], df['eig1'], marker='s', markersize=4, linestyle='-', color='firebrick', label='$\lambda_1$ (Mayor)')
        ax2.plot(df['timestep'], df['eig2'], marker='^', markersize=4, linestyle='-', color='forestgreen', label='$\lambda_2$ (Medio)')
        ax2.plot(df['timestep'], df['eig3'], marker='v', markersize=4, linestyle='-', color='royalblue', label='$\lambda_3$ (Menor)')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Autovalores', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        
        # Guardar gráfico en la misma carpeta que el CSV
        output_img = csv_path.parent / "plot_fabric_tensor.png"
        plt.savefig(output_img, dpi=300)
        plt.close() # Cerrar figura para liberar memoria RAM
        
        print(f"✓ Gráfico guardado: {output_img}")

if __name__ == "__main__":
    plot_simulations()
