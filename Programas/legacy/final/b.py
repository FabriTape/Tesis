import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("~/resultados/merge_resultados.csv").expanduser()

def main():
    df = pd.read_csv(CSV_PATH)
    
    # 1. Grilla Global Vectorizada
    df["x"] = df["posx"].div(1.52).round().astype(int)
    df["y"] = df["posy"].div(1.52).round().astype(int)
    
    reales = df[["x", "y"]].drop_duplicates().assign(existe=True).set_index(["x", "y"])
    
    x_r, y_r = range(df["x"].min(), df["x"].max() + 1), range(df["y"].min(), df["y"].max() + 1)
    base_grid = pd.MultiIndex.from_product([x_r, y_r], names=["x", "y"]).to_frame(index=False).set_index(["x", "y"])
    base_grid = base_grid.join(reales).fillna(False)
    
    # 2. Análisis iterativo
    for ff in [0.15, 0.35]:
        datos_distribucion = [] # Acá guardamos los datos para boxplot e hist
        
        for v in sorted(df.loc[df["ff"] == ff, "vel0"].unique()):
            sim = df[(df["ff"] == ff) & (df["vel0"] == v)]
            if sim.empty: continue
            
            # Filtro de erosión
            i_max = sim["i"].max()
            mask = (sim["i"] == i_max) & (((sim["type"] == 2) & (sim["z"] > 1e-4)) | (sim["type"] == 1))
            yields = sim[mask].groupby(["x", "y"])["id"].nunique().rename("yield")
            
            # Regla 0 / NaN
            res = base_grid.join(yields)
            res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0
            
            # Guardamos los valores reales (sin NaNs) para los estadísticos
            valores_reales = res.loc[res["existe"], "yield"].reset_index(drop=True)
            datos_distribucion.append(pd.DataFrame({"vel0": v, "yield": valores_reales}))
            
            # --- Heatmap de la simulación ---
            matrix = res["yield"].unstack(level="x").sort_index(ascending=False)
            plt.figure(figsize=(8, 7))
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", square=True, cbar_kws={'label': 'Yield'})
            plt.title(f"Erosión (ff={ff}, v={v})")
            plt.savefig(f"heatmap_ff_{ff}_v{v}.png", bbox_inches="tight")
            plt.close()
            
        # --- NUEVO: Gráfico Combinado (Boxplot + Histograma) para este ff ---
        if datos_distribucion:
            df_dist = pd.concat(datos_distribucion)
            
            # Creamos una figura con 2 filas
            fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1.2]})
            
            # 1. Boxplot (Fila Superior)
            sns.boxplot(data=df_dist, x="vel0", y="yield", ax=ax_box, palette="viridis")
            ax_box.set_title(f"Distribución de Erosión - Boxplot (ff={ff})", fontsize=14, fontweight="bold")
            ax_box.set_xlabel("Velocidad de Impacto (vel0)", fontsize=12)
            ax_box.set_ylabel("Partículas Erosionadas (Y)", fontsize=12)
            ax_box.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 2. Histograma (Fila Inferior)
            sns.histplot(
                data=df_dist, x="yield", hue="vel0", discrete=True, 
                multiple="dodge", shrink=0.8, palette="viridis", ax=ax_hist
            )
            ax_hist.set_title(f"Histograma de Frecuencias de Erosión (ff={ff})", fontsize=14, fontweight="bold")
            ax_hist.set_xlabel("Partículas Erosionadas (Y)", fontsize=12)
            ax_hist.set_ylabel("Cantidad de Celdas de la Grilla", fontsize=12)
            ax_hist.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"distribucion_ff_{ff}.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()
