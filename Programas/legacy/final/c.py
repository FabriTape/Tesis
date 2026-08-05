import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        datos_distribucion = [] 
        vels = sorted(df.loc[df["ff"] == ff, "vel0"].unique())
        
        for v in vels:
            sim = df[(df["ff"] == ff) & (df["vel0"] == v)]
            if sim.empty: continue
            
            i_max = sim["i"].max()
            mask = (sim["i"] == i_max) & (((sim["type"] == 2) & (sim["z"] > 1e-4)) | (sim["type"] == 1))
            yields = sim[mask].groupby(["x", "y"])["id"].nunique().rename("yield")
            
            res = base_grid.join(yields)
            res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0
            
            # --- CORRECCIÓN DEL BUG ACÁ ---
            # Extraemos los valores y reseteamos el índice espacial para evitar choques con Seaborn
            valores_reales = res.loc[res["existe"], "yield"].reset_index(drop=True)
            datos_distribucion.append(pd.DataFrame({"vel0": v, "yield": valores_reales}))
            
            # Generar Heatmap
            matrix = res["yield"].unstack(level="x").sort_index(ascending=False)
            plt.figure(figsize=(8, 7))
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", square=True, cbar_kws={'label': 'Yield'})
            plt.title(f"Erosión (ff={ff}, v={v})")
            plt.savefig(f"heatmap_ff_{ff}_v{v}.png", bbox_inches="tight")
            plt.close()
            
        # --- NUEVO: Gráfico Combinado con GridSpec ---
        if datos_distribucion:
            df_dist = pd.concat(datos_distribucion)
            
            # Creamos la figura y la cuadrícula (2 filas, N columnas según velocidades)
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, len(vels), height_ratios=[1, 1.2], figure=fig)
            
            # 1. Boxplot (Ocupa toda la fila superior)
            ax_box = fig.add_subplot(gs[0, :])
            sns.boxplot(data=df_dist, x="vel0", y="yield", ax=ax_box, palette="viridis")
            ax_box.set_title(f"Distribución de Erosión (ff={ff})", fontsize=16, fontweight="bold")
            ax_box.set_xlabel("Velocidad de Impacto", fontsize=12)
            ax_box.set_ylabel("Partículas Erosionadas (Y)", fontsize=12)
            ax_box.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 2. Histogramas Side-by-Side (Fila inferior)
            paleta = sns.color_palette("viridis", n_colors=len(vels)) # Mismos colores que el boxplot
            
            for i, v in enumerate(vels):
                ax_hist = fig.add_subplot(gs[1, i])
                subset = df_dist[df_dist["vel0"] == v]
                
                # Histograma individual
                sns.histplot(data=subset, x="yield", discrete=True, color=paleta[i], ax=ax_hist)
                ax_hist.set_title(f"v = {v}", fontsize=14)
                ax_hist.set_xlabel("Partículas (Y)")
                
                # Solo mostrar la etiqueta del eje Y en el primer gráfico para que quede limpio
                if i == 0:
                    ax_hist.set_ylabel("Celdas de la Grilla")
                else:
                    ax_hist.set_ylabel("")
                
                ax_hist.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"distribucion_ff_{ff}.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()
