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
    
    # Marcamos qué coordenadas existen físicamente
    reales = df[["x", "y"]].drop_duplicates().assign(existe=True).set_index(["x", "y"])
    
    # Armamos el cuadrado envolvente total
    x_r, y_r = range(df["x"].min(), df["x"].max() + 1), range(df["y"].min(), df["y"].max() + 1)
    base_grid = pd.MultiIndex.from_product([x_r, y_r], names=["x", "y"]).to_frame(index=False).set_index(["x", "y"])
    base_grid = base_grid.join(reales).fillna(False) # 'existe' será False en los huecos
    
    # 2. Análisis iterativo
    for ff in [0.15, 0.35]:
        for v in sorted(df.loc[df["ff"] == ff, "vel0"].unique()):
            sim = df[(df["ff"] == ff) & (df["vel0"] == v)]
            if sim.empty: continue
            
            # Filtro de erosión
            i_max = sim["i"].max()
            mask = (sim["i"] == i_max) & (((sim["type"] == 2) & (sim["z"] > 1e-4)) | (sim["type"] == 1))
            yields = sim[mask].groupby(["x", "y"])["id"].nunique().rename("yield")
            
            # 3. Regla 0 / NaN
            res = base_grid.join(yields)
            res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0
            
            # 4. Plot directo pivotando con unstack
            matrix = res["yield"].unstack(level="x").sort_index(ascending=False)
            
            plt.figure(figsize=(8, 7))
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", square=True, cbar_kws={'label': 'Yield'})
            plt.title(f"Erosión (ff={ff}, v={v})")
            plt.savefig(f"heatmap_ff_{ff}_v{v}.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()
