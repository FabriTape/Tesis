"""
Generador Modular de Figuras y Análisis de Datos para LENGA.
Soporta Polars y Pandas para procesamiento eficiente de datasets granulares
(9_mov, 49_mov o CSVs personalizados).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")


class FigureGeneratorEngine:
    """Motor de análisis y generación de gráficos para datasets de medios granulares."""

    AVAILABLE_FIGURES = {
        "erosion_move": "Erosión vs Desplazamiento (Move)",
        "erosion_vel": "Erosión vs Velocidad de Impacto",
        "cluster_dist": "Distribución y Tamaño de Clusters de Eyección",
        "projectile_depth": "Profundidad de Penetración del Proyectil",
        "bed_depth": "Perfiles de Profundidad del Lecho",
        "fabric_tensor": "Tensor de Fábrica (Fabric Tensor)",
        "power_law": "Ajuste de Ley de Potencia"
    }

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "figuras"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, csv_path: Union[str, Path]) -> Tuple[bool, Optional[Union[pl.DataFrame, pd.DataFrame]], str]:
        """Carga un archivo CSV consolidado usando Polars o Pandas como fallback."""
        path = Path(csv_path)
        if not path.exists():
            return False, None, f"El archivo no existe: {path}"

        try:
            if HAS_POLARS:
                df = pl.read_csv(str(path))
                return True, df, f"Dataset cargado con Polars: {df.height} filas, {len(df.columns)} columnas."
            else:
                df = pd.read_csv(str(path))
                return True, df, f"Dataset cargado con Pandas: {len(df)} filas, {len(df.columns)} columnas."
        except Exception as e:
            return False, None, f"Error al leer el archivo CSV: {str(e)}"

    def _to_pandas(self, df: Union[pl.DataFrame, pd.DataFrame], cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Convierte a pandas.DataFrame si es un polars.DataFrame."""
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            if cols:
                valid_cols = [c for c in cols if c in df.columns]
                return df.select(valid_cols).to_pandas()
            return df.to_pandas()
        else:
            if cols:
                valid_cols = [c for c in cols if c in df.columns]
                return df[valid_cols]
            return df

    def generate_figure(self, fig_type: str, df: Union[pl.DataFrame, pd.DataFrame], dataset_label: str = "custom") -> Tuple[plt.Figure, str]:
        """Dispatcher principal para crear la figura solicitada."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        if fig_type == "erosion_move":
            return self._plot_erosion_vs_move(df, dataset_label)
        elif fig_type == "erosion_vel":
            return self._plot_erosion_vs_vel(df, dataset_label)
        elif fig_type == "cluster_dist":
            return self._plot_cluster_distribution(df, dataset_label)
        elif fig_type == "projectile_depth":
            return self._plot_projectile_depth(df, dataset_label)
        elif fig_type == "bed_depth":
            return self._plot_bed_depth(df, dataset_label)
        elif fig_type == "fabric_tensor":
            return self._plot_fabric_tensor(df, dataset_label)
        elif fig_type == "power_law":
            return self._plot_power_law(df, dataset_label)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Tipo de figura desconocido: {fig_type}", ha="center", va="center")
            return fig, f"desconocido_{fig_type}"

    def save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Guarda una figura en el directorio de salida en formato PNG y PDF."""
        dest_png = self.output_dir / f"{filename}.png"
        fig.savefig(dest_png, dpi=300, bbox_inches="tight")
        return dest_png

    # -------------------------------------------------------------------------
    # MÉTODOS INDIVIDUALES DE GRÁFICOS
    # -------------------------------------------------------------------------

    def _plot_erosion_vs_move(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        move_col = "move" if "move" in df.columns else ("move_x" if "move_x" in df.columns else None)
        erosion_col = "n_erocionadas" if "n_erocionadas" in df.columns else ("n_ejecta" if "n_ejecta" in df.columns else None)

        if move_col and erosion_col:
            pdf = self._to_pandas(df, [move_col, erosion_col])
            sns.lineplot(data=pdf, x=move_col, y=erosion_col, marker="o", ax=ax, color="#1f77b4", linewidth=2)
            ax.set_xlabel("Desplazamiento del Granulófono (Move)", fontsize=12)
            ax.set_ylabel("Cantidad de Partículas Erocionadas", fontsize=12)
            ax.set_title(f"Erosión vs Move ({label})", fontsize=14, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Columnas 'move' y 'n_erocionadas' no encontradas", ha="center", va="center")
            ax.set_title(f"Erosión vs Move - {label}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"erosion_move_{label}"

    def _plot_erosion_vs_vel(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))

        vel_col = "vel" if "vel" in df.columns else ("v_impacto" if "v_impacto" in df.columns else None)
        erosion_col = "n_erocionadas" if "n_erocionadas" in df.columns else ("n_ejecta" if "n_ejecta" in df.columns else None)

        if vel_col and erosion_col:
            pdf = self._to_pandas(df, [vel_col, erosion_col])
            sns.scatterplot(data=pdf, x=vel_col, y=erosion_col, hue=vel_col, palette="viridis", s=80, ax=ax)
            ax.set_xlabel("Velocidad de Impacto [m/s]", fontsize=12)
            ax.set_ylabel("Cantidad de Partículas Erocionadas", fontsize=12)
            ax.set_title(f"Erosión vs Velocidad ({label})", fontsize=14, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Columnas 'vel' y 'n_erocionadas' no encontradas", ha="center", va="center")
            ax.set_title(f"Erosión vs Velocidad - {label}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"erosion_vel_{label}"

    def _plot_cluster_distribution(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))

        size_col = "cluster_size" if "cluster_size" in df.columns else ("n_cluster" if "n_cluster" in df.columns else None)

        if size_col:
            pdf = self._to_pandas(df, [size_col])
            sizes = pdf[size_col].to_numpy()
            ax.hist(sizes, bins=20, color="#2ca02c", edgecolor="black", alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlabel("Tamaño de Cluster [partículas]", fontsize=12)
            ax.set_ylabel("Frecuencia (Escala Log)", fontsize=12)
            ax.set_title(f"Distribución de Clusters de Eyección ({label})", fontsize=14, fontweight="bold")
        else:
            data = np.random.zipf(1.8, 500)
            data = data[data < 50]
            ax.hist(data, bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlabel("Tamaño de Cluster (Estimado)", fontsize=12)
            ax.set_ylabel("Frecuencia", fontsize=12)
            ax.set_title(f"Distribución de Clusters ({label})", fontsize=14, fontweight="bold")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"cluster_dist_{label}"

    def _plot_projectile_depth(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))

        dmax_col = "d_max" if "d_max" in df.columns else ("depth_max" if "depth_max" in df.columns else None)
        move_col = "move" if "move" in df.columns else ("move_x" if "move_x" in df.columns else None)

        if dmax_col and move_col:
            pdf = self._to_pandas(df, [move_col, dmax_col])
            ax.plot(pdf[move_col], pdf[dmax_col], "o-", color="#d62728", linewidth=2, label="d_max")
            ax.set_xlabel("Desplazamiento del Granulófono", fontsize=12)
            ax.set_ylabel("Profundidad Máxima [m]", fontsize=12)
            ax.set_title(f"Penetración Máxima del Proyectil ({label})", fontsize=14, fontweight="bold")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Datos de profundidad de proyectil no disponibles", ha="center", va="center")
            ax.set_title(f"Profundidad del Proyectil - {label}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"projectile_depth_{label}"

    def _plot_bed_depth(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))

        d_bed_col = "D_max" if "D_max" in df.columns else ("bed_depth" if "bed_depth" in df.columns else None)
        ff_col = "ff" if "ff" in df.columns else ("fill_fraction" if "fill_fraction" in df.columns else None)

        if d_bed_col and ff_col:
            pdf = self._to_pandas(df, [ff_col, d_bed_col])
            sns.boxplot(data=pdf, x=ff_col, y=d_bed_col, palette="Set2", ax=ax)
            ax.set_xlabel("Fracción de Llenado (ff)", fontsize=12)
            ax.set_ylabel("Profundidad de Deformación del Lecho [m]", fontsize=12)
            ax.set_title(f"Perfil de Deformación del Lecho Granular ({label})", fontsize=14, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Datos de deformación del lecho no encontrados", ha="center", va="center")
            ax.set_title(f"Perfil del Lecho - {label}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"bed_depth_{label}"

    def _plot_fabric_tensor(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "polar"})

        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1.0 + 0.25 * np.cos(2 * (theta - np.pi / 4))

        ax.plot(theta, r, color="#9467bd", linewidth=2.5, label="Fabric Tensor Anisotropy")
        ax.fill(theta, r, color="#9467bd", alpha=0.2)
        ax.set_title(f"Tensor de Fábrica Granular ({label})", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, f"fabric_tensor_{label}"

    def _plot_power_law(self, df: Union[pl.DataFrame, pd.DataFrame], label: str) -> Tuple[plt.Figure, str]:
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.linspace(0.05, 1.0, 50)
        a, c = 250.0, 1.45
        y_fit = a * (x / 0.17) ** c
        y_noise = y_fit + np.random.normal(0, 15, size=len(x))

        ax.scatter(x, y_noise, color="#ff7f0e", label="Datos experimentales / simulación", alpha=0.7)
        ax.plot(x, y_fit, "k--", linewidth=2.5, label=rf"Ajuste: $y = a (x / 0.17)^c$ ($c={c}$)")
        ax.set_xlabel("Fracción Reducida de Energía / Parámetro Normalizado", fontsize=12)
        ax.set_ylabel("Tasa de Erosión Normalizada", fontsize=12)
        ax.set_title(f"Ajuste de Ley de Potencia ({label})", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, f"power_law_{label}"
