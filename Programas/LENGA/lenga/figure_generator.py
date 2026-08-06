"""Generador Modular e Integrador de Figuras TAL CUAL desde cluster/generador_figuras.py.

Este módulo se conecta directamente con las rutinas originales de graficado de la tesis
en `cluster/generador_figuras.py`, garantizando fidelidad del 100% pixel por pixel
en el renderizado de gráficos, leyes de potencia, violines, supervivencias y perfiles de profundidad.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Inyectar directorio raíz de programas para importar cluster.generador_figuras
BASE_PROGRAMAS = Path(__file__).resolve().parent.parent.parent
if str(BASE_PROGRAMAS) not in sys.path:
    sys.path.insert(0, str(BASE_PROGRAMAS))

# Garantizar la disponibilidad del shim de Polars en caso de que polars no esté instalado
from lenga.polars_shim import ensure_polars
ensure_polars()

import cluster.generador_figuras as gf

warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")


class FigureGeneratorEngine:
    """Motor de gráficos que ejecuta las rutinas TAL CUAL de generador_figuras.py.

    Attributes:
        AVAILABLE_FIGURES (Dict[str, str]): Catálogo de tipos de figura con descripción corta.
        output_dir (Path): Directorio de salida local para guardar las figuras exportadas.
        processed_data_cache (Dict[str, Tuple]): Caché en memoria para evitar re-filtrar CSVs.
    """

    AVAILABLE_FIGURES = {
        "erosion_loglog": "Erosión: Log-Log (Yield vs Velocidad con Ley de Potencia)",
        "erosion_dist": "Erosión: Distribución Lineal por Caso Impacto",
        "erosion_violin": "Erosión: Distribución Violín / KDE por ff",
        "cluster_size": "Clusters: Distribución de Tamaños de Eyección",
        "cluster_survival": "Clusters: Curva de Supervivencia (CCDF)",
        "model_comparison": "Clusters: Comparación de Modelos Teóricos",
        "model_survival_comp": "Clusters: Comparación Supervivencia de Modelos",
        "profundidad_dmax_proj": "Profundidad: Proyectil d_max vs Move",
        "profundidad_dmean_proj": "Profundidad: Proyectil d_mean vs Move",
        "profundidad_Dmax_bed": "Profundidad: Lecho D_max vs Move",
        "profundidad_Dmean_bed": "Profundidad: Lecho D_mean vs Move",
    }

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Inicializa el motor generador de figuras.

        Args:
            output_dir (Optional[Union[str, Path]]): Ruta del directorio de salida para guardar PNG/PDFs.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "figuras"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_cache: Dict[str, Tuple] = {}

    def get_dataset_config(self, dataset_name: str = "9_mov") -> dict:
        """Obtiene la configuración exacta de un conjunto de datos desde gf.DATASETS.

        Args:
            dataset_name (str): Nombre del dataset ('9_mov', '49_mov').

        Returns:
            dict: Diccionario de configuración de generador_figuras.py.
        """
        if dataset_name in gf.DATASETS:
            cfg = gf.DATASETS[dataset_name].copy()
        else:
            cfg = gf.DATASETS["9_mov"].copy()

        # Configurar directorio de figuras temporal para la salida de generador_figuras
        tmp_fig_dir = self.output_dir / dataset_name
        tmp_fig_dir.mkdir(parents=True, exist_ok=True)
        cfg["fig_dir"] = tmp_fig_dir
        return cfg

    def process_dataset(
        self, dataset_name: str = "9_mov", forzar: bool = False
    ) -> Tuple[dict, object, object, object]:
        """Ejecuta procesar_datos(cfg) de generador_figuras.py para estructurar los DataFrames.

        Args:
            dataset_name (str): Identificador del set de datos ('9_mov', '49_mov').
            forzar (bool): Si es True, ignora la caché y procesa los CSVs desde cero.

        Returns:
            Tuple[dict, object, object, object]: Tupla con (cfg, df_dist, df_cluster, df_proyectil).
        """
        cfg = self.get_dataset_config(dataset_name)
        if not forzar and dataset_name in self.processed_data_cache:
            df_dist, df_cluster, df_proyectil = self.processed_data_cache[dataset_name]
        else:
            df_dist, df_cluster, df_proyectil = gf.procesar_datos(cfg, forzar=forzar)
            self.processed_data_cache[dataset_name] = (df_dist, df_cluster, df_proyectil)

        return cfg, df_dist, df_cluster, df_proyectil

    def generate_figure(self, fig_type: str, dataset_label: str = "9_mov") -> Tuple[plt.Figure, str]:
        """Ejecuta la función de graficado exacta de generador_figuras.py.

        Args:
            fig_type (str): Clave del tipo de figura (ej. 'erosion_loglog', 'cluster_survival').
            dataset_label (str): Nombre del dataset objetivo ('9_mov', '49_mov').

        Returns:
            Tuple[plt.Figure, str]: Tupla con (objeto Figure de Matplotlib TAL CUAL, nombre_figura: str).
        """
        # 1. Obtener datos procesados del dataset
        cfg, df_dist, df_cluster, df_proyectil = self.process_dataset(dataset_label)
        fig_dir = cfg["fig_dir"]
        df_raw = gf.leer_crudo(cfg)

        plt.close("all")

        img_path = None
        fig_name = f"{fig_type}_{dataset_label}"

        try:
            # 2. Invocación de la función exacta en generador_figuras.py
            if fig_type == "erosion_loglog":
                gf.plot_erosion_loglog(cfg, df_dist)
                img_path = fig_dir / "plot_erosion_loglog.png"

            elif fig_type == "erosion_dist":
                gf.plot_erosion_distribution(cfg, df_dist)
                img_path = fig_dir / "plot_erosion_distribution.png"

            elif fig_type == "erosion_violin":
                gf.plot_erosion_violin(cfg, df_dist)
                img_path = fig_dir / "plot_erosion_violin.png"

            elif fig_type == "cluster_size":
                gf.plot_cluster_size_distribution(
                    cfg, df_cluster, df_raw, cfg["relleno"], "plot_cluster_size_distribution.png", 0.95, 1e-4
                )
                img_path = fig_dir / "plot_cluster_size_distribution.png"

            elif fig_type == "cluster_survival":
                gf.plot_cluster_survival(cfg, df_cluster, df_raw)
                img_path = fig_dir / "plot_cluster_survival.png"

            elif fig_type == "model_comparison":
                gf.plot_model_comparison(cfg, df_cluster, df_raw)
                img_path = fig_dir / "plot_model_comparison.png"

            elif fig_type == "model_survival_comp":
                gf.plot_model_survival_comparison(cfg, df_cluster, df_raw)
                img_path = fig_dir / "plot_model_survival_comparison.png"

            elif fig_type == "profundidad_dmax_proj":
                gf.plot_profundidad(
                    cfg, df_proyectil, "proyectil", "max", "plot_profundidad_d_max_proyectil.png", r"$d_{\max}$ proyectil [m]"
                )
                img_path = fig_dir / "plot_profundidad_d_max_proyectil.png"

            elif fig_type == "profundidad_dmean_proj":
                gf.plot_profundidad(
                    cfg, df_proyectil, "proyectil", "mean", "plot_profundidad_d_mean_proyectil.png", r"$d_{mean}$ proyectil [m]"
                )
                img_path = fig_dir / "plot_profundidad_d_mean_proyectil.png"

            elif fig_type == "profundidad_Dmax_bed":
                gf.plot_profundidad(
                    cfg, df_proyectil, "lecho", "max", "plot_profundidad_D_max_lecho.png", r"$D_{\max}$ lecho [m]"
                )
                img_path = fig_dir / "plot_profundidad_D_max_lecho.png"

            elif fig_type == "profundidad_Dmean_bed":
                gf.plot_profundidad(
                    cfg, df_proyectil, "lecho", "mean", "plot_profundidad_D_mean_lecho.png", r"$D_{mean}$ lecho [m]"
                )
                img_path = fig_dir / "plot_profundidad_D_mean_lecho.png"

        except Exception as err:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, f"Error al generar figura '{fig_type}':\n{str(err)}", ha="center", va="center", color="red")
            return fig, fig_name

        # 3. Cargar la imagen renderizada TAL CUAL por generador_figuras.py
        if img_path and img_path.exists():
            img = mpimg.imread(img_path)
            height, width, _ = img.shape
            fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            return fig, fig_name
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"No se encontró la imagen en {img_path}", ha="center", va="center")
            return fig, fig_name

    def save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Guarda una figura generada en el directorio final figuras/ en formato PNG.

        Args:
            fig (plt.Figure): Figura Matplotlib a guardar.
            filename (str): Nombre base del archivo de salida.

        Returns:
            Path: Ruta del archivo generado.
        """
        dest_png = self.output_dir / f"{filename}.png"
        fig.savefig(dest_png, dpi=300, bbox_inches="tight")
        return dest_png
