"""Generador de Tablas (LaTeX) TAL CUAL desde cluster/generador_figuras.py.

Replica exactamente las tablas que imprime ``_analisis_9`` y ``_analisis_49``
del script original de la tesis: resumen, rebote de proyectil, erosión,
métricas, ajustes, exponentes del lecho, comparativa y datos crudos de erosión.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Inyectar directorio raíz de programas para importar cluster.generador_figuras
BASE_PROGRAMAS = Path(__file__).resolve().parent.parent.parent
if str(BASE_PROGRAMAS) not in sys.path:
    sys.path.insert(0, str(BASE_PROGRAMAS))

# Garantizar la disponibilidad de Polars (real o shim)
from lenga.polars_shim import ensure_polars

ensure_polars()

import polars as pl

import cluster.generador_figuras as gf


class TablesGeneratorEngine:
    """Motor de tablas que ejecuta las rutinas TAL CUAL de generador_figuras.py.

    Attributes:
        AVAILABLE_TABLES (Dict[str, str]): Catálogo de tablas con nombre legible.
        TABLES_FOR_DATASET (Dict[str, list[str]]): Tablas aplicables por dataset.
        output_dir (Path): Directorio de salida local para tablas/figuras auxiliares.
        processed_data_cache (Dict[str, Tuple]): Caché en memoria de datasets procesados.
    """

    AVAILABLE_TABLES = {
        "tabla_resumen": "Tabla Resumen",
        "tabla_rebote": "Tabla de Rebote de Proyectil",
        "tabla_erosion": "Tabla de Erosión de Proyectil",
        "tabla_metricas": "Tabla de Métricas (RMSE / chi²)",
        "tabla_ajustes": "Tabla de Ajustes (exponentes)",
        "tabla_dmax_lecho": "Tabla D_MAX (Lecho)",
        "tabla_dmean_lecho": "Tabla D_MEAN (Lecho)",
        "tabla_erosion_cruda": "Datos Crudos de Erosión (por posx, posy)",
        "tabla_comparativa": "Tabla Comparativa (9 vs 245 repeticiones)",
    }

    TABLES_FOR_DATASET = {
        "9_mov": [
            "tabla_resumen",
            "tabla_rebote",
            "tabla_erosion",
            "tabla_metricas",
            "tabla_ajustes",
            "tabla_dmax_lecho",
            "tabla_dmean_lecho",
            "tabla_erosion_cruda",
        ],
        "49_mov": [
            "tabla_resumen",
            "tabla_rebote",
            "tabla_erosion",
            "tabla_metricas",
            "tabla_ajustes",
            "tabla_erosion_cruda",
            "tabla_comparativa",
        ],
    }

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Inicializa el motor generador de tablas.

        Args:
            output_dir (Optional[Union[str, Path]]): Ruta de salida local para
                los archivos .tex y figuras auxiliares.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "tablas"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_cache: Dict[str, Tuple] = {}

    def _dataset_cfg(self, dataset_label: str) -> dict:
        """Clona la configuración del script redirigiendo la salida local.

        Args:
            dataset_label (str): Identificador ('9_mov', '49_mov').

        Returns:
            dict: Configuración lista para ``gf.procesar_datos``.
        """
        name = dataset_label if dataset_label in gf.DATASETS else "9_mov"
        cfg = gf.DATASETS[name].copy()
        cfg["fig_dir"] = self.output_dir / name
        cfg["out_dir"] = self.output_dir / name
        cfg["fig_dir"].mkdir(parents=True, exist_ok=True)
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        return cfg

    def process_dataset(
        self, dataset_name: str = "9_mov", forzar: bool = False
    ) -> Tuple[dict, object, object, object]:
        """Ejecuta procesar_datos(cfg) de generador_figuras.py para estructurar los DataFrames.

        Args:
            dataset_name (str): Identificador del set de datos ('9_mov', '49_mov').
            forzar (bool): Si es True, ignora la caché y procesa los CSVs desde cero.

        Returns:
            Tuple[dict, object, object, object]: (cfg, df_dist, df_cluster, df_proyectil).
        """
        cfg = self._dataset_cfg(dataset_name)
        if not forzar and dataset_name in self.processed_data_cache:
            df_dist, df_cluster, df_proyectil = self.processed_data_cache[dataset_name]
        else:
            df_dist, df_cluster, df_proyectil = gf.procesar_datos(cfg, forzar=forzar)
            self.processed_data_cache[dataset_name] = (df_dist, df_cluster, df_proyectil)

        return cfg, df_dist, df_cluster, df_proyectil

    def _build_ajustes(self, cfg: dict, df_dist, fit_astrom: bool):
        """Replica el bloque de ajustes de ``_analisis_9``/``_analisis_49``.

        Args:
            cfg (dict): Configuración del set.
            df_dist (pl.DataFrame): Distribución de yields por repetición.
            fit_astrom (bool): Si True incluye el modelo de Aström (set 49).

        Returns:
            tuple: ``(combinaciones, ajustes_raw, ajustes_formatted)``.
        """
        combinaciones = [
            (n, a, ff) for n, a in cfg["casos"] for ff in cfg["ff_vals"]
        ]
        ajustes_raw = {}
        ajustes_formatted = {}
        for n, alfa, ff in combinaciones:
            subset = df_dist.filter(
                (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == ff)
            )
            if subset.is_empty():
                continue
            raw, formatted = gf.ajuste_modelos(cfg, subset, n)
            if fit_astrom:
                ajustes_raw[(n, alfa, ff)] = [*raw[0], *raw[1]]
            else:
                ajustes_raw[(n, alfa, ff)] = [*raw[0]]
            ajustes_formatted[(n, alfa, ff)] = [r for sub in formatted for r in sub]
        return combinaciones, ajustes_raw, ajustes_formatted

    def _latex_metricas(self, cfg, df_dist, combinaciones, ajustes_raw, fit_astrom: bool) -> str:
        """Tabla LaTeX de métricas (RMSE / chi² reducido)."""
        cols_modelo = {"power_law": [0, 1]}
        modelos = {"power_law": gf.power_law}
        if fit_astrom:
            cols_modelo["astrom"] = [2, 3, 4, 5, 6, 7]
            modelos["astrom"] = gf.astrom_model
        metricas = gf._metricas_comunes(
            cfg, df_dist, combinaciones, cols_modelo, ajustes_raw, modelos
        )
        df_metricas = metricas.to_pandas()
        df_metricas["rmse"] = df_metricas["rmse"].round(2)
        df_metricas["chi2_red"] = df_metricas["chi2_red"].round(2)
        return df_metricas.set_index(["n", "angulo", "ff", "modelo"]).to_latex(multirow=True)

    def _latex_ajustes(self, ajustes_formatted: dict) -> str:
        """Tabla LaTeX de ajustes (exponentes por combinación)."""
        records = [
            {"n": n, "angulo": alfa, "ff": ff, r"$\alpha$": fmt[1]}
            for (n, alfa, ff), fmt in ajustes_formatted.items()
        ]
        df_ajustes = pl.DataFrame(records)
        df_ajustes_latex = df_ajustes.to_pandas()
        df_ajustes_latex[r"$\alpha$"] = df_ajustes_latex[r"$\alpha$"].apply(
            lambda x: f"${str(x).replace('+/-', r' \pm ')}$" if x is not None else x
        )
        return df_ajustes_latex.set_index(["n", "angulo", "ff"]).to_latex(
            multirow=True, escape=False
        )

    def generate_tables(self, dataset_label: str = "9_mov") -> Dict[str, str]:
        """Genera todas las tablas del dataset con las rutinas de generador_figuras.py.

        Args:
            dataset_label (str): Identificador del set ('9_mov', '49_mov').

        Returns:
            Dict[str, str]: Mapa ``{clave_tabla: código LaTeX}``.
        """
        name = dataset_label if dataset_label in gf.DATASETS else "9_mov"
        cfg, df_dist, df_cluster, df_proyectil = self.process_dataset(name)
        df_raw = gf.leer_crudo(cfg)
        fit_astrom = bool(cfg.get("fit_astrom"))

        combinaciones, ajustes_raw, ajustes_formatted = self._build_ajustes(
            cfg, df_dist, fit_astrom
        )

        result: Dict[str, str] = {}

        result["tabla_resumen"] = gf.tabla_resumen(cfg, df_dist, df_proyectil, df_raw)

        latex_rebote, latex_erosion, df_erosion_cruda = gf._tablas_rebote_erosion(cfg, df_raw)
        result["tabla_rebote"] = latex_rebote
        result["tabla_erosion"] = latex_erosion

        result["tabla_metricas"] = self._latex_metricas(
            cfg, df_dist, combinaciones, ajustes_raw, fit_astrom
        )

        result["tabla_ajustes"] = self._latex_ajustes(ajustes_formatted)

        df_cruda_resumen = (
            df_erosion_cruda.group_by(["np", "angulo", "ff", "vel0"])
            .agg(
                conteo_posiciones=pl.len(),
                media=pl.col("porcentaje_erosion").mean(),
                std=pl.col("porcentaje_erosion").std(),
                min=pl.col("porcentaje_erosion").min(),
                max=pl.col("porcentaje_erosion").max(),
            )
            .sort(["np", "angulo", "ff", "vel0"])
        )
        result["tabla_erosion_cruda"] = df_cruda_resumen.to_pandas().to_latex(index=False)

        if name == "9_mov":
            df_ajustes_prof = gf.tabla_ajustes_profundidad(cfg, df_proyectil)
            result["tabla_dmax_lecho"] = gf.generar_tabla_latex_profundidad(
                df_ajustes_prof,
                var_nombre="D_max (lecho)",
                caption=r"Exponente \(c\) del ajuste de la profundidad máxima de excavación del lecho para cada configuración.",
                label="tab:exponentes_dmax",
            )
            result["tabla_dmean_lecho"] = gf.generar_tabla_latex_profundidad(
                df_ajustes_prof,
                var_nombre="D_mean (lecho)",
                caption=r"Exponente \(c\) del ajuste de la profundidad media de excavación del lecho para cada configuración.",
                label="tab:exponentes_dmean",
            )
        else:
            cfg_9, df_dist_9, df_cluster_9, df_proyectil_9 = self.process_dataset("9_mov")
            df_raw_9 = gf.leer_crudo(cfg_9)
            result["tabla_comparativa"] = gf.tabla_comparativa_latex(
                cfg_9, df_dist_9, df_proyectil_9, df_raw_9,
                cfg, df_dist, df_proyectil, df_raw,
            )

        return result

    def save_table(self, table_key: str, latex_source: str, dataset_label: str) -> Path:
        """Guarda el código LaTeX de una tabla en el directorio local de salida.

        Args:
            table_key (str): Clave de la tabla.
            latex_source (str): Código LaTeX a guardar.
            dataset_label (str): Identificador del dataset (para nombrar el archivo).

        Returns:
            Path: Ruta del archivo generado.
        """
        dest = self.output_dir / dataset_label / f"{table_key}.tex"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(latex_source, encoding="utf-8")
        return dest
