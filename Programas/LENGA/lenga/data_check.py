"""Chequeo de datos locales requeridos por el pipeline al arrancar.

Verifica que existan los CSV consolidados que la plataforma usa para generar
figuras y tablas, y devuelve un informe legible para mostrarlo en la GUI en
lugar de dejar que las pestañas fallen con errores genéricos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Inyectar directorio raíz de programas para importar cluster.generador_figuras
BASE_PROGRAMAS = Path(__file__).resolve().parent.parent.parent
if str(BASE_PROGRAMAS) not in sys.path:
    sys.path.insert(0, str(BASE_PROGRAMAS))

import cluster.generador_figuras as gf


def check_local_datasets() -> Dict[str, dict]:
    """Verifica los datos locales de cada dataset de generador_figuras.py.

    Returns:
        Dict[str, dict]: Por dataset ('9_mov', '49_mov'), un dict con:
            - 'available': True si el CSV consolidado existe y no está vacío.
            - 'csv': Path del CSV consolidado.
            - 'csv_ok': bool.
            - 'cache_dir': Path de la caché procesada (opcional).
            - 'cache_ok': bool.
            - 'issues': List[str] con descripciones legibles.
    """
    report: Dict[str, dict] = {}
    for name, cfg in gf.DATASETS.items():
        csv_path: Path = cfg["csv"]
        cache_dir: Path = cfg.get("cache_dir")
        csv_ok = csv_path.exists() and csv_path.stat().st_size > 0
        cache_ok = bool(cache_dir) and cache_dir.exists() and any(cache_dir.iterdir())

        issues = []
        if not csv_ok:
            issues.append(f"Falta el CSV consolidado: {csv_path}")
        if not cache_ok:
            issues.append(f"Falta la caché procesada (se regenerará al generar figuras/tablas): {cache_dir}")

        report[name] = {
            "available": csv_ok,
            "csv": csv_path,
            "csv_ok": csv_ok,
            "cache_dir": cache_dir,
            "cache_ok": cache_ok,
            "issues": issues,
        }
    return report


def build_warning_message(report: Dict[str, dict]) -> str:
    """Construye un mensaje legible con los datasets incompletos.

    Args:
        report (Dict[str, dict]): Salida de :func:`check_local_datasets`.

    Returns:
        str: Texto vacío si todo está disponible, o una descripción de lo que falta.
    """
    missing = {name: r for name, r in report.items() if not r["available"]}
    if not missing:
        return ""

    lines = [
        "Faltan datos locales requeridos para generar figuras y tablas:",
        "",
    ]
    for name, r in missing.items():
        lines.append(f"- Dataset '{name}':")
        for issue in r["issues"]:
            lines.append(f"    · {issue}")
    lines += [
        "",
        "Ejecute el análisis CARBOXI (pestaña 'Análisis CARBOXI') en el cluster y",
        "descargue los resultados con SCP para consolidar los CSV antes de continuar.",
    ]
    return "\n".join(lines)
