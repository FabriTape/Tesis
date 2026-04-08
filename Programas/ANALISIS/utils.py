"""Módulo de utilities para análisis de datos LAMMPS.

Contiene funciones helper compartidas entre múltiples módulos.
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

CONFIG: dict[str, Any] = {
    "i_value": 80,
    "threshold_z": 1e-4,
    "csv_path": Path("/home/tutturuu/Documentos/merge_resultados.csv"),
    "output_dir": Path("/home/tutturuu/Documentos/figuras"),
    "ff_values": [0.15, 0.35],
}

COLORS: dict[str, str] = {
    "primary": "#3b82f6",
    "power_law": "#e74c3c",
    "truncated": "#2ecc71",
    "exponential": "#f39c12",
    "lognormal": "#9b59b6",
    "secondary": "#3498db",
}


def hurwitz_zeta(s: float, q: float, n: int = 100) -> float:
    """Calcula la función zeta de Hurwitz ζ(s,q).

    Args:
        s: Parámetro de potencia (> 1).
        q: Parámetro de desplazamiento (>= 1).
        n: Número de términos para la suma.

    Returns:
        Valor de ζ(s,q).
    """
    result = 0.0
    for k in range(n):
        result += 1.0 / ((k + q) ** s)
    return result


def hurwitz_zeta_derivative(s: float, q: float, n: int = 100) -> float:
    """Calcula la derivada de la función zeta de Hurwitz ∂ζ/∂s.

    Args:
        s: Parámetro de potencia (> 1).
        q: Parámetro de desplazamiento (>= 1).
        n: Número de términos para la suma.

    Returns:
        Valor de dζ/dq.
    """
    result = 0.0
    for k in range(n):
        result -= np.log(k + q) / ((k + q) ** s)
    return result


def compute_alpha_mle(data: np.ndarray, xmin: int) -> float:
    """Calcula el estimador MLE para el exponente alpha de power law.

    Resuelve la ecuación: ζ'(α, xmin)/ζ(α, xmin) = -(1/n)∑log(xi)

    Args:
        data: Array de datos (tamaños de cluster).
        xmin: Valor mínimo a considerar.

    Returns:
        Estimador MLE de alpha.
    """
    n = len(data)
    log_sum = np.sum(np.log(data))
    target = -log_sum / n

    def objective(alpha: float) -> float:
        if alpha <= 1:
            return 1e10
        zeta = hurwitz_zeta(alpha, xmin)
        zeta_prime = hurwitz_zeta_derivative(alpha, xmin)
        if zeta <= 0 or np.isnan(zeta) or np.isinf(zeta):
            return 1e10
        return (zeta_prime / zeta) - target

    try:
        return brentq(objective, 1.01, 20)
    except Exception:
        return 1 + n / np.sum(np.log(data[data >= xmin] / xmin))


def compute_ccdf_empirical(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula la CCDF empírica (función de supervivencia complementaria).

    Args:
        data: Array de datos.

    Returns:
        Tupla de (tamaños únicos, valores CCDF).
    """
    unique_sizes, counts = np.unique(data, return_counts=True)
    ccdf = np.cumsum(counts[::-1])[::-1] / len(data)
    ccdf = np.clip(ccdf, 1e-15, 1.0)
    return unique_sizes, ccdf


def compute_pdf_discrete(x: np.ndarray, alpha: float) -> np.ndarray:
    """Calcula PDF discreta de power law normalizada.

    Args:
        x: Array de valores.
        alpha: Exponente de power law.

    Returns:
        PDF normalizada.
    """
    pdf = x ** (-alpha)
    return pdf / np.sum(pdf)


def compute_ccdf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    """Calcula CCDF a partir de PDF (suma acumulada desde la derecha).

    Args:
        pdf: Distribución de probabilidad.

    Returns:
        CCDF (función de supervivencia).
    """
    ccdf = np.cumsum(pdf[::-1])[::-1]
    return np.clip(ccdf, 1e-15, 1.0)
