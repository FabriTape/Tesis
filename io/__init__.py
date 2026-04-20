"""Módulo de I/O para merge de archivos.

Contiene:
- merge: Funciones para fusionar archivos parquet.
"""

from .merge import merge_all_parquet, merge_with_no_erosion

__all__ = ["merge_all_parquet", "merge_with_no_erosion"]