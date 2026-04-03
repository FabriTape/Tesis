"""Punto de entrada para ejecutar CARBOXI como módulo.

Uso:
    python3.11 -m CARBOXI [ruta_a_carpeta_de_simulaciones]

Si no se proporciona ruta, se abre el navegador interactivo de carpetas.
"""

from .main_pipeline import main

if __name__ == "__main__":
    main()
