"""
Generador de variables shell desde config.toml.

Lee el archivo de configuración TOML centralizado y escribe un archivo
shell con todas las variables necesarias para los scripts de simulación
y análisis.

Uso:
    python read_config.py [config.toml]
    source config_vars.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def read_config(config_path: Path) -> dict:
    """Lee el archivo TOML y retorna el diccionario crudo.

    Args:
        config_path: Ruta al archivo config.toml.

    Returns:
        Diccionario con todas las secciones y parámetros.
    """
    with config_path.open("rb") as f:
        return tomllib.load(f)


def generate_shell_vars(raw: dict, output_path: Path) -> None:
    """Genera archivo shell con variables desde el diccionario TOML.

    Recorre todas las secciones del TOML y convierte cada valor a la
    sintaxis de bash correspondiente (arrays, strings, números).

    Args:
        raw: Diccionario crudo de config.toml.
        output_path: Ruta del archivo shell a generar.
    """
    lines: list[str] = [
        "# Auto-generado por read_config.py — NO editar a mano",
        "# Fuente: config.toml",
        "",
    ]

    for section, values in raw.items():
        if not isinstance(values, dict):
            continue

        lines.append(f"# [{section}]")
        for key, value in values.items():
            var_name = key.upper()
            lines.append(f"{var_name}={_format_value(value)}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def _format_value(value: object) -> str:
    """Convierte un valor TOML a sintaxis de bash.

    Args:
        value: Valor del TOML (int, float, str, list).

    Returns:
        Representación en sintaxis bash.
    """
    if isinstance(value, list):
        elements = " ".join(f'"{v}"' for v in value)
        return f"({elements})"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return f'"{value}"'

    return str(value)


def main() -> None:
    """Punto de entrada CLI."""
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("config.toml")

    if not config_path.exists():
        print(f"Error: {config_path} no encontrado.", file=sys.stderr)
        sys.exit(1)

    raw = read_config(config_path)
    output_path = Path("config_vars.sh")
    generate_shell_vars(raw, output_path)
    print(f"Configuración cargada: {config_path} → {output_path}")


if __name__ == "__main__":
    main()
