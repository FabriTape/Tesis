"""Configuración del logging global de LENGA+.

Escribe logs en consola (stderr) y en un archivo rotatorio en
``LENGA/logs/lenga.log``, y registra excepciones no capturadas para poder
diagnosticar fallas de la GUI sin depender de la terminal.
"""

from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOGS_DIR / "lenga.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logging raíz una sola vez (consola + archivo rotatorio).

    Args:
        level (int): Nivel de logging por defecto (``logging.INFO``).
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        logging.getLogger("lenga").warning(
            "No se pudo inicializar el archivo de log %s: %s", LOG_FILE, exc
        )

    logging.getLogger("lenga").info("Inicio de LENGA+ — log en %s", LOG_FILE)


def log_uncaught_exception() -> None:
    """Instala un hook que registra excepciones no capturadas en el archivo de log."""

    def _hook(exc_type, exc_value, exc_tb):
        logger = logging.getLogger("lenga")
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Interrupción del usuario (KeyboardInterrupt).")
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Excepción no capturada:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _hook
