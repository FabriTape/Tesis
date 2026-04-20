"""Templates para ejecución de simulaciones."""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

SLURM_TEMPLATE = TEMPLATES_DIR / "slurm.sh.template"
LAMMPS_TEMPLATE = TEMPLATES_DIR / "lammps.input"
CONFIG_TEMPLATE = TEMPLATES_DIR / "config.toml"
