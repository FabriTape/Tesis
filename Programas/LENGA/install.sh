#!/bin/bash
# =============================================================================
# Instalador para LENGA+ — genera el lanzador .desktop con rutas locales.
#
# Uso:
#   ./install.sh            # instala el lanzador en ~/.local/share/applications
# =============================================================================
set -e

LENGA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_FILE="$APPS_DIR/lengaplus.desktop"

mkdir -p "$APPS_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=LENGA+ — Medios Granulares
GenericName=Plataforma de Simulación y Análisis Granular
Comment=GUI para simulaciones LAMMPS, análisis CARBOXI, generación de figuras y tablas
Exec="$LENGA_DIR/run_lenga.sh"
Icon=$LENGA_DIR/LengaPlus.ico
Terminal=false
Categories=Science;Education;Physics;
Path=$LENGA_DIR
StartupNotify=true
EOF

chmod +x "$LENGA_DIR/run_lenga.sh"

echo "Lanzador instalado en: $DESKTOP_FILE"
echo "Para registrarlo en el menú del escritorio:"
echo "  update-desktop-database $APPS_DIR"
