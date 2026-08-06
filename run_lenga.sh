#!/bin/bash
# =============================================================================
# Script Lanzador para LENGA (Plataforma de Simulación y Análisis Granular)
# =============================================================================

APP_DIR="/home/tutturuu/Documentos/Universidad/Tesis/Programas/LENGA"

if [ ! -d "$APP_DIR" ]; then
    echo "Error: No se encontró el directorio de LENGA en $APP_DIR"
    exit 1
fi

cd "$APP_DIR" || exit 1

# Configuración de variables de entorno para Qt y Matplotlib
export MPLCONFIGDIR="/tmp"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Ejecutar aplicación LENGA
python3 -m lenga.main "$@"
