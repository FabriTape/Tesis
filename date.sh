#!/bin/bash

# Nombre base del directorio
base_name="simulacion"

# Fecha con hora
fecha=$(date +%Y-%m-%d_%H-%M-%S)

# Crear directorio con la fecha en el nombre
mkdir "${base_name}_${fecha}"

echo "Directorio creado: ${base_name}_${fecha}"
