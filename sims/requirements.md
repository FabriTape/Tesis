# Especificaciones Técnicas y de Calidad - Proyecto LAMMPS Analysis

### 1. Visión General del Sistema
Actualización de una herramienta CLI en Python para el análisis de simulaciones de impactos granulares utilizando Polars y la API de OVITO. El foco está en la precisión física, usabilidad y trazabilidad.

### 2. Especificación del Agente de IA (Misión y Restricciones)
* Rol: Ingeniero de Pruebas y Arquitecto Socrático.
* Restricción: No debe entregar código final. Su función es guiar mediante lógica, pseudocódigo y la creación de archivos de prueba (TDD).

### 3. Diagnóstico y Filtro de Erosión (Física)
* Problema: El filtro actual (Z > altura AND vz > 0) ignora partículas en los bordes del cráter.
* Requerimiento: Implementar un criterio compuesto (OR).
    * Rama A: Eyeción vertical con energía cinética mínima.
    * Rama B: Desplazamiento lateral significativo o energía total alta en la zona de superficie.

### 4. Interfaz de Usuario (UX) y Navegación
* Wrapper de Entrada: Implementar una función `solicitar_entrada_segura` para centralizar la lógica de captación de datos.
* Comandos Globales:
    * 'B' (Back): Retroceder al paso anterior en la configuración.
    * 'Q' (Quit): Salida segura con confirmación del usuario.
* Radio Default: Si el input del radio está vacío, asignar automáticamente 0.76 um (7.6e-7 m).

### 5. Trazabilidad y Multiplicidad
* Inyección de Metadatos: Guardar los umbrales de los filtros como columnas en el archivo .parquet resultante.
* Nomenclatura Dinámica: Los nombres de archivo deben incluir los valores de los filtros para evitar colisiones.
* Verificación Inteligente: Solo preguntar por sobreescritura si los parámetros de simulación Y de filtrado son idénticos a un análisis previo.

### 6. Automatización de Visualización
* Integración: Refactorizar `plot.py` en una función ejecutable al final de `data_analysis`.
* Eficiencia: Pasar el DataFrame directamente en memoria desde Polars a Matplotlib.

### 7. Estándares de Código y Calidad (PEP 8 & Ruff)
* PEP 8: Todo el código debe seguir estrictamente la guía de estilo oficial (nomenclatura snake_case, indentación de 4 espacios).
* Linter y Formateador (Ruff): Se utilizará Ruff para garantizar la limpieza del código. Esto incluye:
    * Eliminación automática de imports no usados.
    * Ordenamiento de imports (isort).
    * Corrección de espaciados y estilos según PEP 8.
* Tipado Estricto: Uso de Type Hints en todas las firmas de funciones.
* Documentación: Docstrings obligatorios explicando la lógica física de cada módulo.
