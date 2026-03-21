Actúa como un desarrollador experto en Python 3.11, especializado en análisis de datos de alto rendimiento con Polars y análisis topológico con OVITO. Necesito que refactorices y escribas el código para mi programa llamado "CARBOXI_MINI" a partir del codigo de "CARBOXI", el cual procesa resultados de simulaciones de física granular hechas en LAMMPS.

A continuación te detallo el contexto, los cambios recientes en los datos de entrada y la arquitectura lógica que debes implementar:

1. CAMBIOS REALIZADOS EN LAS SALIDAS DE LAMMPS (INPUT)
Ten en cuenta que los archivos fuente cambiaron drásticamente:
- Separación de Dumps: LAMMPS ya no genera un único archivo masivo. Ahora produce archivos filtrados llamados `output_aire.*.%.txt` (livianos, solo contienen las partículas eyectadas y el proyectil) y `output_cama.*.%.txt` (pesados, el lecho granular).
- Columnas del Dump de Aire: El archivo `output_aire.*.%.txt` ahora incluye explícitamente la fuerza en Z. Las columnas exactas (sin encabezado en la lectura) son: `id type x y z vx vy vz omegax omegay omegaz fz`.
- Freno de Emergencia: La simulación puede abortar antes del paso final si detecta una onda de choque. Tu código NO debe hardcodear el número total de archivos a leer, sino buscar si lammps detectó una onda en el archivo `salida.log` buscando una linea de la forma:
"Fix halt condition for fix-id wave_trigger met on step 28000 with value 1 (../fix_halt.cpp:240)"
- Guardado de datos: La escritura de datos sigue un patrón de la forma `nombre.*.%.txt` donde * es el paso de tiempo y % es el núcleo que escribió ese fragmento. La lectura de datos se adaptará a esta nueva forma de lectura.

2. REQUISITOS DE EJECUCIÓN DEL MÓDULO (CLI)
El programa debe poder ejecutarse desde la terminal como un módulo, recibiendo la ruta de la carpeta de simulación como argumento posicional.
- Comando esperado: `python3.11 -m CARBOXI_MINI ruta/a/la/carpeta`
- Usa la librería `argparse` en el `__main__.py` para capturar el directorio y pasárselo a la función principal.

3. LÓGICA DEL PIPELINE A IMPLEMENTAR

A. Ingesta de Datos (Lazy Polars)
- Construye el patrón de búsqueda apuntando a los archivos de aire.
- Define un schema estricto basado en las columnas de LAMMPS mencionadas arriba.
- Usa `pl.scan_csv()` con el patrón y el schema para lograr una lectura perezosa (LazyFrame). ¡No uses `.collect()` hasta que sea estrictamente necesario!
- Si el programa CARBOXI_MINI detecta que la dirección recibida tiene varias simulaciones las va a analizar a todas.

B. Análisis Topológico (OVITO)
- Crea una función que tome el subconjunto de columnas [x, y, z] y las convierta a un arreglo de NumPy.
- Inicializa un DataCollection de OVITO con estas coordenadas.
- Aplica el modificador `ovito.modifiers.ClusterAnalysis(cutoff=radio_corte, sort_by_size=True)`.
- Extrae y retorna un arreglo o Serie de Polars con la propiedad 'Cluster' (el ID del cluster) generada por OVITO para cada partícula.

C. Fusión y Física Computacional
- Realiza el `.collect()` para materializar los datos.
- Añade el arreglo de clusters obtenido en el paso B como una nueva columna llamada `Cluster_ID`.
- Agrupa el DataFrame por `Cluster_ID`.
- Para cada grupo (cluster), usa agregaciones nativas de Polars para calcular:
  1. Fuerza Z neta (suma de `fz`).
  2. Masa total del cluster (conteo de partículas * masa_particula).
  3. Aceleración Z del Centro de Masa (Fuerza Z neta / Masa total).
- Filtro de Erosión: Filtra el DataFrame agrupado para quedarte EXCLUSIVAMENTE con los `Cluster_IDs` cuya aceleración Z indique un vuelo balístico libre (sin empuje de la cama inferior).

4. CONSIDERACIONES CRÍTICAS DE RENDIMIENTO
- Prioriza siempre las expresiones nativas de Polars (`pl.col()`, agregaciones directas) sobre el uso de `apply()` con Lambdas de Python, para aprovechar la velocidad de Rust.
- El filtrado espacial de altura YA fue realizado por LAMMPS. NO programes filtros de altura de la cama; asume que todo lo que está en `output_aire.*.%.txt` es la zona de interés topológico.
