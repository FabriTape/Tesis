"""
LENGA: Plataforma Unificada de Simulación y Análisis de Medios Granulares.
"""

__version__ = "1.1.0"

CHANGELOG = """\
v1.1.0 (Ago 2026)
- Explorador de archivos local y remoto (SSH) en todas las búsquedas de la app.
- Login unificado: un solo campo usuario@servidor en el diálogo de conexión.
- Log de cambios desplegable en el diálogo "Acerca de".
- Sin credenciales hardcodeadas: valores por defecto genéricos (mock / $HOME).

v1.0.0 (base inicial)
- GUI por pestañas: Cluster y Nodos, Análisis CARBOXI, Descarga SCP, Generador de Figuras y Generar Tablas.
- Conexión SSH con contraseña y Modo Simulación / Mock para desarrollo local.
- Envío de simulaciones LAMMPS a Slurm y monitoreo de la cola.
- Análisis remoto CARBOXI (onda de choque, OVITO, merge_all).
- Generación de figuras y de tablas LaTeX para Documentos/.
- Chequeo de datos al arranque y logging rotativo en logs/lenga.log.
"""
