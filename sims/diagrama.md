# Diagrama de Flujo - Programa de Análisis LAMMPS

## Visión General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITECTURA DEL SISTEMA                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   config.yaml   │    │ input_handler.py│    │state_manager.py │         │
│  │                 │    │                 │    │                 │         │
│  │ - defaults      │    │ - safe_input()  │    │ - StateManager  │         │
│  │ - analisis_onda │    │ - validación    │    │ - Contexto      │         │
│  │ - filtros       │    │ - comandos b/q  │    │ - Estados       │         │
│  │ - material      │    │ - tipos         │    │ - es_critica()  │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                     │                      │                   │
│           └─────────────────────┼──────────────────────┘                   │
│                                 │                                           │
│                    ┌────────────▼────────────┐                             │
│                    │   data_extractor.py     │                             │
│                    │                         │                             │
│                    │  (orquestador principal)│                             │
│                    └────────────┬────────────┘                             │
│                                 │                                           │
│           ┌─────────────────────┼──────────────────────┐                   │
│           │                     │                      │                   │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌──────────▼────────┐          │
│  │ utils.py        │  │ lammps_utils.py │  │ ovito_utils.py    │          │
│  │                 │  │                 │  │                   │          │
│  │ - extraer_      │  │ - lector()      │  │ - analisis_ovito()│          │
│  │   metadatos()   │  │ - max_altura()  │  │ - pipeline OVITO  │          │
│  │ - guardar_      │  │ - control_caja()│  │ - coordinacion    │          │
│  │   resultados()  │  │                 │  │ - clusters        │          │
│  └─────────────────┘  └─────────────────┘  └───────────────────┘          │
│                                                                             │
│                    ┌─────────────────────┐                                  │
│                    │    plot.py          │                                  │
│                    │                     │                                  │
│                    │ - plot_erosion()    │                                  │
│                    │ - metadatos en SVG  │                                  │
│                    └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Máquina de Estados

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MÁQUINA DE ESTADOS PRINCIPAL                         │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │  INICIO                                                              │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 0: SELECCION_CARPETAS                                        │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Mostrar menú de navegación                                  │  │
    │  │ 2. Opciones:                                                   │  │
    │  │    - [-1] Carpeta actual                                       │  │
    │  │    - [-2] Volver atrás                                         │  │
    │  │    - [-3] Todas las carpetas erosion_bed_SI_*                  │  │
    │  │    - [1..N] Navegar a subcarpeta                               │  │
    │  │    - [Q] Salir                                                 │  │
    │  │ 3. Input: buscar_carpeta()                                     │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: contexto.rutas = list[Path]                                 │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 1: CARGAR_CONFIG                                             │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Cargar config.yaml                                          │  │
    │  │ 2. Si no existe → usar defaults                                │  │
    │  │ 3. Mostrar información de configuración                        │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: contexto.config = dict                                      │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 2: CONFIGURACION_RADIO                                       │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Mostrar default desde config (0.76 µm)                      │  │
    │  │ 2. Input: safe_input(tipo=float, default=0.76)                 │  │
    │  │ 3. Validar rango: 0.01 - 1000 µm                               │  │
    │  │ 4. Convertir a metros: R = radio_um * 1e-6                     │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: contexto.R = float (en metros)                              │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 3: CONFIGURACION_ONDA (GLOBAL)                               │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Mostrar default desde config                                │  │
    │  │ 2. Explicar propósito del análisis de onda                     │  │
    │  │ 3. Input: safe_input(tipo=bool, default=config)                │  │
    │  │ 4. NOTA: La criticidad individual se evalúa en el loop         │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: contexto.hacer_onda = bool                                  │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 4: CONFIGURACION_FILTROS                                     │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. Mostrar defaults desde config.yaml                          │  │
    │  │ 2. Input de umbrales:                                          │  │
    │  │    - umbral_vz_min [m/s]                                       │  │
    │  │    - umbral_energia_min [m²/s²]                                │  │
    │  │    - umbral_desplazamiento_lateral [m]                         │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: contexto.umbrales_filtro = dict                             │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LOOP PRINCIPAL: Para CADA ruta en contexto.rutas                    │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ESTADO 5: VERIFICACION_PREVIA (por simulación)                      │
    │                                                                      │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │ 1. datos = ruta.name.replace("erosion_bed_SI_", "", 1)         │  │
    │  │ 2. Buscar archivos previos en resultados/                      │  │
    │  │ 3. Si existe → mostrar estado y preguntar sobreescribir        │  │
    │  │ 4. Comparar umbrales de filtro si existen                      │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Salida: continuar = bool, sobreescribir_todo = bool                 │
    └─────────────────┬────────────────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  ¿continuar?  │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │ NO         │ SI         │
         │            │            │
         ▼            │            ▼
    ┌─────────┐       │    ┌───────────────────────────────────────────────┐
    │ Saltar  │       │    │  EVALUACIÓN DE CRITICIDAD (INDIVIDUAL)        │
    │ simul.  │       │    │                                               │
    └─────────┘       │    │  ┌─────────────────────────────────────────┐  │
                      │    │  │ 1. metadatos = extraer_metadatos(datos) │  │
                      │    │  │ 2. es_critica = es_simulacion_critica() │  │
                      │    │  │                                          │  │
                      │    │  │ Condiciones de criticidad:               │  │
                      │    │  │ • |vel0 - (-50.0)| < 0.1                 │  │
                      │    │  │ • np ∈ [1, 3]                            │  │
                      │    │  │ • ||move - [0,0,0]|| > 1e-6              │  │
                      │    │  └─────────────────────────────────────────┘  │
                      │    └───────────────────────────────────────────────┘
                      │                       │
                      │                       ▼
                      │              ┌─────────────────┐
                      │              │  ¿es_critica?   │
                      │              └────────┬────────┘
                      │                       │
                      │            ┌──────────┼──────────┐
                      │            │ NO       │ SI       │
                      │            │          │          │
                      │            │          ▼          │
                      │            │  ┌─────────────────┐│
                      │            │  │ [!] DETECTADA  ││
                      │            │  │ SIMULACIÓN     ││
                      │            │  │ CRÍTICA        ││
                      │            │  │                ││
                      │            │  │ Si default=False:│
                      │            │  │ preguntar override│
                      │            │  └─────────────────┘│
                      │            │          │          │
                      │            └──────────┼──────────┘
                      │                       │
                      ▼                       ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │  ESTADO 6: PREPARACIÓN DE ARCHIVOS                                    │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ 1. archivos = preparar_archivos(ruta)                           │  │
    │  │ 2. limites_i = extractor_de_borde(ruta, archivos[0])            │  │
    │  │ 3. limites_f = extractor_de_borde(ruta, archivos[-1])           │  │
    │  │ 4. usar_caja = control_caja(limites_i, limites_f, archivos)     │  │
    │  │ 5. altura = max_altura(ruta, R)                                 │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────┬─────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │  ESTADO 7: EJECUCION_FASE1 (detección de onda)                        │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ ¿hacer_onda_simulacion?                                         │  │
    │  │                                                                  │  │
    │  │ SI → ejecutar_fase_onda()                                       │  │
    │  │       • Iterar sobre archivos                                    │  │
    │  │       • detector_de_ondas():                                     │  │
    │  │         - Filtro vertical: z < altura - umbral_techo*R           │  │
    │  │         - Filtro cinético: v² > UMBRAL_V2_MIN                    │  │
    │  │         - Filtro perimetral: cerca de paredes                    │  │
    │  │       • cortar_analisis_onda():                                  │  │
    │  │         - Si cantidad > PARTICULAS_CRITICAS → guardar y cortar   │  │
    │  │         - Si max_v² >= UMBRAL_V2_MAX → guardar y cortar          │  │
    │  │       • Retorna: onda_detectada, paso_corte                      │  │
    │  │                                                                  │  │
    │  │ NO → onda_detectada = False, paso_corte = len(archivos)          │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────┬─────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │  ESTADO 8: EJECUCION_FASE2 (análisis OVITO)                           │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ 1. pipeline = inicializar_pipeline_ovito(cutoff=R)              │  │
    │  │ 2. archivos_fase2 = archivos[:paso_corte]                       │  │
    │  │ 3. Para CADA archivo:                                           │  │
    │  │    a) leer archivo con lector() → LazyFrame                     │  │
    │  │    b) analisis_ovito():                                         │  │
    │  │       - Filtrar: z > altura AND vz > 0                          │  │
    │  │       - Calcular coordinación con OVITO                         │  │
    │  │       - Calcular clusters con OVITO                             │  │
    │  │       - Preservar proyectil (type=2)                            │  │
    │  │    c) Inyectar metadatos:                                       │  │
    │  │       - ff, np, vel0, angulo, posx, posy, posz                  │  │
    │  │       - paso, i, onda                                           │  │
    │  │ 4. Retornar: lista_resultados = list[DataFrame]                 │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────┬─────────────────────────────────────────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │  ESTADO 9: EXPORTACION                                                │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ 1. guardar_resultados(lista_resultados, datos, fecha_hora)      │  │
    │  │    • Si NO hay resultados → crear archivo _NADA_.txt            │  │
    │  │    • Si HAY resultados →                                        │  │
    │  │      - Concatenar DataFrames                                    │  │
    │  │      - Inyectar umbrales como columnas                          │  │
    │  │      - Guardar como {datos}_TOTAL_{fecha_hora}.parquet          │  │
    │  │                                                                  │  │
    │  │ 2. Generar gráfico:                                             │  │
    │  │    • df_concat = pl.concat(lista_resultados)                    │  │
    │  │    • Extraer metadatos del DataFrame                            │  │
    │  │    • plot_erosion():                                            │  │
    │  │      - Agrupar por iteración 'i'                                │  │
    │  │      - Contar partículas eyectadas                              │  │
    │  │      - Plotear con matplotlib                                   │  │
    │  │      - Incluir metadatos en título y leyenda                    │  │
    │  │      - Guardar como SVG en resultados/Imagenes/                 │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────┬─────────────────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ ¿Más rutas?   │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │ SI         │ NO         │
         │            │            │
         └────────────┘            ▼
                      │    ┌───────────────────────────────────────────────┐
                      │    │  ESTADO 10: COMPLETADO                        │
                      │    │                                               │
                      │    │  - Mostrar mensaje de éxito                   │
                      │    │  - Imprimir directorio de resultados          │
                      │    └───────────────────────────────────────────────┘
                      │                       │
                      └───────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │      FIN        │
                          └─────────────────┘
```

---

## Flujo de Extracción de Metadatos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXTRACCIÓN DE METADATOS DESDE NOMBRE DE CARPETA          │
└─────────────────────────────────────────────────────────────────────────────┘

Ejemplo de carpeta:
/home/tutturuu/Documentos/Universidad/Tesis/sims/
    erosion_bed_SI_ff_0.15_np_1_vel_-50_angle_0_move_0.1_0_0/
    │
    └─> ruta.name = "erosion_bed_SI_ff_0.15_np_1_vel_-50_angle_0_move_0.1_0_0"
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  data_extractor.py (línea ~101)                                     │
    │                                                                     │
    │  datos = ruta.name.replace("erosion_bed_SI_", "", 1)                │
    │                                                                     │
    │  Resultado: "ff_0.15_np_1_vel_-50_angle_0_move_0.1_0_0"             │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  utils.py → extraer_metadatos(datos)                                │
    │                                                                     │
    │  Patrones regex:                                                    │
    │  • ff:     r"ff_([^_]+)"         → "0.15"                           │
    │  • np:     r"np_([^_]+)"         → "1"                              │
    │  • vel0:   r"vel_([^_]+)"        → "-50"                            │
    │  • angulo: r"angle_([^_]+)"      → "0"                              │
    │  • move:   r"move_([^_]+)_([^_]+)_([^_]+)" → ("0.1", "0", "0")      │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Diccionario de metadatos retornado:                                │
    │                                                                     │
    │  {                                                                  │
    │      "ff": "0.15",                                                  │
    │      "np": "1",                                                     │
    │      "vel0": "-50",                                                 │
    │      "angulo": "0",                                                 │
    │      "posx": "0.1",                                                 │
    │      "posy": "0",                                                   │
    │      "posz": "0"                                                    │
    │  }                                                                  │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  state_manager.py → es_simulacion_critica(metadatos, config)        │
    │                                                                     │
    │  Evaluación:                                                        │
    │  1. |vel0 - (-50.0)| < 0.1  →  |-50 - (-50)| = 0 < 0.1  ✓           │
    │  2. np ∈ [1, 3]             →  1 ∈ [1, 3]                ✓           │
    │  3. ||move|| > 1e-6         →  √(0.1² + 0² + 0²) = 0.1 > 1e-6  ✓    │
    │                                                                     │
    │  Resultado: TRUE → ¡SIMULACIÓN CRÍTICA!                             │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ovito_utils.py → ejecutar_fase_ovito()                             │
    │                                                                     │
    │  Inyección de metadatos en el DataFrame:                            │
    │                                                                     │
    │  df_procesados_final = df.with_columns(                             │
    │      ff=pl.lit("0.15"),                                             │
    │      np=pl.lit("1"),                                                │
    │      vel0=pl.lit("-50"),                                            │
    │      angulo=pl.lit("0"),                                            │
    │      posx=pl.lit("0.1"),                                            │
    │      posy=pl.lit("0"),                                              │
    │      posz=pl.lit("0"),                                              │
    │  )                                                                  │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  utils.py → guardar_resultados()                                    │
    │                                                                     │
    │  Serialización a Parquet con metadatos como columnas:               │
    │                                                                     │
    │  {datos}_TOTAL_{fecha_hora}.parquet                                 │
    │                                                                     │
    │  Columnas adicionales:                                              │
    │  • umbral_vz_min                                                    │
    │  • umbral_energia_min                                               │
    │  • umbral_desplazamiento_lateral                                    │
    └─────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  plot.py → plot_erosion()                                           │
    │                                                                     │
    │  Lectura de metadatos desde DataFrame para el gráfico:              │
    │                                                                     │
    │  ff = float(df_concat["ff"].item(0))     → 0.15                     │
    │  np_ = float(df_concat["np"].item(0))    → 1                        │
    │  vel = float(df_concat["vel0"].item(0))  → -50                      │
    │  angle = float(df_concat["angulo"].item(0)) → 0                     │
    │  move1 = float(df_concat["posx"].item(0)) → 0.1                     │
    │  move2 = float(df_concat["posy"].item(0)) → 0                       │
    │  move3 = float(df_concat["posz"].item(0)) → 0                       │
    │                                                                     │
    │  Título del gráfico:                                                │
    │  "Evolución de partículas eyectadas                                 │
    │   (v₀=-50 m/s, ff=0.15, N=1, θ=0°)                                  │
    │   Silice | Config v2.0.0"                                           │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Evaluación de Criticidad

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUNCIÓN: es_simulacion_critica()                         │
└─────────────────────────────────────────────────────────────────────────────┘

Entrada:
  • metadatos: dict extraído del nombre de carpeta
  • config: dict cargado desde config.yaml

┌──────────────────────────────────────────────────────────────────────────────┐
│ PASO 1: Verificación de velocidad                                            │
│                                                                              │
│   v_objetivo = config['analisis_onda']['v_objetivo'] = -50.0                 │
│   tolerancia_v = config['analisis_onda']['tolerancia_v'] = 0.1               │
│                                                                              │
│   vel0 = float(metadatos['vel0'])                                            │
│                                                                              │
│   ¿|vel0 - v_objetivo| < tolerancia_v?                                       │
│                                                                              │
│   Ejemplo 1: vel0 = -50  →  |-50 - (-50)| = 0 < 0.1       ✓ CUMPLE          │
│   Ejemplo 2: vel0 = -25  →  |-25 - (-50)| = 25 > 0.1      ✗ NO CUMPLE       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PASO 2: Verificación de cantidad de partículas del proyectil                 │
│                                                                              │
│   np_criticos = config['analisis_onda']['np_criticos'] = [1, 3]              │
│                                                                              │
│   np_val = int(metadatos['np'])                                              │
│                                                                              │
│   ¿np_val ∈ np_criticos?                                                     │
│                                                                              │
│   Ejemplo 1: np = 1  →  1 ∈ [1, 3]                        ✓ CUMPLE          │
│   Ejemplo 2: np = 5  →  5 ∉ [1, 3]                        ✗ NO CUMPLE       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PASO 3: Verificación del vector de movimiento                                │
│                                                                              │
│   move_referencia = [0, 0, 0]                                                │
│   tolerancia_move = 1e-6                                                     │
│                                                                              │
│   move = [float(metadatos['posx']),                                          │
│           float(metadatos['posy']),                                          │
│           float(metadatos['posz'])]                                          │
│                                                                              │
│   norma = √((move[0]-0)² + (move[1]-0)² + (move[2]-0)²)                      │
│                                                                              │
│   ¿norma > tolerancia_move?                                                  │
│                                                                              │
│   Ejemplo 1: move = [0.1, 0, 0]  →  √(0.1²) = 0.1 > 1e-6  ✓ CUMPLE          │
│   Ejemplo 2: move = [0, 0, 0]    →  √(0) = 0 < 1e-6       ✗ NO CUMPLE       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  ¿Las 3 condiciones   │
                          │  se cumplieron?       │
                          └───────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │ NO              │ SI              │
                    │                 │                 │
                    ▼                 ▼                 │
              ┌───────────┐   ┌───────────────────┐     │
              │ RETORNA:  │   │ RETORNA: TRUE     │     │
              │ FALSE     │   │                   │     │
              │           │   │ [!] DETECTADA     │     │
              │           │   │ SIMULACIÓN        │     │
              │           │   │ CRÍTICA           │     │
              └───────────┘   └───────────────────┘     │
                                                        │
                                                        │
                    ┌───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │  COMPORTAMIENTO EN LA TERMINAL                                    │
    │                                                                   │
    │  Si es_critica = TRUE:                                            │
    │  ┌─────────────────────────────────────────────────────────────┐  │
    │  │ [!] DETECTADA SIMULACIÓN CRÍTICA                            │  │
    │  │     v=-50 m/s, np=1, move=(0.1, 0, 0)                       │  │
    │  │                                                             │  │
    │  │     El análisis de onda es ALTAMENTE RECOMENDADO            │  │
    │  │                                                             │  │
    │  │     ¿Desea realizar el análisis de onda para ESTA           │  │
    │  │     simulacion crítica?                                     │  │
    │  │     (Y/N, default=Y): _                                     │  │
    │  └─────────────────────────────────────────────────────────────┘  │
    │                                                                   │
    │  Si es_critica = FALSE:                                           │
    │  → Usar configuración global sin preguntar                        │
    └───────────────────────────────────────────────────────────────────┘
```

---

## Comandos Globales (Input Wrapper)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INPUT WRAPPER - COMANDOS GLOBALES                        │
└─────────────────────────────────────────────────────────────────────────────┘

En CUALQUIER punto del programa donde se solicite input al usuario:

┌──────────────────────────────────────────────────────────────────────────────┐
│ COMANDO 'b' (BACK) - Retroceder                                             │
│                                                                              │
│  Usuario ingresa: "b"                                                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. safe_input() detecta entrada.lower() == 'b'                         │  │
│  │ 2. Retorna: (None, 'back')                                             │  │
│  │ 3. El handler del estado actual recibe la señal                        │  │
│  │ 4. Retorna estado_anterior = estado_actual - 1                         │  │
│  │ 5. La máquina de estados decrementa el contador                        │  │
│  │ 6. Se vuelve a ejecutar el estado anterior                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Ejemplo:                                                                    │
│  - En CONFIGURACION_RADIO → presiona 'b' → vuelve a SELECCION_CARPETAS      │
│  - En CONFIGURACION_FILTROS → presiona 'b' → vuelve a CONFIGURACION_ONDA    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ COMANDO 'q' (QUIT) - Salir                                                  │
│                                                                              │
│  Usuario ingresa: "q"                                                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. safe_input() detecta entrada.lower() == 'q'                         │  │
│  │ 2. Solicita confirmación: "¿Está seguro que desea salir? (Y/N)"        │  │
│  │ 3. Si confirma → sys.exit(0)                                           │  │
│  │ 4. Si no confirma → continuar en el estado actual                      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Ejemplo:                                                                    │
│  - En cualquier estado → presiona 'q' → confirma → sale del programa        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ VALOR POR DEFECTO (ENTER)                                                    │
│                                                                              │
│  Usuario presiona: ENTER (sin escribir nada)                                 │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. safe_input() detecta entrada == ""                                  │  │
│  │ 2. Si default is not None → retorna default                            │  │
│  │ 3. Si default is None → solicita nuevamente                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Ejemplos:                                                                   │
│  - Radio [µm] (default: 0.76): [ENTER] → usa 0.76                           │
│  - ¿Análisis de onda? (Y/N, default=Y): [ENTER] → usa True                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura de Archivos de Salida

```
resultados/
│
├── ff_0.15_np_1_vel_-50_angle_0_move_0.1_0_0_TOTAL_2026-03-12_18-30-00.parquet
│   │
│   │ Columnas de datos:
│   ├─ id, type, x, y, z, vx, vy, vz
│   ├─ coordinacion, cluster
│   ├─ proyectil_erosion
│   │
│   │ Metadatos de simulación (extraídos del nombre de carpeta):
│   ├─ ff = 0.15
│   ├─ np = 1
│   ├─ vel0 = -50
│   ├─ angulo = 0
│   ├─ posx = 0.1
│   ├─ posy = 0
│   └─ posz = 0
│   │
│   │ Metadatos de ejecución:
│   ├─ paso = 100000
│   ├─ i = 0, 1, 2, ...
│   ├─ onda = True/False
│   │
│   └─ Umbrales de filtro (desde config.yaml):
│      ├─ umbral_vz_min = 0.0
│      ├─ umbral_energia_min = 0.0
│      └─ umbral_desplazamiento_lateral = 0.0
│
├── ff_0.15_np_1_vel_-50_angle_0_move_0.1_0_0_ONDA_80_2026-03-12_18-30-00.parquet
│   │
│   │ (Solo si se detectó onda en Fase 1)
│   │ Contiene las partículas de la onda en el timestep de corte
│   │
│   └─ Columnas: x, y, z, vx, vy, vz, v²
│
├── ff_0.15_np_1_vel_-5_angle_0_move_0_0_0_NADA_2026-03-12_18-35-00.txt
│   │
│   │ (Solo si NO hubo erosión significativa)
│   └─ Contenido: "No se obtuvo erosión significativa en esta simulación"
│
└── Imagenes/
    │
    ├── plot_erosion_bed_SI_ff_0.15_np_1.0_vel_-50.0_angle_0.0_move_0.1_0.0_0.0.svg
    │   │
    │   │ Título:
    │   │ "Evolución de partículas eyectadas
    │   │  (v₀=-50 m/s, ff=0.15, N=1, θ=0°)
    │   │  Silice | Config v2.0.0"
    │   │
    │   │ Leyenda con parámetros:
    │   │ • Fricción: ff = 0.15
    │   │ • Partículas: N = 1
    │   │ • Velocidad: v₀ = -50 m/s
    │   │ • Ángulo: θ = 0°
    │   │ • Posición: (0.1, 0, 0)
    │   │
    │   └─ Gráfico: iteración (i) vs cantidad_eyectados
    │
    └── plot_erosion_bed_SI_ff_0.35_np_1.0_vel_-25.0_angle_0.0_move_0.0_0.0_0.0.svg
        └─ (mismo formato para cada simulación)
```

---

## Resumen de Estados

| ID | Estado | Descripción | Datos de Entrada | Datos de Salida |
|----|--------|-------------|------------------|-----------------|
| 0 | `SELECCION_CARPETAS` | Navegación y selección de directorios | - | `contexto.rutas` |
| 1 | `CARGAR_CONFIG` | Carga de config.yaml | - | `contexto.config` |
| 2 | `CONFIGURACION_RADIO` | Ingreso del radio | default desde config | `contexto.R` |
| 3 | `CONFIGURACION_ONDA` | Configuración global de onda | default desde config | `contexto.hacer_onda` |
| 4 | `CONFIGURACION_FILTROS` | Umbrales de erosión | defaults desde config | `contexto.umbrales_filtro` |
| 5 | `VERIFICACION_PREVIA` | Check de análisis previos | `contexto.umbrales_filtro` | `continuar`, `sobreescribir_todo` |
| 6 | `PREPARACION` | Archivos y caja | - | `archivos`, `limites`, `altura` |
| 7 | `EJECUCION_FASE1` | Detección de onda | `hacer_onda_simulacion` | `onda_detectada`, `paso_corte` |
| 8 | `EJECUCION_FASE2` | Análisis OVITO | - | `lista_resultados` |
| 9 | `EXPORTACION` | Guardado + plot | - | archivos .parquet + .svg |
| 10 | `COMPLETADO` | Fin del procesamiento | - | - |

---

## Notas Importantes

1. **Trazabilidad completa**: Todos los parámetros físicos (ff, np, vel, angle, move) se extraen del nombre de CADA carpeta individual y se guardan como columnas en los archivos Parquet.

2. **Criticidad individual**: La evaluación de simulaciones críticas se hace POR CADA simulación, no globalmente. Esto permite que el programa recomiende análisis de onda solo para las simulaciones que realmente lo necesitan.

3. **Navegación hacia atrás**: El comando 'b' permite retroceder al estado anterior sin perder los datos ya ingresados, que se mantienen en el `contexto`.

4. **Metadatos en gráficos**: Los archivos SVG generados incluyen toda la información de la simulación en el título y en una leyenda, facilitando la identificación posterior.

5. **Configuración externalizada**: Todos los valores por defecto, umbrales y parámetros físicos están en `config.yaml`, permitiendo modificar el comportamiento del programa sin tocar el código.
