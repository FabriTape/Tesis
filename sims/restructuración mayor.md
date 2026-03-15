# Reescritura de programa, simplificación y reestructuración siguiendo principios SOLID, KISS y YAGNI
# Índice
- [Estado actual](#estado-actual)
- [Problemas actuales](#problemas-actuales)
- [Skills](#skills)
- [Objetivos](#objetivos)
- [Orden de desarrollo](#orden-de-desarrollo)
- [Criterios de aceptación](#criterios-de-aceptación)
- [Testing](#testing)
- [Estandares de Código y Calidad](#estándares-de-código-y-calidad)
- [MCPs](#mcp)
- [Tecnologías usadas](#tecnologias-usadas)

## Estado actual
El programa que se encuentra en esta carpeta se compone por los siguientes archivos:
- data_analysis.py
- utils.py
- analisis_onda.py
- data_extractor.py
- input_handler.py
- ovito_utils.py
- lammps_utils.py
- plot.py
- state_manager.py
El funcionamiento actual del programa se encuentra esquematizado en el archivo diagrama.md y fue hecho con agentes a partir del archivo requirements.md que dejará de ser utilizado para esta reestructuración.
## Problemas actuales
Los problemas actuales del programa son:
- Interacción innecesaria con el usuario:
  - El usuario solo debe de ingresar la carpeta donde se encuentran los archivos de datos.
  - El programa deberá leer un archivo de configuración para obtener umbrales y otros parámetros.
  - El programa leera las carpetas y archivos de datos para decidir si procesar una serie de simulaciones o una simulación.
  - En la simulación que esté analizando, detectará si hacer análisis de onda o no.
- El programa no toma los datos de config.yaml sino que usa unos preestablecidos en el código.
- El nombre de los graficos al guardarlos en un .svg no tenía ni fecha ni hora de la simulación de la cual se tomó los datos.
- Debe quitarse la funcionalidad de back en la parte de interaccion del usuario asi que las clases no son troncales en el desarrollo del programa.
- El ingreso del radio se quitará, toda configuración se hará a través de un archivo de configuración .yaml.
- La clasificación de si es simulación critica o no, se saltará la verificación si np (número de partículas) es 1 o 3, pues, no hay otra opción.
## Skills
El agente primero tendrá que tener en cuenta la SKILL.md que se encuentra en la carpeta ./skills/solid/references/.
Las Skills que se utilizarán en esta reestructuración son las que se encuentran en la carpeta ./skills/solid/references/
Se describen bajo que situaciones se utilizarán cada Skill.

| Situación | Skill a usar | Referencia |
| Voy a crear una función o método nuevo | SOLID (sobre todo SRP) + nombres claros y métodos cortos | references/solid-principles.md |
| Estoy decidiendo arquitectura, tecnologías | KISS – mantenerlo simple, no agregar antes de tiempo | references/complexity.md, ver también, references/achitecture.md |
| Una clase o archivo hace demasiadas cosas | Single Responsibility - dividir en clases más pequeñas | references/solid-principles.md, ver también, references/object-design.md |
| Quiero evitar código e complicado innecesario | YAGNI + KISS – no agregar cosas "por si acaso" | references/complexity.md |
| Revisando código escrito | Clean code + code smells | references/clean-code.md, ver también, references/code-smells.md |
## Objetivo
Crear un nuevo programa en una carpeta separada utilizando como referencia la funcionalidad actual del programa. Sin embargo todas sus funciones deberán cambiarse (acortarse, simplificarse, implementar librerias que solucionen funcionalidades que necesite el programa y no programarlas desde 0) para obtener un programa facil de mantener y entender, siguiendo los principios SOLID, KISS y YAGNI (más desarrollo en la siguiente sección).
## Criterios de aceptación
- El código cumple con el objetivo de analisis de datos y generación de figuras.
- Las funciones están acortadas, simplificadas y implementan librerías adecuadas, siguiendo los principios SOLID, KISS y YAGNI.
- Al ejecutar una revisión de lo programado con las skills no se encuentran problemas de calidad o complejidad innecesarios.
## Orden de desarrollo
1. Usar la carpeta ./CARBOXI para el desarrollo del nuevo programa.
2. Hacer reconocimiento de la funcionalidad actual del programa leyendo el código fuente.
3. Guardar las ideas principales del desarrollo en engram.
3. Generar una arquitectura de carpetas y módulos para el nuevo programa.
4. Empezar a escribir el nuevo programa siguiendo la arquitectura generada.
5. Revisar el código escrito y refactorizarlo según sea necesario.
## Testing
El testing será llevado a cabo por el equipo de desarrollo.
## Estándares de Código y Calidad 
- PEP 8: Todo el código debe seguir estrictamente la guía de estilo oficial (nomenclatura snake_case, indentación de 4 espacios).
- Linter y Formateador (Ruff): Se utilizará Ruff para garantizar la limpieza del código. Esto incluye:
    - Eliminación automática de imports no usados.
    - Ordenamiento de imports (isort).
    - Corrección de espaciados y estilos según PEP 8.
- Tipado Estricto: Uso de Type Hints en todas las firmas de funciones.
- Documentación: Docstrings obligatorios explicando la lógica física de cada módulo.

## MCPs
Reglas de uso de engram y context7. El agente **no** debe decidir por su cuenta cuándo usarlas; solo debe hacerlo en los momentos explícitamente indicados abajo.

### 1. Uso de engram (memoria persistente)

**Cuándo guardar (solo en estos momentos):**

- Después de terminar la refactorización completa de **un archivo entero** (ej: data_analysis.py, plot.py, etc.)
- Después de tomar una decisión de arquitectura importante que afecte a varios módulos  
  (ej: nueva estructura de carpetas, elección de patrón principal, convención de nombres definitiva)
- Despues de terminar el reconocimiento, análisis de un archivo entero y al hacer un resumen de como se relacionan los componentes.
**Qué guardar (formato mínimo y obligatorio):**

Título de la entrada:  
`[Módulo/Decisión] - [fecha corta] - Resumen`

Contenido (máximo 300–400 palabras, conciso):

- Nombre del archivo o tema de la decisión
- Resumen en 2–4 oraciones de los cambios más relevantes
- Lista corta de nombres de clases / funciones clave creadas o renombradas
- Decisiones de diseño importantes (ej: "se eligió clase Config en vez de dict global", "se eliminó back-navigation por SRP")
- Opcional: 1–2 razones breves por las que se rechazó otra alternativa

**Cuándo consultar engram (obligatorio):**

- Antes de empezar a trabajar en un archivo que ya existe en la carpeta CARBOXI
- Antes de proponer cualquier cambio estructural que pueda contradecir una decisión anterior

Instrucción literal que el agente debe seguir:  
“Primero busco en engram si existe entrada para este archivo o tema. Si existe, la leo completa antes de escribir cualquier línea.”

### 2. Uso de context7 (generación / mejora de docstrings)

**Cuándo usarlo (solo en estos momentos):**

- Justo antes de dar por terminada la refactorización de un archivo
- Siempre que se cree o modifique una función o clase pública (con visibilidad fuera del módulo)

**Qué hacer exactamente:**

1. Invocar context7 con la función/clase completa ya escrita
2. Pedir docstring en formato Google o NumPy (elegir uno y ser consistente)
3. El docstring **debe** contener:
   - Descripción corta de qué hace (en términos del dominio: física, simulación, etc.)
   - Parámetros con tipos y unidades cuando corresponda
   - Retorno con tipo y significado físico
   - Excepciones importantes si las hay
   - Ejemplo mínimo si la lógica no es obvia

**Instrucción literal:**

“Antes de marcar este archivo como terminado, genero docstrings para todas las funciones y clases públicas usando context7. No paso al siguiente archivo sin docstrings completos.”

## Tecnologías Utilizadas
- Python 3.13.5
- Polars (total, sin pandas o numpy)
- ovito
- matplotlib
- numpy
- pathlib
- datetime
- os
- re
- otras librerías pueden ser necesarias según el caso, preguntar al desarrollador al editar el código, dar explicación de por que usarla, las ventajas y si necesita instalación.
