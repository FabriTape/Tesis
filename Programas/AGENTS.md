# Guía de principios SOLID, KISS y YAGNI
# Índice
- [Skills](#skills)
- [Criterios de aceptación](#criterios-de-aceptación)
- [Testing](#testing)
- [Estandares de Código y Calidad](#estándares-de-código-y-calidad)
- [MCPs](#mcp)


## Skills
El agente primero tendrá que tener en cuenta la SKILL.md que se encuentra en la carpeta ./solid/references.
Las Skills que se utilizarán en esta reestructuración son las que se encuentran en la carpeta ./solid/references
Se describen bajo que situaciones se utilizarán cada Skill.

| Situación | Skill a usar | Referencia |
| Voy a crear una función o método nuevo | SOLID (sobre todo SRP) + nombres claros y métodos cortos | references/solid-principles.md |
| Estoy decidiendo arquitectura, tecnologías | KISS – mantenerlo simple, no agregar antes de tiempo | references/complexity.md, ver también, references/achitecture.md |
| Una clase o archivo hace demasiadas cosas | Single Responsibility - dividir en clases más pequeñas | references/solid-principles.md, ver también, references/object-design.md |
| Quiero evitar código e complicado innecesario | YAGNI + KISS – no agregar cosas "por si acaso" | references/complexity.md |
| Revisando código escrito | Clean code + code smells | references/clean-code.md, ver también, references/code-smells.md |
## Criterios de aceptación
- El código cumple con el objetivo de analisis de datos y generación de figuras.
- Las funciones están acortadas, simplificadas y implementan librerías adecuadas, siguiendo los principios SOLID, KISS y YAGNI.
- Al ejecutar una revisión de lo programado con las skills no se encuentran problemas de calidad o complejidad innecesarios.
- Los docstrings generados siguen el formato Google y contienen toda la información necesaria. El docstring **debe** contener:
   - Descripción corta de qué hace (en términos del dominio: física, simulación, etc.)
   - Parámetros con tipos y unidades cuando corresponda
   - Retorno con tipo y significado físico
   - Excepciones importantes si las hay
   - Ejemplo mínimo si la lógica no es obvia

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
### 1. Uso de engram (memoria persistente)

**Cuándo guardar:**

- Después de terminar la refactorización completa de un archivo entero
- Después de tomar una decisión de arquitectura importante que afecte a varios módulos  
- Despues de terminar el reconocimiento, análisis de un archivo entero y al hacer un resumen de como se relacionan los componentes.
**Qué guardar (formato mínimo y obligatorio):**

Título de la entrada:  
`[Módulo/Decisión] - [fecha corta] - Resumen`

Contenido (máximo 400–500 palabras, conciso):

- Nombre del archivo o tema de la decisión
- Resumen en 4-6 oraciones de los cambios más relevantes
- Lista corta de nombres de clases / funciones clave creadas o renombradas
- Decisiones de diseño importantes (ej: "se eligió clase Config en vez de dict global", "se eliminó back-navigation por SRP")
- Razones breves por las que se rechazó otra alternativa

**Cuándo consultar engram (obligatorio antes de cualquier edición de código):**

- Antes de empezar a trabajar en un archivo que ya existe
- Antes de proponer cualquier cambio estructural que pueda contradecir una decisión anterior
- Antes de crear una función o clase nueva que pueda duplicar algo existente

Instrucción literal que el agente debe seguir:  
“Primero busco en engram si existe entrada para este archivo o tema. Si existe, la leo completa antes de escribir cualquier línea.”

## Flujo de trabajo obligatorio

Cada vez que se edita, crea o revisa código, se deben seguir estos pasos en orden:

1. **Consultar engram** — buscar entradas previas sobre el archivo/tema antes de escribir nada
2. **Leer skill/reference** correspondiente según la situación (ver tabla Skills)
3. **Implementar** siguiendo SOLID/KISS/YAGNI
4. **Ejecutar Ruff** (`ruff check`) para verificar calidad del código
5. **Commitear** con mensaje descriptivo (qué, por qué, archivos afectados)
6. **Guardar en enram** checkpoint si fue refactorización completa o decisión de arquitectura
