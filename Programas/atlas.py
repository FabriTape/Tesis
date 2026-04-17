import ast
import json
import os
from pathlib import Path

from pyflowchart import Flowchart

# Configuración
PROJECT_DIR = Path("./CARBOXI")
OUTPUT_FILE = "atlas_analisis.html"


def analizar_archivo(filepath):
    """Escanea el archivo Python para extraer su estructura y lógica."""
    with open(filepath, "r", encoding="utf-8") as f:
        codigo = f.read()

    try:
        arbol = ast.parse(codigo)
    except Exception as e:
        print(f"Error parseando {filepath.name}: {e}")
        return None

    desc = ast.get_docstring(arbol) or "Sin descripción detallada."
    funciones = [
        nodo.name + "()"
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.FunctionDef)
    ]

    variables = []
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign):
            for target in nodo.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    variables.append(target.id)

    dependencias_locales = []
    dependencias_externas = []

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                dependencias_externas.append(alias.name.split(".")[0])
        elif isinstance(nodo, ast.ImportFrom):
            modulo = nodo.module or ""
            if "ANALISIS" in modulo or nodo.level > 0:
                dependencias_locales.append(modulo.split(".")[-1])
            else:
                dependencias_externas.append(modulo.split(".")[0])

    try:
        # Extraemos el flujo. simplify=True ayuda a que no se haga tan largo
        fc = Flowchart.from_code(codigo, simplify=True)
        dsl = fc.flowchart()
    except:
        dsl = ""

    return {
        "name": filepath.name,
        "desc": desc.strip().replace("\n", "<br>"),
        "functions": list(set(funciones)),
        "variables": list(set(variables)),
        "depends": list(set(dependencias_locales)),
        "importsExt": list(set(dependencias_externas)),
        "dsl": dsl,
    }


def generar_atlas():
    print(f"🔍 Escaneando proyecto en {PROJECT_DIR}...")
    datos_proyecto = {}

    for py_file in PROJECT_DIR.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        print(f"  Analizando {py_file.name}...")
        datos = analizar_archivo(py_file)
        if datos:
            key = py_file.name.replace(".py", "")
            datos_proyecto[key] = datos

    json_datos = json.dumps(datos_proyecto, ensure_ascii=False)

    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Atlas - ANALISIS</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/raphael/2.3.0/raphael.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/flowchart/1.17.1/flowchart.min.js"></script>
    <style>
        :root {{ --primary: #2563eb; --sidebar: #1e293b; --bg: #f8fafc; --text: #0f172a; --card: #ffffff; }}
        body {{ margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }}
        .sidebar {{ width: 280px; background: var(--sidebar); color: #f1f5f9; display: flex; flex-direction: column; flex-shrink: 0; }}
        .sidebar-header {{ padding: 24px; font-weight: 700; font-size: 1.25rem; border-bottom: 1px solid #334155; text-align: center; }}
        .nav-list {{ list-style: none; padding: 12px; margin: 0; overflow-y: auto; flex: 1; }}
        .nav-item {{ padding: 12px 16px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; display: flex; align-items: center; transition: 0.2s; }}
        .nav-item:hover, .nav-item.active {{ background: var(--primary); color: white; }}
        .nav-item i {{ margin-right: 12px; font-style: normal; }}

        .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }}
        .header {{ padding: 20px 40px; background: white; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px; }}
        .tabs {{ display: flex; gap: 8px; background: #f1f5f9; padding: 4px; border-radius: 8px; }}
        .tab-btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; background: transparent; color: #64748b; transition: 0.2s; }}
        .tab-btn.active {{ background: white; color: var(--primary); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}

        .content {{ flex: 1; padding: 40px; overflow-y: auto; display: none; }}
        .content.active {{ display: block; }}

        #arch-graph {{ width: 100%; height: 600px; background: white; border-radius: 12px; border: 1px solid #e2e8f0; }}

        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
        .card {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
        .card h3 {{ margin: 0 0 16px; font-size: 1.1rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; color: var(--sidebar); }}
        ul.feature-list {{ list-style: none; padding: 0; margin: 0; }}
        ul.feature-list li {{ margin-bottom: 8px; }}
        code {{ background: #eff6ff; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.95rem; }}

        /* Contenedor del flujo optimizado para diagramas grandes */
        #flow-canvas {{
            background: #fafafa;
            border-radius: 12px;
            padding: 40px;
            border: 1px dashed #cbd5e1;
            min-height: 500px;
            overflow: auto; /* Permite scroll horizontal y vertical */
            text-align: center;
        }}
        #flow-canvas svg {{ max-width: none; }} /* Evita que el SVG se encoja a la fuerza */
        .empty-flow {{ color: #64748b; margin-top: 100px; }}
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="sidebar-header">Atlas: ANALISIS</div>
        <ul class="nav-list" id="nav-list"></ul>
    </aside>

    <main class="main">
        <header class="header">
            <div>
                <h1 id="active-title" style="margin:0; font-size: 1.5rem;">Cargando...</h1>
            </div>
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-general')">Vista General</button>
                <button class="tab-btn" onclick="switchTab('tab-explorer')">Explorador</button>
                <button class="tab-btn" onclick="switchTab('tab-flow')">Flujo Lógico</button>
            </div>
        </header>

        <div id="tab-general" class="content active"><div id="arch-graph"></div></div>

        <div id="tab-explorer" class="content">
            <p id="mod-desc" style="color: #64748b; font-size: 1.1rem; margin-bottom: 24px;"></p>
            <div class="grid" id="explorer-grid"></div>
        </div>

        <div id="tab-flow" class="content"><div id="flow-canvas"></div></div>
    </main>

    <script>
        const projectData = {json_datos};
        let activeMod = Object.keys(projectData)[0] || null;

        const navList = document.getElementById('nav-list');
        Object.keys(projectData).forEach(key => {{
            const li = document.createElement('li');
            li.className = `nav-item ${{key === activeMod ? 'active' : ''}}`;
            li.innerHTML = `<i>📄</i> ${{projectData[key].name}}`;
            li.onclick = () => selectModule(key);
            navList.appendChild(li);
        }});

        function selectModule(key) {{
            activeMod = key;
            document.querySelectorAll('.nav-item').forEach((el, i) => {{
                el.classList.toggle('active', Object.keys(projectData)[i] === key);
            }});
            renderActiveContent();
            if(document.getElementById('tab-general').classList.contains('active')) renderArchitecture();
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
            if(tabId === 'tab-general') renderArchitecture();
        }}

        function renderActiveContent() {{
            if(!activeMod) return;
            const mod = projectData[activeMod];
            document.getElementById('active-title').innerText = mod.name;
            document.getElementById('mod-desc').innerHTML = mod.desc;

            const grid = document.getElementById('explorer-grid');
            grid.innerHTML = `
                <div class="card">
                    <h3>ƒ Funciones Exportadas</h3>
                    <ul class="feature-list">${{mod.functions.map(f => `<li><code>${{f}}</code></li>`).join('')}}</ul>
                </div>
                <div class="card">
                    <h3>⚙️ Variables Globales</h3>
                    <ul class="feature-list">${{mod.variables.map(v => `<li><code>${{v}}</code></li>`).join('')}}</ul>
                </div>
                <div class="card">
                    <h3>📥 Dependencias</h3>
                    <div style="margin-bottom:10px;"><strong>Locales:</strong></div>
                    <ul class="feature-list">${{mod.depends.map(d => `<li><code>${{d}}.py</code></li>`).join('')}}</ul>
                    <div style="margin:15px 0 10px;"><strong>Externas:</strong></div>
                    <ul class="feature-list">${{mod.importsExt.map(d => `<li><code>${{d}}</code></li>`).join('')}}</ul>
                </div>
            `;

            const canvas = document.getElementById('flow-canvas');
            canvas.innerHTML = '';
            if(mod.dsl && mod.dsl.trim() !== '') {{
                try {{
                    const chart = flowchart.parse(mod.dsl);

                    // Configuraciones actualizadas para mejor legibilidad
                    chart.drawSVG('flow-canvas', {{
                        'x': 0,
                        'y': 0,
                        'line-width': 2,
                        'line-length': 50,    // Más separación entre cajas
                        'text-margin': 15,    // Cajas más amplias
                        'font-size': 14,      // Letra más grande y legible
                        'font-family': 'Segoe UI, Helvetica, sans-serif',
                        'font-color': '#1e293b',
                        'line-color': '#94a3b8',
                        'element-color': '#475569',
                        'fill': 'white',
                        'yes-text': 'SÍ',
                        'no-text': 'NO',
                        'arrow-end': 'block',
                        'scale': 1,
                        'symbols': {{
                            'start': {{ 'fill': '#22c55e', 'font-color': 'white', 'element-color': '#16a34a' }},
                            'end': {{ 'fill': '#ef4444', 'font-color': 'white', 'element-color': '#dc2626' }},
                            'condition': {{ 'fill': '#fef08a', 'font-color': '#854d0e', 'element-color': '#eab308' }}
                        }}
                    }});
                }} catch(e) {{
                    canvas.innerHTML = '<div class="empty-flow">Error dibujando diagrama. Revisa la complejidad del código.</div>';
                }}
            }} else {{
                canvas.innerHTML = '<div class="empty-flow">Este módulo es puramente procedural o no tiene lógica diagramable.</div>';
            }}
        }}

        function renderArchitecture() {{
            const container = document.getElementById('arch-graph');
            container.innerHTML = '';
            const width = container.offsetWidth || 800;
            const height = container.offsetHeight || 600;

            const nodes = Object.keys(projectData).map(k => ({{ id: k, name: projectData[k].name }}));
            const links = [];
            Object.keys(projectData).forEach(u => {{
                projectData[u].depends.forEach(v => {{
                    if(projectData[v]) links.push({{ source: u, target: v }});
                }});
            }});

            const svg = d3.select("#arch-graph").append("svg").attr("width", width).attr("height", height);

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2));

            const link = svg.append("g").selectAll("line").data(links).enter().append("line")
                .attr("stroke", "#cbd5e1").attr("stroke-width", 2)
                .attr("marker-end", "url(#arrow)");

            svg.append("defs").append("marker").attr("id", "arrow")
                .attr("viewBox", "0 -5 10 10").attr("refX", 20).attr("refY", 0)
                .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
                .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#cbd5e1");

            const node = svg.append("g").selectAll("circle").data(nodes).enter().append("circle")
                .attr("r", d => d.id === activeMod ? 14 : 10)
                .attr("fill", d => d.id === activeMod ? "#2563eb" : "#64748b")
                .attr("stroke", "white").attr("stroke-width", 2)
                .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

            const text = svg.append("g").selectAll("text").data(nodes).enter().append("text")
                .text(d => d.name).attr("font-size", "13px").attr("dx", 18).attr("dy", 4).attr("fill", "#1e293b");

            simulation.on("tick", () => {{
                link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                node.attr("cx", d => d.x).attr("cy", d => d.y);
                text.attr("x", d => d.x).attr("y", d => d.y);
            }});

            function dragstarted(event, d) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
            function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
            function dragended(event, d) {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
        }}

        window.onload = () => {{
            renderActiveContent();
            renderArchitecture();
        }};
    </script>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"✨ ¡Listo! Abrí el archivo '{OUTPUT_FILE}' en tu navegador.")


if __name__ == "__main__":
    generar_atlas()
