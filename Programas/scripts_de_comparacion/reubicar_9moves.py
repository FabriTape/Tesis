#!/usr/bin/env python3
"""Reubica los resultados de 9_moves (post-merge) a su carpeta destino.

- Mueve los parquets y SVG generados el 2026-10-21 de ~/resultados a
  ~/test_reconciliacion/9_moves_all/resultados/
- Elimina en el destino las versiones viejas (misma clave de simulación),
  verificando antes que el contenido del parquet sea idéntico.
- NO toca los parquets de junio que siguen en ~/resultados (otro dataset).

Uso:  python reubicar_9moves.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import polars as pl

SRC = Path("/home/fbazzoni/resultados")
DST = Path("/home/fbazzoni/test_reconciliacion/9_moves_all/resultados")
STAMP = "2026-10-21"


def key_parquet(name: str) -> str:
    return name.split("_TOTAL_")[0]


def key_svg(name: str) -> str:
    return name.removeprefix("plot_").split("_F2C_")[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frames_iguales(a: Path, b: Path) -> bool:
    da = pl.read_parquet(a)
    db = pl.read_parquet(b)
    return da.columns == db.columns and da.equals(db)


def mover_parquets(dry: bool) -> tuple[int, int]:
    nuevos = sorted(SRC.glob(f"*_TOTAL_F2C_{STAMP}_*.parquet"))
    nombres_nuevos = {n.name for n in nuevos}
    viejos = sorted(p for p in DST.glob("*.parquet") if p.name not in nombres_nuevos)
    movidos = 0
    borrados = 0
    claves_conservadas: set[str] = set()
    for viejo in viejos:
        k = key_parquet(viejo.name)
        candidatos = [n for n in nuevos if key_parquet(n.name) == k]
        if not candidatos:
            print(f"  [KEEP] {viejo.name}  (no hay equivalente del 21-oct)")
            claves_conservadas.add(k)
            continue
        nuevo = candidatos[0]
        if not frames_iguales(viejo, nuevo):
            print(f"  [DIFF] {viejo.name} != {nuevo.name} -> se conserva")
            claves_conservadas.add(k)
            continue
        print(f"  [DUP ] {viejo.name} (idéntico a la versión 21-oct) -> se borra")
        if not dry:
            viejo.unlink()
        borrados += 1
    for nuevo in nuevos:
        destino = DST / nuevo.name
        if destino.exists():
            destino.unlink()
        print(f"  [MOVE] {nuevo.name}")
        if not dry:
            shutil.move(str(nuevo), str(destino))
        movidos += 1
    return movidos, borrados, claves_conservadas


def mover_svgs(dry: bool, claves_conservadas: set[str]) -> tuple[int, int]:
    src_img = SRC / "Imagenes"
    dst_img = DST / "Imagenes"
    dst_img.mkdir(parents=True, exist_ok=True)
    nuevos = sorted(src_img.glob(f"*_{STAMP}_*.svg"))
    nombres_nuevos = {n.name for n in nuevos}
    viejos = sorted(p for p in dst_img.glob("*.svg") if p.name not in nombres_nuevos)
    movidos = 0
    borrados = 0
    claves_nuevas = {key_svg(p.name) for p in nuevos}
    for viejo in viejos:
        k = key_svg(viejo.name)
        if k in claves_conservadas:
            print(f"  [KEEP] {viejo.name}  (su parquet viejo se conservó)")
        elif k in claves_nuevas:
            print(f"  [DUP ] {viejo.name} -> se borra (reemplazado por la versión 21-oct)")
            if not dry:
                viejo.unlink()
            borrados += 1
        else:
            print(f"  [KEEP] {viejo.name}")
    for nuevo in nuevos:
        destino = dst_img / nuevo.name
        if destino.exists():
            destino.unlink()
        print(f"  [MOVE] {nuevo.name}")
        if not dry:
            shutil.move(str(nuevo), str(destino))
        movidos += 1
    return movidos, borrados


def verificar() -> None:
    parquets = sorted(DST.glob("*.parquet"))
    claves = [key_parquet(p.name) for p in parquets]
    svgs = sorted((DST / "Imagenes").glob("*.svg"))
    print(f"  destino: {len(parquets)} parquets, {len(set(claves))} claves únicas")
    print(f"  destino: {len(svgs)} SVG")
    restantes = sorted(SRC.glob("*.parquet"))
    print(f"  quedan en {SRC}: {len(restantes)} parquets (junio, otro dataset)")
    de_hoy = sum(1 for p in parquets if STAMP in p.name)
    print(f"  en destino, con sello 21-oct: {de_hoy}; restantes (viejos conservados): {len(parquets) - de_hoy}")
    if len(parquets) != len(set(claves)):
        print("  !! hay claves duplicadas en el destino")
        sys.exit(1)
    if len(restantes) != 244:
        print(f"  !! se esperaba 244 parquets de junio en {SRC}, hay {len(restantes)}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "Imagenes").mkdir(parents=True, exist_ok=True)

    print("=== parquets ===")
    m, b, conservadas = mover_parquets(args.dry_run)
    print(f"  movidos: {m}, duplicados viejos eliminados: {b}, claves conservadas: {len(conservadas)}")
    print("=== imágenes SVG ===")
    m2, b2 = mover_svgs(args.dry_run, conservadas)
    print(f"  movidos: {m2}, SVG viejos eliminados: {b2}")
    print("=== verificación ===")
    verificar()
    if args.dry_run:
        print("(dry-run: no se movió ni borró nada)")


if __name__ == "__main__":
    main()
