#!/usr/bin/env python3
"""Compara dos corridas de CARBOXI (antes vs después) simulación por simulación.

Uso:
    python comparar_versiones.py <dir_antes> <dir_despues> [--out DIR] [--verbose]

Matching: por clave de simulación (parte del nombre antes de _TOTAL_,
ignorando el sello temporal y el filtro).

Clasificación de diferencias:
  - ESPERADO: columnas rot_* (solo en "después"), o claves presentes en una
    sola corrida (gate de metadatos / marcadores NADA). También el sufijo
    de columnas umbral_* si aparece en una sola.
  - INESPERADO: cualquier discrepancia de filas o de valor en las columnas
    físicas comunes (x,y,z,vx,vy,vz,coordinacion,cluster,proyectil_erosion,
    lecho_erosion,paso,i,id,type,ff,np,vel0,angulo,posx,posy,posz).

Salida: un .txt legible + un .csv resumen en --out (por defecto, junto al "después").
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

ROT = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]
NO_COMUNES_ESPERADAS = set(ROT) | {"umbral_vz_min", "umbral_techo_radios"}

_TS = re.compile(r"_TOTAL_[A-Z0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def clave(nombre: str) -> str:
    """Clave de simulación desde el nombre de archivo."""
    stem = nombre[: -len(".parquet")] if nombre.endswith(".parquet") else nombre
    return _TS.sub("", stem)


def indexar(d: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(d.glob("*.parquet")):
        if "_ONDA_" in p.name:
            continue
        out[clave(p.name)] = p
    return out


def comparar_df(antes: pl.DataFrame, despues: pl.DataFrame) -> dict:
    """Compara dos DataFrames y devuelve un resumen de diferencias.

    Si la columna `i` (índice de frame) existe en ambos, primero compara sólo
    sobre los frames comunes; los frames presentes en una sola corrida se
    reportan aparte como `frames_solo_antes` / `frames_solo_despues`.
    """
    res: dict = {"filas_antes": antes.height, "filas_despues": despues.height}
    cols_a = set(antes.columns)
    cols_d = set(despues.columns)
    comunes = [c for c in antes.columns if c in cols_d]
    solo_a = sorted(cols_a - cols_d)
    solo_d = sorted(cols_d - cols_a)
    res["cols_solo_antes"] = solo_a
    res["cols_solo_despues"] = solo_d

    no_esperadas_solo = [
        c for c in (solo_a + solo_d) if c not in NO_COMUNES_ESPERADAS
    ]
    res["cols_no_esperadas"] = no_esperadas_solo
    res["frames_solo_antes"] = []
    res["frames_solo_despues"] = []

    # Alinear por frames comunes si hay columna i
    if "i" in comunes:
        fa = set(antes["i"].unique().to_list())
        fd = set(despues["i"].unique().to_list())
        res["frames_solo_antes"] = sorted(fa - fd)
        res["frames_solo_despues"] = sorted(fd - fa)
        comunes_i = sorted(fa & fd)
        a0 = antes.filter(pl.col("i").is_in(comunes_i))
        d0 = despues.filter(pl.col("i").is_in(comunes_i))
        res["filas_comunes_antes"] = a0.height
        res["filas_comunes_despues"] = d0.height
    else:
        a0, d0 = antes, despues
        res["filas_comunes_antes"] = antes.height
        res["filas_comunes_despues"] = despues.height

    difs: dict[str, dict] = {}
    if a0.height == d0.height:
        orden = [c for c in ("id", "i", "x", "y", "z") if c in comunes]
        a = a0.sort(orden) if orden else a0
        d = d0.sort(orden) if orden else d0
        for c in comunes:
            try:
                igual = a[c].equals(d[c])
            except Exception:
                igual = False
            if not igual:
                info = {"tipo": "valor"}
                try:
                    if a[c].dtype.is_numeric():
                        info["max_delta"] = (a[c] - d[c]).abs().max()
                except Exception:
                    pass
                try:
                    info["n_filas_distintas"] = int((a[c] != d[c]).sum())
                except Exception:
                    pass
                difs[c] = info
    else:
        difs["_filas_comunes_"] = {
            "tipo": "conteo",
            "antes": a0.height,
            "despues": d0.height,
        }
    res["difs"] = difs
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("antes", type=Path)
    ap.add_argument("despues", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    idx_a = indexar(args.antes)
    idx_d = indexar(args.despues)
    claves = sorted(set(idx_a) | set(idx_d))
    out_dir = args.out or args.despues
    out_dir.mkdir(parents=True, exist_ok=True)

    filas_txt: list[str] = []
    filas_csv: list[dict] = []
    n_ok = n_inesp = n_solo = 0

    for k in claves:
        pa, pd = idx_a.get(k), idx_d.get(k)
        if pa is None or pd is None:
            n_solo += 1
            lado = "solo_antes" if pa else "solo_despues"
            filas_txt.append(f"[SOLO-{lado.upper()}] {k}")
            filas_csv.append({"clave": k, "estado": f"solo_{lado.split('_')[1]}"})
            continue
        r = comparar_df(pl.read_parquet(pa), pl.read_parquet(pd))
        inesp = [c for c in r["difs"] if c not in NO_COMUNES_ESPERADAS]
        inesp += r["cols_no_esperadas"]
        tiene_frames_extra = bool(r["frames_solo_antes"] or r["frames_solo_despues"])
        if not inesp:
            if r["difs"] or tiene_frames_extra:
                estado = "ESPERADO" if not r["difs"] else "ESPERADO"
            else:
                estado = "IDENTICO"
            n_ok += 1
            detalle = ",".join(r["difs"].keys())
            if tiene_frames_extra:
                detalle = (detalle + "," if detalle else "") + "frames_extra"
            filas_csv.append({"clave": k, "estado": estado, "detalle": detalle})
            if args.verbose or tiene_frames_extra:
                extra = ""
                if r["frames_solo_despues"]:
                    extra += f" frames_extra={r['frames_solo_despues']}"
                if r["frames_solo_antes"]:
                    extra += f" frames_faltan={r['frames_solo_antes']}"
                filas_txt.append(
                    f"[{estado}] {k} (difs: {list(r['difs'].keys())}){extra}"
                )
        else:
            n_inesp += 1
            filas_txt.append(f"[INESPERADO] {k}")
            for c, info in r["difs"].items():
                filas_txt.append(f"    {c}: {info}")
            if r["cols_no_esperadas"]:
                filas_txt.append(f"    cols_no_esperadas: {r['cols_no_esperadas']}")
            if tiene_frames_extra:
                filas_txt.append(
                    f"    frames_solo_despues={r['frames_solo_despues']} "
                    f"frames_solo_antes={r['frames_solo_antes']}"
                )
            filas_csv.append(
                {"clave": k, "estado": "INESPERADO", "detalle": ",".join(inesp)}
            )

    resumen = [
        "=" * 60,
        f"Comparación antes vs después",
        f"  antes:   {args.antes}  ({len(idx_a)} sims)",
        f"  despues: {args.despues}  ({len(idx_d)} sims)",
        "=" * 60,
        f"  IDÉNTICO/ESPERADO : {n_ok}",
        f"  INESPERADO        : {n_inesp}",
        f"  solo en una corrida: {n_solo}",
        "=" * 60,
        "",
        *filas_txt,
    ]
    txt = out_dir / "comparacion_antes_vs_despues.txt"
    txt.write_text("\n".join(resumen) + "\n")
    csv = out_dir / "comparacion_resumen.csv"
    if filas_csv:
        pl.DataFrame(filas_csv).write_csv(csv)

    print("\n".join(resumen[:9]))
    print(f"\nDetalle: {txt}")
    if filas_csv:
        print(f"Resumen CSV: {csv}")

    if n_inesp > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
