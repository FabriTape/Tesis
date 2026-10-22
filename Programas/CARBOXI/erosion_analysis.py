# -*- coding: utf-8 -*-
from __future__ import annotations

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

from .config_and_discovery import print_progress
from .io import LammpsTrajectoryReader

_SCHEMA = {
    "id": pl.UInt32,
    "type": pl.UInt32,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "vz": pl.Float64,
    "coordinacion": pl.Float64,
    "cluster": pl.UInt32,
    "proyectil_erosion": pl.Boolean,
    "lecho_erosion": pl.Boolean,
}


def _make_data(positions=None):
    data = DataCollection()
    if positions is not None:
        p = data.create_particles(count=positions.shape[0])
        p.create_property("Position", data=positions)
    else:
        data.create_particles().create_property("Position", data=np_empty((0, 3)))
    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)
    return data


def build_ovito_pipeline(cutoff: float) -> Pipeline:
    pipeline = Pipeline(source=StaticSource(data=_make_data()))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))
    return pipeline


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    pipeline.source = StaticSource(
        data=_make_data(df.select(["x", "y", "z"]).to_numpy())
    )
    computed = pipeline.compute()
    return df.with_columns(
        pl.Series(
            "coordinacion", computed.particles["Coordination"][:], dtype=pl.Float64
        ),
        pl.Series("cluster", computed.particles["Cluster"][:], dtype=pl.UInt32),
    )


def _filter_F2C(
    df: pl.DataFrame, bed_height, max_height, pipeline=None, freeze_height=0.0
):
    df = df.filter((pl.col("z") > bed_height) | (pl.col("type") == 2))

    if pipeline is None or df.is_empty():
        return df

    df = _run_ovito(df, pipeline)

    cluster_cm = df.group_by("cluster").agg(
        pl.col("z").mean().alias("z_cm"),
        pl.col("vz").mean().alias("vz_cm"),
    )
    valid = cluster_cm.filter(
        ((pl.col("z_cm") > max_height) & (pl.col("vz_cm") > 0))
        | (pl.col("z_cm") > freeze_height)
    )["cluster"].to_list()

    df = df.filter(pl.col("cluster").is_in(valid) | pl.col("type").eq(2))

    return df.with_columns(
        pl.when(pl.col("type").eq(1))
        .then(pl.lit(None).cast(pl.Boolean))
        .when(pl.col("type").eq(2) & pl.col("cluster").is_in(valid))
        .then(True)
        .otherwise(False)
        .alias("proyectil_erosion")
    )


def run_topology_analysis(
    reader,
    cut_index,
    pipeline,
    bed_height,
    max_height,
    metadata,
    timestep_interval,
    freeze_height,
):
    total = cut_index

    if total == 0:
        return []

    # --- Paso 1: último frame → obtener eroded_ids ---
    print_progress(0, total, label="erosión")
    i_last = total - 1

    raw_lf_last = reader.get_frame_lazy(i_last)
    raw_df_last = raw_lf_last.collect()
    filtered_last = _filter_F2C(
        df=raw_df_last,
        bed_height=bed_height,
        max_height=max_height,
        pipeline=pipeline,
        freeze_height=freeze_height,
    )
    eroded_ids = filtered_last["id"]
    if len(eroded_ids) == 0:
        print_progress(100, 100, label="erosión")
        print()
        return []

    # --- Paso 2: frames 0..total-1 con merge raw(eroded_ids) + filtered ---
    retro_results = []

    for i in range(total):
        print_progress(i + 1, total, label="erosión")

        raw_lf = reader.get_frame_lazy(i)
        raw_df = raw_lf.collect()

        retro_df = raw_df.filter(pl.col("id").is_in(eroded_ids))

        if i == i_last:
            filtered_df = filtered_last
        else:
            filtered_df = _filter_F2C(
                df=raw_df,
                bed_height=bed_height,
                max_height=max_height,
                pipeline=pipeline,
                freeze_height=freeze_height,
            )

        if not filtered_df.is_empty():
            retro_df = retro_df.join(
                filtered_df.select("id", "coordinacion", "cluster", "proyectil_erosion"),
                on="id",
                how="left",
            )
        else:
            retro_df = retro_df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("coordinacion"),
                pl.lit(None).cast(pl.UInt32).alias("cluster"),
                pl.lit(None).cast(pl.Boolean).alias("proyectil_erosion"),
            )

        retro_df = retro_df.with_columns(
            pl.when(pl.col("type").eq(1) & pl.col("id").is_in(filtered_df["id"]))
            .then(True)
            .when(pl.col("type").eq(1))
            .then(False)
            .otherwise(None)
            .alias("lecho_erosion"),
            **{k: pl.lit(v) for k, v in metadata.items()},
            paso=pl.lit(timestep_interval),
            i=pl.lit(i),
        )

        retro_results.append(retro_df)

    print_progress(100, 100, label="erosión")
    print()

    return [df for df in retro_results if not df.is_empty()]
