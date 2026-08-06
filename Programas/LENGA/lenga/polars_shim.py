"""Shim de Compatibilidad Polars -> Pandas para LENGA.

Permite ejecutar cluster.generador_figuras de forma nativa TAL CUAL incluso si
el paquete 'polars' no se encuentra instalado en la instalación de Python.
"""

from __future__ import annotations

import sys
import types
import pandas as pd
import numpy as np


def ensure_polars():
    """Garantiza la disponibilidad de polars importando polars real o inyectando el shim."""
    try:
        import polars as pl
        return pl
    except ImportError:
        pass

    if "polars" in sys.modules and hasattr(sys.modules["polars"], "_is_shim"):
        return sys.modules["polars"]

    class Expr:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, df):
            return self.fn(df)

        def abs(self):
            return Expr(lambda df: self(df).abs())

        def alias(self, new_name):
            def _alias(df):
                s = self(df)
                s.name = new_name
                return s
            return Expr(_alias)

        def replace(self, old_val, new_val):
            def _replace(df):
                s = self(df)
                return s.replace(old_val, new_val)
            return Expr(_replace)

        def fill_null(self, value):
            def _fill(df):
                s = self(df)
                return s.fillna(value)
            return Expr(_fill)

        def is_null(self):
            return Expr(lambda df: self(df).isna())

        def is_not_null(self):
            return Expr(lambda df: self(df).notna())

        def is_not_nan(self):
            return Expr(lambda df: ~self(df).isna())

        def eq(self, val):
            return Expr(lambda df: self(df) == val)

        def mean(self):
            return Expr(lambda df: self(df).mean())

        def max(self):
            return Expr(lambda df: self(df).max())

        def min(self):
            return Expr(lambda df: self(df).min())

        def sum(self):
            return Expr(lambda df: self(df).sum())

        def std(self, ddof=1):
            return Expr(lambda df: self(df).std(ddof=ddof))

        def var(self, ddof=0):
            return Expr(lambda df: self(df).var(ddof=ddof))

        def n_unique(self):
            return Expr(lambda df: self(df).nunique())

        def cum_sum(self):
            return Expr(lambda df: self(df).cumsum())

        def shift(self, n=1):
            return Expr(lambda df: self(df).shift(n))

        def cast(self, dtype):
            return self

        def over(self, *args, **kwargs):
            return self

        def __add__(self, other):
            return Expr(lambda df: self(df) + (other(df) if isinstance(other, Expr) else other))

        def __sub__(self, other):
            return Expr(lambda df: self(df) - (other(df) if isinstance(other, Expr) else other))

        def __mul__(self, other):
            return Expr(lambda df: self(df) * (other(df) if isinstance(other, Expr) else other))

        def __truediv__(self, other):
            return Expr(lambda df: self(df) / (other(df) if isinstance(other, Expr) else other))

        def __eq__(self, other):
            return Expr(lambda df: self(df) == (other(df) if isinstance(other, Expr) else other))

        def __ne__(self, other):
            return Expr(lambda df: self(df) != (other(df) if isinstance(other, Expr) else other))

        def __ge__(self, other):
            return Expr(lambda df: self(df) >= (other(df) if isinstance(other, Expr) else other))

        def __le__(self, other):
            return Expr(lambda df: self(df) <= (other(df) if isinstance(other, Expr) else other))

        def __gt__(self, other):
            return Expr(lambda df: self(df) > (other(df) if isinstance(other, Expr) else other))

        def __lt__(self, other):
            return Expr(lambda df: self(df) < (other(df) if isinstance(other, Expr) else other))

        def __and__(self, other):
            return Expr(lambda df: self(df) & (other(df) if isinstance(other, Expr) else other))

        def __or__(self, other):
            return Expr(lambda df: self(df) | (other(df) if isinstance(other, Expr) else other))

    class WhenExpr:
        def __init__(self, cond_expr):
            self.cond_expr = cond_expr

        def then(self, val):
            def _then(df):
                cond_s = self.cond_expr(df) if isinstance(self.cond_expr, Expr) else self.cond_expr
                val_s = val(df) if isinstance(val, Expr) else val
                return pd.Series(np.where(cond_s, val_s, np.nan), index=df.index)
            return Expr(_then)

    def col(name):
        return Expr(lambda df: df[name])

    def when(cond):
        return WhenExpr(cond)

    class PLSeries:
        def __init__(self, s):
            self._s = s

        @property
        def dtype(self):
            return self._s.dtype

        @property
        def shape(self):
            return self._s.shape

        def unique(self):
            return PLSeries(pd.Series(self._s.unique()))

        def to_list(self):
            return self._s.tolist()

        def to_numpy(self):
            return self._s.to_numpy()

        def abs(self):
            return PLSeries(self._s.abs())

        def replace(self, old_val, new_val):
            return PLSeries(self._s.replace(old_val, new_val))

        def alias(self, name):
            s_new = self._s.copy()
            s_new.name = name
            return PLSeries(s_new)

        def drop_nulls(self):
            return PLSeries(self._s.dropna())

        def is_null(self):
            return PLSeries(self._s.isna())

        def is_not_null(self):
            return PLSeries(~self._s.isna())

        def cast(self, dtype):
            return self

        def __getitem__(self, item):
            res = self._s[item]
            if isinstance(res, pd.Series):
                return PLSeries(res)
            return res

    class PLDataFrame:
        def __init__(self, df):
            if isinstance(df, PLDataFrame):
                self._df = df._df.copy()
            else:
                self._df = df.copy()

        @property
        def height(self):
            return len(self._df)

        @property
        def columns(self):
            return list(self._df.columns)

        @property
        def shape(self):
            return self._df.shape

        def filter(self, expr):
            mask = expr(self._df) if callable(expr) else expr
            if isinstance(mask, PLSeries):
                mask = mask._s
            return PLDataFrame(self._df[mask])

        def rename(self, mapping):
            return PLDataFrame(self._df.rename(columns=mapping))

        def with_columns(self, *exprs):
            df_new = self._df.copy()
            for e in exprs:
                s = e(df_new)
                if isinstance(s, PLSeries):
                    s = s._s
                df_new[s.name] = s
            return PLDataFrame(df_new)

        def select(self, cols):
            if isinstance(cols, list):
                names = [c.name if isinstance(c, Expr) else c for c in cols]
                return PLDataFrame(self._df[names])
            return PLDataFrame(self._df[[cols]])

        def drop(self, *cols):
            return PLDataFrame(self._df.drop(columns=list(cols), errors="ignore"))

        def sort(self, by, descending=False):
            return PLDataFrame(self._df.sort_values(by=by, ascending=not descending))

        def to_pandas(self):
            return self._df

        def __getitem__(self, item):
            if isinstance(item, str):
                return PLSeries(self._df[item])
            return self._df[item]

    mod = types.ModuleType("polars")
    mod._is_shim = True
    mod.col = col
    mod.when = when
    mod.lit = lambda x: Expr(lambda df: x)
    mod.Float32 = np.float32
    mod.Float64 = np.float64
    mod.Int64 = np.int64
    mod.Utf8 = str
    mod.read_csv = lambda p, **kw: PLDataFrame(pd.read_csv(p, **kw))
    mod.DataFrame = PLDataFrame
    mod.concat = lambda dfs: PLDataFrame(pd.concat([d.to_pandas() for d in dfs], ignore_index=True))

    sys.modules["polars"] = mod
    return mod
