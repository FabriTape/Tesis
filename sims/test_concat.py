import polars as pl

df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
df1 = df1.with_columns([
    pl.Series("c", [5, 6], dtype=pl.Float64),
    pl.Series("d", [7, 8], dtype=pl.UInt32)
])

df2 = pl.DataFrame({"a": [], "b": []})
# If we don't add columns c and d to df2:
try:
    pl.concat([df1, df2], how="vertical_relaxed")
except Exception as e:
    print("Error without columns:", e)

# If we add them:
df2 = df2.with_columns([
    pl.lit(None).alias("c").cast(pl.Float64),
    pl.lit(None).alias("d").cast(pl.UInt32)
])
res = pl.concat([df1, df2], how="vertical_relaxed")
print("Success with columns:", res.shape)
