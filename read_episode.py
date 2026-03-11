#!/usr/bin/env python3
"""Read and inspect a recorded episode parquet file."""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/my_task/data/chunk-000/episode_000003.parquet"
df = pd.read_parquet(path)

print(f"File: {path}")
print(f"Rows: {len(df)}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    val = df[col].iloc[0]
    print(f"  {col:40s}  dtype={str(df[col].dtype):10s}")
