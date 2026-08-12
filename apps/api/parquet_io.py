"""Read signal samples back out of Parquet by `storage_uri`.

Mirrors `TRD.md` §8: raw time-history samples live in Parquet, not Postgres;
`Signal.storage_uri` (packages/domain/core.py) points at the file written by
`scripts/generate_synthetic_dataset.py`.
"""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
import pandas as pd


def read_signal(storage_uri: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(time_s, values)` for one column of a signal-group Parquet file."""
    path = urlparse(storage_uri).path if "://" in storage_uri else storage_uri
    # Windows: urlparse on "file:///H:/x" yields "/H:/x" — strip the leading slash.
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    # pandas-stubs' `engine="pyarrow"` overload requires `to_pandas_kwargs`;
    # omit `engine` and let pandas auto-select the installed pyarrow backend.
    df = pd.read_parquet(path, columns=["time_s", column])
    return df["time_s"].to_numpy(), df[column].to_numpy()
