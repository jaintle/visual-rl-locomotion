"""
Lightweight CSV metrics logger.

Appends one row per log() call.  Missing fields are written as empty
strings (not NaN) so the file remains valid CSV throughout training.
The header is written exactly once when the logger is first created.
"""

import csv
import os
from typing import Any, Dict, List


# Canonical column order for metrics.csv.
METRIC_FIELDS: List[str] = [
    "global_step",
    "episode_return",
    "episode_length",
    "eval_return_mean",
    "eval_return_std",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
]


class CSVLogger:
    """
    Append-mode CSV writer with a fixed set of column names.

    Usage::

        logger = CSVLogger("runs/run1/metrics.csv")
        logger.log({"global_step": 256, "policy_loss": 0.03, ...})
    """

    def __init__(self, path: str, fieldnames: List[str] = METRIC_FIELDS) -> None:
        """
        Args:
            path:       Destination CSV file path. Parent dirs are created.
            fieldnames: Ordered list of column names.  Defaults to the
                        canonical METRIC_FIELDS list required by CLAUDE.md.
        """
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Write header once; truncate any previous file.
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """
        Append one row to the CSV.

        Any key in *row* that is not in self.fieldnames is silently ignored.
        Any fieldname not present in *row* is written as an empty string.

        Args:
            row: Dict mapping column names to values.
        """
        full_row = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(full_row)
