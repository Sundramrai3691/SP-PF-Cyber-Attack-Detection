"""Non-destructive adapter for the faculty three-phase voltage workbook.

This module intentionally returns voltage measurements only.  It does not
claim that phase voltages are generator angles/speeds or that they can feed the
current synthetic SP-PF observation function without a validated model.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def _normalise_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def list_faculty_scenarios(workbook: str | Path) -> list[str]:
    """Return workbook sheet names without altering the source workbook."""
    return pd.ExcelFile(workbook).sheet_names


def _find_time_column(frame: pd.DataFrame) -> str:
    matches = [str(column) for column in frame.columns if _normalise_column_name(column) in {"time", "timems"}]
    if len(matches) != 1:
        raise ValueError("Expected exactly one Time or Time (ms) column")
    return matches[0]


def _phase_column_map(frame: pd.DataFrame) -> dict[str, str]:
    found: dict[str, str] = {}
    for column in frame.columns:
        normalised = _normalise_column_name(column)
        match = re.fullmatch(r"v([abc])(?:in)?kv", normalised)
        if match:
            found[match.group(1).upper()] = str(column)
    if set(found) != {"A", "B", "C"}:
        raise ValueError("Expected one voltage-in-kV column for each phase A, B, and C")
    return found


def load_faculty_scenario(workbook: str | Path, scenario: str) -> dict[str, Any]:
    """Load one scenario as an ordered, clean three-phase voltage structure.

    Returned ``measurements_kv`` has columns A, B, C regardless of raw Excel
    column order.  Time is preserved both in workbook milliseconds and seconds.
    No resampling, filtering, phasor extraction, or state inference occurs.
    """
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(f"Faculty workbook not found: {workbook}")
    frame = pd.read_excel(workbook, sheet_name=scenario)
    time_column = _find_time_column(frame)
    phase_columns = _phase_column_map(frame)
    selected_columns = [time_column, phase_columns["A"], phase_columns["B"], phase_columns["C"]]
    numeric = frame.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"Scenario {scenario!r} has missing/non-numeric time or phase-voltage values")
    time_ms = numeric[time_column].to_numpy(dtype=float)
    if time_ms.size < 2 or not np.all(np.diff(time_ms) > 0):
        raise ValueError(f"Scenario {scenario!r} time values must be strictly increasing")
    dt_ms = np.diff(time_ms)
    return {
        "scenario": scenario,
        "source_file": str(workbook),
        "time_ms": time_ms,
        "time_s": time_ms * 1e-3,
        "measurements_kv": numeric[[phase_columns["A"], phase_columns["B"], phase_columns["C"]]].to_numpy(dtype=float),
        "measurement_labels": ("VA (kV)", "VB (kV)", "VC (kV)"),
        "raw_columns": {"time": time_column, "A": phase_columns["A"], "B": phase_columns["B"], "C": phase_columns["C"]},
        "sampling": {
            "mean_dt_ms": float(np.mean(dt_ms)),
            "min_dt_ms": float(np.min(dt_ms)),
            "max_dt_ms": float(np.max(dt_ms)),
            "is_uniform": bool(np.allclose(dt_ms, dt_ms[0], rtol=1e-9, atol=1e-12)),
        },
    }
