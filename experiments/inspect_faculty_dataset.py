"""Inspect and visualize the faculty workbook without coupling it to SP-PF.

Run from the repository root:
``python -m experiments.inspect_faculty_dataset``

The default discovers either ``data_statedata/`` or the workbook currently
provided at ``data_statedata.xlsx``.  It writes a JSON report and voltage plots
under ``results/``; it never alters the raw source data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook

try:
    from data.faculty_adapter import load_faculty_scenario
except ModuleNotFoundError:  # Support direct execution as well as -m invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.faculty_adapter import load_faculty_scenario


SUPPORTED_WORKBOOKS = {".xlsx", ".xls", ".xlsm"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, count: int = 5) -> list[dict[str, object]]:
    return [{str(key): _json_value(value) for key, value in record.items()} for record in frame.head(count).to_dict(orient="records")]


def _tail_records(frame: pd.DataFrame, count: int = 5) -> list[dict[str, object]]:
    return [{str(key): _json_value(value) for key, value in record.items()} for record in frame.tail(count).to_dict(orient="records")]


def _units(column: object) -> str | None:
    match = re.search(r"\(([^)]+)\)", str(column))
    return match.group(1).strip() if match else None


def _time_summary(frame: pd.DataFrame) -> dict[str, object] | None:
    time_column = next((column for column in frame.columns if re.sub(r"[^a-z0-9]", "", str(column).lower()) in {"time", "timems"}), None)
    if time_column is None:
        return None
    values = pd.to_numeric(frame[time_column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        return {"column": str(time_column), "unit": _units(time_column), "time_start": None, "time_end": None}
    differences = np.diff(values)
    return {
        "column": str(time_column), "unit": _units(time_column),
        "time_start": float(values[0]), "time_end": float(values[-1]),
        "mean_dt": float(np.mean(differences)), "min_dt": float(np.min(differences)), "max_dt": float(np.max(differences)),
        "is_uniform": bool(np.allclose(differences, differences[0], rtol=1e-9, atol=1e-12)),
        "strictly_increasing": bool(np.all(differences > 0)),
    }


def _signal_summary(frame: pd.DataFrame, time_column: str | None) -> dict[str, object]:
    output: dict[str, object] = {}
    dt_seconds: float | None = None
    if time_column is not None and _units(time_column) and _units(time_column).lower() in {"ms", "millisecond", "milliseconds"}:
        time_values = pd.to_numeric(frame[time_column], errors="coerce").dropna().to_numpy(dtype=float)
        if time_values.size > 1:
            dt_seconds = float(np.mean(np.diff(time_values))) * 1e-3
    for column in frame.select_dtypes(include="number").columns:
        if str(column) == time_column:
            continue
        values = frame[column].dropna().to_numpy(dtype=float)
        entry: dict[str, object] = {
            "unit": _units(column), "min": float(np.min(values)), "max": float(np.max(values)),
            "mean": float(np.mean(values)), "std": float(np.std(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
            "zero_crossings": int(np.count_nonzero(np.diff(np.signbit(values)))),
        }
        if dt_seconds and values.size > 2:
            spectrum = np.abs(np.fft.rfft(values - np.mean(values)))
            frequencies = np.fft.rfftfreq(values.size, d=dt_seconds)
            peak_index = int(np.argmax(spectrum[1:]) + 1)
            entry["dominant_frequency_hz"] = float(frequencies[peak_index])
        output[str(column)] = entry
    return output


def _representation_assessment(frame: pd.DataFrame, time: dict[str, object] | None, signals: dict[str, object]) -> str:
    phase_columns = [name for name in signals if re.fullmatch(r"v[abc].*", re.sub(r"[^a-z0-9]", "", name.lower()))]
    signed = all(float(signals[name]["min"]) < 0 < float(signals[name]["max"]) for name in phase_columns)
    fast_uniform = bool(time and time.get("is_uniform") and time.get("mean_dt") and float(time["mean_dt"]) <= 0.01)
    if len(phase_columns) == 3 and signed and fast_uniform:
        return "Evidence supports instantaneous sampled three-phase voltage waveforms in the labelled kV unit; this is not a voltage-phasor/PMU representation."
    return "Representation cannot be conclusively determined from the provided data."


def _sheet_report(workbook: Path, sheet: str) -> dict[str, object]:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    time = _time_summary(frame)
    signals = _signal_summary(frame, time["column"] if time else None)
    return {
        "sheet": sheet, "rows": int(frame.shape[0]), "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "data_types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {str(column): int(count) for column, count in frame.isna().sum().items()},
        "duplicated_rows": int(frame.duplicated().sum()),
        "first_rows": _records(frame), "last_rows": _tail_records(frame),
        "time": time, "signal_ranges": signals,
        "representation_assessment": _representation_assessment(frame, time, signals),
    }


def _workbook_metadata(workbook: Path) -> dict[str, object]:
    """Read available Office metadata; absence of a description is reported."""
    properties = load_workbook(workbook, read_only=True).properties
    return {
        "creator": properties.creator,
        "title": properties.title,
        "subject": properties.subject,
        "description": properties.description,
        "keywords": properties.keywords,
        "created": properties.created.isoformat() if properties.created else None,
        "modified": properties.modified.isoformat() if properties.modified else None,
    }


def _scenario_observations(workbooks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Report numeric differences without inferring causes from scenario names."""
    observations: list[dict[str, object]] = []
    for workbook in workbooks:
        for sheet in workbook["sheets"]:
            signals = sheet["signal_ranges"]
            rms_values = [float(value["rms"]) for value in signals.values()]
            peak_values = [max(abs(float(value["min"])), abs(float(value["max"]))) for value in signals.values()]
            observations.append({
                "scenario_sheet_name": sheet["sheet"],
                "mean_phase_rms_kv": float(np.mean(rms_values)),
                "maximum_absolute_phase_voltage_kv": float(np.max(peak_values)),
                "statement": "This is an observed waveform-amplitude comparison only; the sheet name supplies no validated physical mechanism or event metadata.",
            })
    return observations


def _plot_scenario(workbook: Path, sheet: str, output_dir: Path) -> list[str]:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adapted = load_faculty_scenario(workbook, sheet)
    time_ms = adapted["time_ms"]
    voltages = adapted["measurements_kv"]
    slug = _slug(sheet)
    created: list[str] = []
    fig, axis = plt.subplots(figsize=(11, 4.5))
    for index, label in enumerate(adapted["measurement_labels"]):
        axis.plot(time_ms, voltages[:, index], label=label, linewidth=0.75)
    axis.set(title=f"{sheet}: three-phase voltage", xlabel="Time (ms)", ylabel="Voltage (kV)")
    axis.grid(alpha=0.25); axis.legend(ncol=3); fig.tight_layout()
    path = output_dir / f"{slug}_voltage.png"
    fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    # A short initial window makes the high-frequency three-phase waveform visible.
    zoom_end = time_ms[0] + min(0.10, float(time_ms[-1] - time_ms[0]))
    selection = time_ms <= zoom_end
    fig, axis = plt.subplots(figsize=(11, 4.5))
    for index, label in enumerate(adapted["measurement_labels"]):
        axis.plot(time_ms[selection], voltages[selection, index], label=label, linewidth=1.0)
    axis.set(title=f"{sheet}: voltage waveform detail", xlabel="Time (ms)", ylabel="Voltage (kV)")
    axis.grid(alpha=0.25); axis.legend(ncol=3); fig.tight_layout()
    path = output_dir / f"{slug}_voltage_zoom.png"
    fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))
    return created


def _discover(source: str | Path | None) -> list[Path]:
    if source is not None:
        candidate = Path(source)
    elif Path("data_statedata").exists():
        candidate = Path("data_statedata")
    elif Path("data_statedata.xlsx").is_file():
        candidate = Path("data_statedata.xlsx")
    else:
        raise FileNotFoundError("Could not find data_statedata/ or data_statedata.xlsx")
    return [candidate] if candidate.is_file() else sorted(path for path in candidate.rglob("*") if path.is_file())


def inspect_faculty_dataset(source: str | Path | None = None, make_plots: bool = True) -> dict[str, object]:
    """Inspect all discovered files, write JSON, and optionally plot workbooks."""
    files = _discover(source)
    output_dir = Path("results")
    plot_dir = output_dir / "faculty_data"
    report: dict[str, object] = {
        "inspection_title": "FACULTY DATASET INSPECTION",
        "files": [{"path": str(path), "file_type": path.suffix.lower() or "no_extension", "bytes": path.stat().st_size} for path in files],
        "workbooks": [],
        "integration_assessment": (
            "The faculty dataset is retained as an external measurement dataset and requires a measurement/state observation model before it can be used directly by the current SP-PF state estimator. "
            "Its three phase voltages are not equivalent to [delta, omega] generator states or to the synthetic model's nine measurements."
        ),
    }
    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_WORKBOOKS:
            continue
        excel = pd.ExcelFile(file_path)
        workbook: dict[str, object] = {
            "file": str(file_path), "sheet_names": excel.sheet_names,
            "workbook_metadata": _workbook_metadata(file_path), "sheets": [],
        }
        for sheet in excel.sheet_names:
            sheet_info = _sheet_report(file_path, sheet)
            if make_plots:
                sheet_info["plots"] = _plot_scenario(file_path, sheet, plot_dir)
            workbook["sheets"].append(sheet_info)
        report["workbooks"].append(workbook)
    report["scenario_observations"] = _scenario_observations(report["workbooks"])
    output_dir.mkdir(exist_ok=True)
    (output_dir / "faculty_dataset_summary.json").write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")
    return report


def print_report(report: dict[str, object]) -> None:
    print("=" * 53)
    print("FACULTY DATASET INSPECTION")
    print("=" * 53)
    print("Files:")
    for file in report["files"]:
        print(f"  {file['path']} ({file['file_type']}, {file['bytes']} bytes)")
    for workbook in report["workbooks"]:
        print(f"\nWorkbook: {workbook['file']}")
        print("Sheets:", ", ".join(workbook["sheet_names"]))
        print("Workbook metadata:", workbook["workbook_metadata"])
        for sheet in workbook["sheets"]:
            print(f"\nSheet: {sheet['sheet']}")
            print(f"Rows: {sheet['rows']}; Columns: {sheet['columns']}")
            print("Columns:", ", ".join(sheet["column_names"]))
            print("Data types:", sheet["data_types"])
            print("Missing values:", sheet["missing_values"], "; duplicated rows:", sheet["duplicated_rows"])
            print("Time:", sheet["time"])
            print("Signal ranges:")
            for name, values in sheet["signal_ranges"].items():
                print(f"  {name}: {values}")
            print("Representation:", sheet["representation_assessment"])
            print("First rows:", sheet["first_rows"])
            print("Last rows:", sheet["last_rows"])
    print("\nScenario observations (numeric only):")
    for observation in report["scenario_observations"]:
        print(" ", observation)
    print("\nSP-PF integration:", report["integration_assessment"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw faculty dataset files and create validation plots.")
    parser.add_argument("source", nargs="?", help="Workbook or directory; default discovers data_statedata")
    parser.add_argument("--no-plots", action="store_true", help="Write report only")
    arguments = parser.parse_args()
    print_report(inspect_faculty_dataset(arguments.source, make_plots=not arguments.no_plots))
