"""One-off inspection of both faculty Excel workbooks. Temporary helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "New Microsoft Excel Worksheet.xlsx",
    ROOT / "data_statedata.xlsx",
]


def time_like(name: object) -> bool:
    n = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return n in {"time", "timems", "t"} or n.startswith("time")


def unwrap_hint(name: str, vals: np.ndarray) -> dict:
    out: dict = {}
    jumps = np.abs(np.diff(vals)) if vals.size > 1 else np.array([])
    out["max_abs_step"] = float(jumps.max()) if jumps.size else None
    out["large_jumps_gt_pi"] = int(np.sum(jumps > np.pi)) if jumps.size else 0
    out["large_jumps_gt_pi_over_2"] = int(np.sum(jumps > (np.pi / 2)) ) if jumps.size else 0
    out["range"] = [float(vals.min()), float(vals.max())]
    out["looks_wrapped_0_2pi"] = bool(vals.min() >= -0.2 and vals.max() <= 2 * np.pi + 0.2 and (vals.max() - vals.min()) > 3)
    out["looks_wrapped_pm_pi"] = bool(vals.min() >= -np.pi - 0.2 and vals.max() <= np.pi + 0.2 and (vals.max() - vals.min()) > 3)
    out["looks_deg_0_360"] = bool(vals.min() >= -1 and vals.max() <= 361 and (vals.max() - vals.min()) > 300)
    unwrapped = np.unwrap(vals)
    out["unwrapped_range"] = [float(unwrapped.min()), float(unwrapped.max())]
    unwrapped_deg = np.unwrap(np.deg2rad(vals))
    out["if_deg_unwrapped_rad_range"] = [float(unwrapped_deg.min()), float(unwrapped_deg.max())]
    return out


def inspect_sheet(path: Path, sheet: str) -> dict:
    df = pd.read_excel(path, sheet_name=sheet)
    info: dict = {
        "sheet": sheet,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": {str(c): int(v) for c, v in df.isna().sum().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "head": json.loads(df.head(5).to_json(orient="records")),
        "tail": json.loads(df.tail(5).to_json(orient="records")),
    }
    numeric: dict = {}
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        entry = {
            "n": int(vals.size),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "first": float(vals[0]),
            "last": float(vals[-1]),
        }
        if time_like(c) and vals.size > 1:
            diffs = np.diff(vals)
            entry["dt_mean"] = float(np.mean(diffs))
            entry["dt_min"] = float(np.min(diffs))
            entry["dt_max"] = float(np.max(diffs))
            entry["strictly_increasing"] = bool(np.all(diffs > 0))
            entry["uniform"] = bool(np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12))
            entry["duration"] = float(vals[-1] - vals[0])
        lowered = str(c).lower()
        if any(k in lowered for k in ("angle", "delta", "rotor", "load")):
            entry["angle_wrap"] = unwrap_hint(str(c), vals)
        numeric[str(c)] = entry
    info["numeric"] = numeric

    tcol = next((c for c in df.columns if time_like(c)), None)
    vcols = [c for c in df.columns if any(tok in str(c).lower().replace(" ", "") for tok in ("va", "vb", "vc")) or "kv" in str(c).lower()]
    if tcol is not None and vcols:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        dt = float(np.mean(np.diff(t)))
        x = pd.to_numeric(df[vcols[0]], errors="coerce").dropna().to_numpy(dtype=float)
        zc = int(np.count_nonzero(np.diff(np.signbit(x))))
        freq = {
            "time_col": str(tcol),
            "voltage_col": str(vcols[0]),
            "voltage_columns": [str(c) for c in vcols],
            "dt_in_column_units": dt,
            "duration_in_column_units": float(t[-1] - t[0]),
            "zero_crossings": zc,
        }
        for name, dts in [
            ("literal_Time_ms_to_seconds", dt * 1e-3),
            ("treat_column_values_as_seconds", dt),
        ]:
            spec = np.abs(np.fft.rfft(x - x.mean()))
            freqs = np.fft.rfftfreq(x.size, d=dts)
            peak = int(np.argmax(spec[1:]) + 1)
            freq[name] = {
                "dt_s": dts,
                "fs_hz": 1.0 / dts,
                "fft_peak_hz": float(freqs[peak]),
                "f0_from_zero_crossings_hz": (zc / 2.0) / (x.size * dts),
            }
        info["frequency_interpretations"] = freq
    return info


def main() -> None:
    report = {"files": []}
    for path in FILES:
        xl = pd.ExcelFile(path)
        print("=" * 72)
        print(f"{path.name}  sheets={xl.sheet_names}  bytes={path.stat().st_size}")
        wb = {"path": str(path), "bytes": path.stat().st_size, "sheet_names": xl.sheet_names, "sheets": []}
        for sheet in xl.sheet_names:
            info = inspect_sheet(path, sheet)
            wb["sheets"].append(info)
            print(f"\n  SHEET: {info['sheet']}  rows={info['rows']} cols={info['cols']}")
            print(f"  columns: {info['columns']}")
            print(f"  missing: {info['missing']}  dups={info['duplicated_rows']}")
            print(f"  dtypes: {info['dtypes']}")
            print("  HEAD:")
            for row in info["head"][:3]:
                print("   ", row)
            print("  TAIL:")
            for row in info["tail"][-2:]:
                print("   ", row)
            for name, stats in info["numeric"].items():
                compact = {k: stats[k] for k in stats if k != "angle_wrap"}
                print(f"  RANGE {name}: {compact}")
                if "angle_wrap" in stats:
                    print(f"    WRAP {name}: {stats['angle_wrap']}")
            if "frequency_interpretations" in info:
                print(f"  FREQ: {info['frequency_interpretations']}")
        report["files"].append(wb)

    # Alignment: do not assume row i matches until checked.
    print("\n" + "=" * 72)
    print("CROSS-FILE ALIGNMENT")
    state_wb = report["files"][0]
    volt_wb = report["files"][1]
    pairs = [
        ("Stable swing", "Stable swing"),
        ("Voltage instable", "Voltage stress"),
        ("Load encroachment", "load encroachement"),
    ]
    alignment = []
    for a, b in pairs:
        sa = next((s for s in state_wb["sheets"] if s["sheet"] == a), None)
        vb = next((s for s in volt_wb["sheets"] if s["sheet"] == b), None)
        rec = {"state_sheet": a, "voltage_sheet": b, "state_found": sa is not None, "voltage_found": vb is not None}
        if sa and vb:
            rec["row_counts"] = {"state": sa["rows"], "voltage": vb["rows"], "equal": sa["rows"] == vb["rows"]}
            rec["has_shared_time_column"] = any("time" in c.lower() for c in sa["columns"]) and any(
                "time" in c.lower() for c in vb["columns"]
            )
            rec["state_columns"] = sa["columns"]
            rec["voltage_columns"] = vb["columns"]
            rec["can_align_by_identical_time_vector"] = False
            rec["can_align_by_equal_row_index"] = sa["rows"] == vb["rows"]
            rec["note"] = (
                "State workbook has no Time column in typical P/angle sheets; "
                "voltage workbook has Time (ms). Equal row counts would be necessary but not sufficient."
            )
        alignment.append(rec)
        print(rec)
    report["alignment"] = alignment
    out = ROOT / "results" / "faculty_both_workbooks_inspection.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
