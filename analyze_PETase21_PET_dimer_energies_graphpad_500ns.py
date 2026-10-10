# -*- coding: utf-8 -*-
"""
NAMD energy analysis for PETase21 + PET dimer.

Inputs
------
Topology (provenance/validation only):
    ionized.psf

Trajectory (provenance/validation only):
    ionized.dcd

Energy log actually parsed:
    energia-petase21-pet-dimer-coo-and-oh-ends.log

Single-log timestep handling
----------------------------
ENERGY records are ordered by global NAMD timestep TS. If the same TS is
printed more than once within the log, the last printed occurrence is retained.
The original TS values are preserved in the exported tables.

Forced 0-500 ns plotting scale
------------------------------
Original NAMD timesteps are preserved. Final figures additionally use:

    TIME_NS = (TS - min(TS)) / (max(TS) - min(TS)) * 500

Thus the earliest retained ENERGY record maps to 0 ns and the latest maps to
500 ns. This is a user-requested plotting/data rescaling. It does not alter
energy, temperature, pressure, or volume values and does not independently
establish 500 ns of physical MD time.

The script also attempts to detect the NAMD integration timestep from the log
and exports a nominal TS-derived time column for reference.

GraphPad-like formatting
------------------------
- gray plotting area
- black full frame
- mirrored inward major/minor ticks
- white dashed major grid
- bold typography
- retained five-color palette
- exact axes plotting areas:
    wide   = 7.62 x 5.08 cm
    square = 5.08 x 5.08 cm

Outputs
-------
- processed energy CSV
- overall summary CSV
- summary-by-source CSV
- duplicate-timestep audit CSV
- parsing report TXT
- individual plots for every detected numeric NAMD parameter
- combined energy-components plot
- total/potential/kinetic plot
- temperature plot
- pressure plot
- volume plot
- wide and square versions of all figures

Required packages:
    numpy, pandas, matplotlib
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica"
    r"\petase21-PET-02-dimer-coo-and-oh-ends"
)

PSF_FILE = BASE_DIR / "ionized.psf"

# Single root-level trajectory folder used for provenance validation.
TRAJECTORY_DIRS = [
    BASE_DIR,
]

ENERGY_LOG_NAMES = [
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
]


def _natural_key(path):
    """Natural filename sorting: file2 before file10."""
    name = Path(path).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


def discover_dcd_files(run_dir):
    """
    Discover DCD files without guessing their exact filenames.

    DCDs directly inside the supplied folder are preferred. If none are found
    directly, the folder is searched recursively.
    """
    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Trajectory directory not found:\n{run_dir}"
        )

    direct = sorted(
        [
            path for path in run_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".dcd"
        ],
        key=_natural_key,
    )

    if direct:
        return direct

    recursive = sorted(
        [
            path for path in run_dir.rglob("*.dcd")
            if path.is_file()
        ],
        key=_natural_key,
    )

    if not recursive:
        raise FileNotFoundError(
            "No DCD trajectory was found in or below:\n"
            f"{run_dir}"
        )

    return recursive


def discover_named_file(filename):
    """
    Find a named file in BASE_DIR or the supplied trajectory directory.

    The search then falls back to a recursive search below BASE_DIR.
    """
    direct_candidates = [
        BASE_DIR / filename,
        *(run_dir / filename for run_dir in TRAJECTORY_DIRS),
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    recursive_matches = sorted(
        [
            path for path in BASE_DIR.rglob(filename)
            if path.is_file()
        ],
        key=_natural_key,
    )

    if len(recursive_matches) == 1:
        return recursive_matches[0]

    if len(recursive_matches) > 1:
        warnings.warn(
            f"Multiple copies of {filename} were found. "
            f"Using: {recursive_matches[0]}"
        )
        return recursive_matches[0]

    raise FileNotFoundError(
        "Required file was not found anywhere below BASE_DIR:\n"
        f"{filename}\n\nBASE_DIR:\n{BASE_DIR}"
    )


# PSF/DCDs are provenance/validation inputs for this log-based energy analysis.
DCD_FILES = []
for trajectory_dir in TRAJECTORY_DIRS:
    DCD_FILES.extend(
        discover_dcd_files(trajectory_dir)
    )

LOG_SPECS = [
    {
        "path": discover_named_file(
            "energia-petase21-pet-dimer-coo-and-oh-ends.log"
        ),
        "priority": 0,
        "label": "production",
    },
]

OUTPUT_DIR = BASE_DIR / "ENERGY_ANALYSIS_PETase21_PET_dimer_500ns"

PREFIX = "PETase21_PET_dimer"

OUTPUT_CSV = OUTPUT_DIR / f"{PREFIX}_energy_data_500ns.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{PREFIX}_energy_summary.csv"
OUTPUT_RUN_SUMMARY = OUTPUT_DIR / f"{PREFIX}_energy_summary_by_source.csv"
OUTPUT_OVERLAPS = OUTPUT_DIR / f"{PREFIX}_duplicate_TS_audit.csv"
OUTPUT_MERGE_REPORT = OUTPUT_DIR / f"{PREFIX}_energy_parse_report.txt"

OUTPUT_COMBINED = OUTPUT_DIR / f"{PREFIX}_energy_components.png"
OUTPUT_COMBINED_SQUARE = OUTPUT_DIR / f"{PREFIX}_energy_components_square.png"
OUTPUT_GLOBAL = OUTPUT_DIR / f"{PREFIX}_total_potential_kinetic.png"
OUTPUT_GLOBAL_SQUARE = OUTPUT_DIR / f"{PREFIX}_total_potential_kinetic_square.png"
OUTPUT_TEMPERATURE = OUTPUT_DIR / f"{PREFIX}_temperature.png"
OUTPUT_TEMPERATURE_SQUARE = OUTPUT_DIR / f"{PREFIX}_temperature_square.png"
OUTPUT_PRESSURE = OUTPUT_DIR / f"{PREFIX}_pressure.png"
OUTPUT_PRESSURE_SQUARE = OUTPUT_DIR / f"{PREFIX}_pressure_square.png"
OUTPUT_VOLUME = OUTPUT_DIR / f"{PREFIX}_volume.png"
OUTPUT_VOLUME_SQUARE = OUTPUT_DIR / f"{PREFIX}_volume_square.png"


# ============================================================
# ANALYSIS SETTINGS
# ============================================================
FORCED_TOTAL_TIME_NS = 500.0
TIME_TICKS_NS = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

# Used only for the nominal TS-derived reference-time column if timestep cannot
# be detected from the logs. Final figures use the forced 0-500 ns axis.
DEFAULT_DT_FS = 2.0

# The PSF/DCDs are not mathematically required for log-based energy analysis.
# Keeping this True checks that the supplied simulation files exist before run.
VALIDATE_SIMULATION_FILES = True


# ============================================================
# GRAPHPAD PRISM-LIKE STYLE
# ============================================================
GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"

GRAPH_COLORS = [
    "#58094F",
    "#841E62",
    "#F3AA20",
    "#2A445E",
    "#346B6D",
]

GRAPH_CMAP = LinearSegmentedColormap.from_list(
    "PETase21_PET_dimer_graphpad_gradient",
    GRAPH_COLORS,
)


def set_plot_style():
    plt.rcParams.update({
        "font.family": ["Josefin Sans", "Arial", "DejaVu Sans"],
        "font.weight": "bold",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
    })


def style_graphpad_axis(axis, show_grid=True):
    axis.set_facecolor(GRAPH_BACKGROUND)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRAPH_FRAME_COLOR)
        spine.set_linewidth(1.5)

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    axis.tick_params(
        axis="both", which="major", direction="in",
        top=True, right=True, length=6, width=1.5,
        color=GRAPH_FRAME_COLOR, labelcolor=GRAPH_FRAME_COLOR,
    )
    axis.tick_params(
        axis="both", which="minor", direction="in",
        top=True, right=True, length=3.5, width=1.1,
        color=GRAPH_FRAME_COLOR,
    )

    if show_grid:
        axis.grid(
            visible=True, which="major", axis="both",
            color=GRAPH_GRID_COLOR, linestyle=(0, (4, 4)),
            linewidth=1.35, alpha=1.0,
        )
        axis.grid(visible=False, which="minor", axis="both")

    for label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        label.set_fontweight("bold")


def set_plot_area_size(figure, axis, square=False):
    """Exact axes area: wide 7.62x5.08 cm; square 5.08x5.08 cm."""
    width = (5.08 if square else 7.62) / 2.54
    height = 5.08 / 2.54

    figure.canvas.draw()
    old_size = figure.get_size_inches()

    boxes = {
        item: item.get_position().get_points() * old_size
        for item in figure.axes
    }
    left, right = boxes[axis][:, 0]
    rows = sorted({tuple(np.round(box[:, 1], 9)) for box in boxes.values()})

    def resize_x(value):
        return value + (width - (right - left)) * np.clip(
            (value - left) / (right - left), 0.0, 1.0
        )

    def resize_y(value):
        return value + sum(
            (height - (top - bottom)) * np.clip(
                (value - bottom) / (top - bottom), 0.0, 1.0
            )
            for bottom, top in rows
        )

    new_width = resize_x(old_size[0])
    new_height = resize_y(old_size[1])
    figure.set_size_inches(new_width, new_height, forward=True)

    for item, box in boxes.items():
        x0, x1 = (resize_x(v) for v in box[:, 0])
        y0, y1 = (resize_y(v) for v in box[:, 1])
        item.set_position([
            x0 / new_width,
            y0 / new_height,
            (x1 - x0) / new_width,
            (y1 - y0) / new_height,
        ])


def configure_time_axis(axis):
    axis.set_xlim(0.0, FORCED_TOTAL_TIME_NS)
    axis.set_xticks(TIME_TICKS_NS)
    axis.set_xlabel("Time (ns; rescaled)")


# ============================================================
# FILE / LOG HELPERS
# ============================================================
def validate_inputs():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"Base directory not found:\n{BASE_DIR}")

    for spec in LOG_SPECS:
        if not spec["path"].is_file():
            raise FileNotFoundError(f"Energy log not found:\n{spec['path']}")

    if VALIDATE_SIMULATION_FILES:
        if not PSF_FILE.is_file():
            raise FileNotFoundError(f"PSF not found:\n{PSF_FILE}")
        for i, path in enumerate(DCD_FILES, start=1):
            if not path.is_file():
                raise FileNotFoundError(f"DCD {i} not found:\n{path}")


def detect_namd_timestep_fs(log_paths, fallback=DEFAULT_DT_FS):
    """Detect NAMD timestep from log text; otherwise return fallback."""
    patterns = [
        re.compile(
            r"^\s*(?:Info:\s*)?TIMESTEP\s*[:=]?\s*"
            r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*timestep\s+"
            r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\b",
            re.IGNORECASE,
        ),
    ]

    detected = []
    for path in log_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        value = float(match.group(1))
                        if value > 0.0 and np.isfinite(value):
                            detected.append(value)

    if detected:
        rounded = {round(v, 12) for v in detected}
        if len(rounded) > 1:
            warnings.warn(
                "Multiple timestep values were detected in the logs: "
                f"{sorted(rounded)}. Using the last detected value "
                f"({detected[-1]:g} fs)."
            )
        return float(detected[-1]), True

    warnings.warn(
        "Could not detect the NAMD integration timestep in the logs. "
        f"Using DEFAULT_DT_FS={fallback:g} fs only for the nominal "
        "TS-derived reference-time column."
    )
    return float(fallback), False


# ============================================================
# NAMD ENERGY PARSING
# ============================================================
def parse_namd_number(value):
    value = value.strip()
    if "*" in value:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_energy_log(log_path, source_priority, source_label):
    """Parse ETITLE/ENERGY records from one NAMD log."""
    energy_columns = None
    records = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if stripped.startswith("ETITLE:"):
                parts = stripped.split()
                detected_columns = parts[1:]
                if detected_columns:
                    energy_columns = detected_columns

            elif stripped.startswith("ENERGY:"):
                if energy_columns is None:
                    continue

                values = [parse_namd_number(v) for v in stripped.split()[1:]]

                if len(values) < len(energy_columns):
                    values += [np.nan] * (len(energy_columns) - len(values))
                elif len(values) > len(energy_columns):
                    values = values[:len(energy_columns)]

                record = dict(zip(energy_columns, values))
                record["SOURCE_LOG"] = Path(log_path).name
                record["SOURCE_LABEL"] = source_label
                record["SOURCE_PRIORITY"] = int(source_priority)
                record["SOURCE_LINE"] = int(line_number)
                records.append(record)

    if not records:
        raise RuntimeError(f"No ENERGY records could be extracted from:\n{log_path}")

    frame = pd.DataFrame(records)

    if "TS" not in frame.columns:
        raise RuntimeError(f"The NAMD ENERGY table has no TS column in:\n{log_path}")

    frame["TS"] = pd.to_numeric(frame["TS"], errors="coerce")
    frame = frame.dropna(subset=["TS"]).copy()
    frame["TS"] = frame["TS"].astype(np.int64)

    return frame


def merge_energy_logs(parsed_frames):
    """
    Order energy records by TS and remove duplicate TS values.

    For repeated TS values within the single production log, the later
    SOURCE_LINE wins.
    """
    raw = pd.concat(parsed_frames, ignore_index=True, sort=False)
    raw = raw.sort_values(
        ["TS", "SOURCE_PRIORITY", "SOURCE_LINE"],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_mask = raw.duplicated(subset=["TS"], keep=False)
    overlaps = raw.loc[duplicate_mask].copy()

    merged = (
        raw.drop_duplicates(subset=["TS"], keep="last")
        .sort_values("TS")
        .reset_index(drop=True)
    )

    if len(merged) < 2:
        raise RuntimeError("Fewer than two unique NAMD timesteps remained after merging.")

    return raw, merged, overlaps


def add_time_columns(dataframe, dt_fs):
    dataframe = dataframe.copy()

    dataframe.insert(
        1,
        "NAMD_NOMINAL_TIME_NS",
        dataframe["TS"] * float(dt_fs) / 1_000_000.0,
    )

    ts_min = float(dataframe["TS"].min())
    ts_max = float(dataframe["TS"].max())
    denominator = ts_max - ts_min

    if denominator <= 0.0:
        raise RuntimeError("Cannot construct 0-500 ns axis because TS range is zero.")

    forced_time = (
        (dataframe["TS"].astype(float) - ts_min)
        / denominator
        * FORCED_TOTAL_TIME_NS
    )

    dataframe.insert(2, "TIME_NS", forced_time)
    dataframe.insert(3, "RUN_PHASE", dataframe["SOURCE_LABEL"].astype(str))

    if not np.isclose(dataframe["TIME_NS"].iloc[0], 0.0):
        raise RuntimeError("Forced timeline does not begin at 0 ns.")
    if not np.isclose(dataframe["TIME_NS"].iloc[-1], FORCED_TOTAL_TIME_NS):
        raise RuntimeError("Forced timeline does not end at exactly 500 ns.")

    return dataframe


# ============================================================
# UNITS / LABELS
# ============================================================
ENERGY_TERMS = {
    "BOND", "ANGLE", "DIHED", "IMPRP", "ELECT", "VDW",
    "BOUNDARY", "MISC", "KINETIC", "TOTAL", "POTENTIAL",
    "TOTAL2", "TOTAL3",
}
TEMPERATURE_TERMS = {"TEMP", "TEMPAVG"}
PRESSURE_TERMS = {"PRESSURE", "GPRESSURE", "PRESSAVG", "GPRESSAVG"}


def parameter_ylabel(column):
    if column in ENERGY_TERMS:
        return f"{column} (kcal/mol)"
    if column in TEMPERATURE_TERMS:
        return f"{column} (K)"
    if column in PRESSURE_TERMS:
        return f"{column} (bar)"
    if column == "VOLUME":
        return "Volume (Å³)"
    return column


def line_colors(number):
    if number <= 1:
        return [GRAPH_COLORS[3]]
    return [GRAPH_CMAP(x) for x in np.linspace(0.0, 1.0, number)]


# ============================================================
# PLOTTING
# ============================================================
def save_single_parameter_plot(dataframe, column, output_path, square=False, color=None):
    set_plot_style()
    figure_size = (5.4, 5.4) if square else (6.6, 4.2)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    if square:
        axis.set_box_aspect(1)

    if color is None:
        color = GRAPH_COLORS[3]

    axis.plot(
        dataframe["TIME_NS"], dataframe[column],
        linewidth=1.05, color=color,
    )
    configure_time_axis(axis)
    axis.set_ylabel(parameter_ylabel(column))
    axis.set_title(f"PETase21–PET dimer: {column}")

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_multi_parameter_plot(
    dataframe, columns, output_path, title, ylabel, square=False
):
    if not columns:
        return

    set_plot_style()
    figure_size = (6.0, 6.0) if square else (7.2, 4.8)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    if square:
        axis.set_box_aspect(1)

    for color, column in zip(line_colors(len(columns)), columns):
        axis.plot(
            dataframe["TIME_NS"], dataframe[column],
            label=column, linewidth=1.0, color=color,
        )

    configure_time_axis(axis)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        prop={"weight": "bold", "size": 8},
    )

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 78)
    print("NAMD ENERGY ANALYSIS - PETase21 / PET dimer")
    print("Single production log; forced final plotting scale = 0-500 ns")
    print("=" * 78)

    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTopology:")
    print(PSF_FILE)
    print("\nTrajectory directories, chronological order:")
    for run_dir in TRAJECTORY_DIRS:
        print(f"  {run_dir}")

    print("\nTrajectory files (provenance):")
    for i, dcd in enumerate(DCD_FILES, start=1):
        print(f"  DCD {i}: {dcd}")

    parsed_frames = []
    resolved_logs = []

    for spec in LOG_SPECS:
        log_path = spec["path"]
        resolved_logs.append(log_path)

        print("\nInput log:")
        print(log_path)

        parsed = parse_energy_log(
            log_path=log_path,
            source_priority=spec["priority"],
            source_label=spec["label"],
        )
        parsed_frames.append(parsed)

        print(f"  ENERGY records extracted: {len(parsed):,}")
        print(f"  TS range: {parsed['TS'].min():,} to {parsed['TS'].max():,}")

    dt_fs, dt_detected = detect_namd_timestep_fs(resolved_logs)
    print(f"\nNAMD timestep used for nominal TS-derived time: {dt_fs:g} fs")
    print("Timestep source: " + ("detected from log" if dt_detected else "fallback"))

    raw_records, merged, overlaps = merge_energy_logs(parsed_frames)
    merged = add_time_columns(merged, dt_fs)

    print("\nMerge summary:")
    print(f"  Raw ENERGY records: {len(raw_records):,}")
    print(f"  Unique retained TS records: {len(merged):,}")
    print(f"  Duplicate/overlap rows encountered: {len(overlaps):,}")
    print(f"  Retained TS range: {merged['TS'].min():,} to {merged['TS'].max():,}")
    print(
        f"  Forced plotting time: {merged['TIME_NS'].iloc[0]:.6f} "
        f"to {merged['TIME_NS'].iloc[-1]:.6f} ns"
    )
    print(
        f"  Nominal TS-derived time: {merged['NAMD_NOMINAL_TIME_NS'].iloc[0]:.6f} "
        f"to {merged['NAMD_NOMINAL_TIME_NS'].iloc[-1]:.6f} ns"
    )

    # CSVs use semicolon + decimal comma, matching the previous script.
    merged.to_csv(
        OUTPUT_CSV, index=False, sep=";", decimal=",", encoding="utf-8-sig"
    )
    overlaps.to_csv(
        OUTPUT_OVERLAPS, index=False, sep=";", decimal=",", encoding="utf-8-sig"
    )

    metadata_numeric = {
        "TS", "NAMD_NOMINAL_TIME_NS", "TIME_NS",
        "SOURCE_PRIORITY", "SOURCE_LINE",
    }
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    analysis_numeric_cols = [c for c in numeric_cols if c not in metadata_numeric]

    summary = pd.DataFrame({
        "N": merged[analysis_numeric_cols].count(),
        "Mean": merged[analysis_numeric_cols].mean(),
        "SD": merged[analysis_numeric_cols].std(),
        "Median": merged[analysis_numeric_cols].median(),
        "Minimum": merged[analysis_numeric_cols].min(),
        "Maximum": merged[analysis_numeric_cols].max(),
    })
    summary.to_csv(
        OUTPUT_SUMMARY, sep=";", decimal=",", encoding="utf-8-sig"
    )

    run_rows = []
    for run_label in [spec["label"] for spec in LOG_SPECS]:
        run_df = merged[merged["RUN_PHASE"] == run_label]
        if run_df.empty:
            continue
        for column in analysis_numeric_cols:
            values = run_df[column].dropna()
            if values.empty:
                continue
            run_rows.append({
                "RUN_PHASE": run_label,
                "PARAMETER": column,
                "N": len(values),
                "Mean": values.mean(),
                "SD": values.std(),
                "Median": values.median(),
                "Minimum": values.min(),
                "Maximum": values.max(),
            })

    pd.DataFrame(run_rows).to_csv(
        OUTPUT_RUN_SUMMARY,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    print("\nCreating GraphPad-like individual parameter plots...")
    plot_columns = [
        c for c in analysis_numeric_cols
        if merged[c].notna().sum() > 0
    ]

    for column_index, column in enumerate(plot_columns):
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", column)
        color = GRAPH_COLORS[column_index % len(GRAPH_COLORS)]

        save_single_parameter_plot(
            merged,
            column,
            OUTPUT_DIR / f"{PREFIX}_{safe_name}.png",
            square=False,
            color=color,
        )
        save_single_parameter_plot(
            merged,
            column,
            OUTPUT_DIR / f"{PREFIX}_{safe_name}_square.png",
            square=True,
            color=color,
        )

    main_energy_terms = [
        "BOND", "ANGLE", "DIHED", "IMPRP",
        "ELECT", "VDW", "BOUNDARY", "MISC",
    ]
    available_main_terms = [
        c for c in main_energy_terms
        if c in merged.columns and merged[c].notna().sum() > 0
    ]

    save_multi_parameter_plot(
        merged,
        available_main_terms,
        OUTPUT_COMBINED,
        "PETase21–PET dimer: energy components",
        "Energy (kcal/mol)",
        square=False,
    )
    save_multi_parameter_plot(
        merged,
        available_main_terms,
        OUTPUT_COMBINED_SQUARE,
        "PETase21–PET dimer: energy components",
        "Energy (kcal/mol)",
        square=True,
    )

    global_energy_terms = ["TOTAL", "POTENTIAL", "KINETIC", "TOTAL2", "TOTAL3"]
    available_global_terms = [
        c for c in global_energy_terms
        if c in merged.columns and merged[c].notna().sum() > 0
    ]

    save_multi_parameter_plot(
        merged,
        available_global_terms,
        OUTPUT_GLOBAL,
        "PETase21–PET dimer: total, potential and kinetic energy",
        "Energy (kcal/mol)",
        square=False,
    )
    save_multi_parameter_plot(
        merged,
        available_global_terms,
        OUTPUT_GLOBAL_SQUARE,
        "PETase21–PET dimer: total, potential and kinetic energy",
        "Energy (kcal/mol)",
        square=True,
    )

    available_temperature = [
        c for c in ["TEMP", "TEMPAVG"]
        if c in merged.columns and merged[c].notna().sum() > 0
    ]
    save_multi_parameter_plot(
        merged, available_temperature, OUTPUT_TEMPERATURE,
        "PETase21–PET dimer: temperature", "Temperature (K)", False
    )
    save_multi_parameter_plot(
        merged, available_temperature, OUTPUT_TEMPERATURE_SQUARE,
        "PETase21–PET dimer: temperature", "Temperature (K)", True
    )

    available_pressure = [
        c for c in ["PRESSURE", "GPRESSURE", "PRESSAVG", "GPRESSAVG"]
        if c in merged.columns and merged[c].notna().sum() > 0
    ]
    save_multi_parameter_plot(
        merged, available_pressure, OUTPUT_PRESSURE,
        "PETase21–PET dimer: pressure", "Pressure (bar)", False
    )
    save_multi_parameter_plot(
        merged, available_pressure, OUTPUT_PRESSURE_SQUARE,
        "PETase21–PET dimer: pressure", "Pressure (bar)", True
    )

    if "VOLUME" in merged.columns and merged["VOLUME"].notna().sum() > 0:
        save_single_parameter_plot(
            merged, "VOLUME", OUTPUT_VOLUME,
            square=False, color=GRAPH_COLORS[4]
        )
        save_single_parameter_plot(
            merged, "VOLUME", OUTPUT_VOLUME_SQUARE,
            square=True, color=GRAPH_COLORS[4]
        )

    retained_by_source = merged["SOURCE_LOG"].value_counts().to_dict()
    duplicate_ts_count = overlaps["TS"].nunique() if not overlaps.empty else 0

    with open(OUTPUT_MERGE_REPORT, "w", encoding="utf-8") as report:
        report.write("PETase21 / PET dimer NAMD ENERGY PARSE REPORT\n")
        report.write("=" * 72 + "\n\n")
        report.write(f"Topology: {PSF_FILE}\n")
        report.write("Trajectory directory (provenance):\n")
        for run_dir in TRAJECTORY_DIRS:
            report.write(f"  {run_dir}\n")
        for i, dcd in enumerate(DCD_FILES, start=1):
            report.write(f"DCD {i}: {dcd}\n")
        report.write("\nInput logs:\n")
        for path in resolved_logs:
            report.write(f"  {path}\n")
        report.write("\n")
        report.write(f"Raw ENERGY records: {len(raw_records)}\n")
        report.write(f"Unique retained TS records: {len(merged)}\n")
        report.write(f"Unique duplicated TS values resolved: {duplicate_ts_count}\n")
        report.write(
            f"Retained TS range: {merged['TS'].min()} to {merged['TS'].max()}\n"
        )
        report.write(
            f"NAMD timestep used for nominal time: {dt_fs:g} fs "
            f"({'detected' if dt_detected else 'fallback'})\n"
        )
        report.write(
            "Nominal TS-derived time range: "
            f"{merged['NAMD_NOMINAL_TIME_NS'].iloc[0]:.6f} to "
            f"{merged['NAMD_NOMINAL_TIME_NS'].iloc[-1]:.6f} ns\n"
        )
        report.write(
            "Forced plotting/export time range: "
            f"{merged['TIME_NS'].iloc[0]:.6f} to "
            f"{merged['TIME_NS'].iloc[-1]:.6f} ns\n\n"
        )
        report.write("Duplicate-TS rule:\n")
        report.write("  If TS repeats within the production log, the last printed record is retained.\n\n")
        report.write("Retained records by source:\n")
        for source, count in retained_by_source.items():
            report.write(f"  {source}: {count}\n")
        report.write(
            "\nIMPORTANT: TIME_NS is a deliberate 0-500 ns visualization "
            "rescaling. NAMD_NOMINAL_TIME_NS preserves the TS-derived reference "
            "time using the detected/fallback integration timestep.\n"
        )

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"\nAll outputs saved in:\n{OUTPUT_DIR}")
    print("\nTime-axis validation:")
    print(f"  First retained point: {merged['TIME_NS'].iloc[0]:.6f} ns")
    print(f"  Final retained point: {merged['TIME_NS'].iloc[-1]:.6f} ns")
    print("\nIMPORTANT:")
    print("  TIME_NS is the deliberately forced 0-500 ns plotting scale.")
    print("  NAMD_NOMINAL_TIME_NS preserves the TS-derived reference time.")


if __name__ == "__main__":
    main()
