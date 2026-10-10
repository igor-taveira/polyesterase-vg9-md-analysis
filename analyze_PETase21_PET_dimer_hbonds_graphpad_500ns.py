# -*- coding: utf-8 -*-

"""
Hydrogen-bond analysis for PETase21--PET dimer.

The verified root-level DCD is used:
    ionized.dcd

The original hydrogen-bond criteria are retained:
    donor--acceptor distance cutoff = 3.5 A
    D-H...A angle cutoff           = 150 deg

The script exports:
    - raw H-bond events
    - H-bonds per frame
    - donor/acceptor pair occupancy
    - individual PETase21 residue occupancy
    - PETase21 residue-type occupancy
    - GraphPad-like wide and square figures

For plotting/export, the GLOBAL frame index of the trajectory is linearly
rescaled from the first frame (0 ns) to the final frame (500 ns):

    time_ns = global_frame / (trajectory_frames - 1) * 500

This is a requested plotting convention and does not independently establish
500 ns of physical MD time. Verify actual physical duration from the NAMD
integration timestep and DCD output interval.

The named NAMD energy log:
    energia-petase21-pet-dimer-coo-and-oh-ends.log
is reported for provenance only and is not used mathematically in the H-bond
calculation.

Requires:
    numpy, pandas, matplotlib, MDAnalysis
"""

import os
import re
import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica"
    r"\petase21-PET-02-dimer-coo-and-oh-ends"
)

TOPOLOGY = os.path.join(
    BASE_DIR,
    "ionized.psf",
)

# Single verified root-level trajectory folder.
TRAJECTORY_DIRS = [
    BASE_DIR,
]

ENERGY_LOG_NAMES = [
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
]


def _natural_key(path):
    """Natural filename ordering: part2.dcd precedes part10.dcd."""
    name = os.path.basename(path)
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


def discover_dcd_files(run_dir):
    """
    Discover DCD files without guessing exact filenames.

    DCDs directly inside the supplied PET-dimer folder are preferred. If none
    are found there, the folder is searched recursively.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Trajectory directory not found:\\n{run_dir}"
        )

    direct = sorted(
        [
            os.path.join(run_dir, name)
            for name in os.listdir(run_dir)
            if name.lower().endswith(".dcd")
            and os.path.isfile(os.path.join(run_dir, name))
        ],
        key=_natural_key,
    )

    if direct:
        return direct

    recursive = []

    for root, dirs, files in os.walk(run_dir):

        # Avoid accidentally ingesting generated analysis files on reruns.
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.upper().startswith("HBONDS_")
        ]

        for name in files:
            if name.lower().endswith(".dcd"):
                recursive.append(
                    os.path.join(root, name)
                )

    recursive = sorted(
        recursive,
        key=_natural_key,
    )

    if not recursive:
        raise FileNotFoundError(
            "No DCD trajectory was found in or below:\\n"
            f"{run_dir}"
        )

    return recursive


def discover_energy_log(filename):
    """Locate a named NAMD energy log for provenance reporting only."""
    direct_candidates = [
        os.path.join(BASE_DIR, filename),
    ] + [
        os.path.join(run_dir, filename)
        for run_dir in TRAJECTORY_DIRS
    ]

    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            return candidate

    recursive = []

    for root, _, files in os.walk(BASE_DIR):
        if filename in files:
            recursive.append(
                os.path.join(root, filename)
            )

    recursive = sorted(
        recursive,
        key=_natural_key,
    )

    if recursive:
        return recursive[0]

    return f"NOT FOUND: {filename}"


TRAJECTORIES = []

for trajectory_dir in TRAJECTORY_DIRS:
    TRAJECTORIES.extend(
        discover_dcd_files(
            trajectory_dir
        )
    )


ENERGY_LOGS = [
    discover_energy_log(
        filename
    )
    for filename in ENERGY_LOG_NAMES
]


# ============================================================
# OUTPUT SUBFOLDER
# ============================================================

OUTDIR = os.path.join(
    BASE_DIR,
    "HBONDS_PETase21_PET_dimer_500ns",
)

os.makedirs(
    OUTDIR,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

EVENTS_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_all_events.csv",
)

PER_FRAME_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_per_frame.csv",
)

PAIR_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_pair_occupancy.csv",
)

RESIDUE_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_residue_occupancy.csv",
)

RESNAME_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_residue_type_occupancy.csv",
)

SUMMARY_OUT = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_summary.txt",
)

COUNT_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_per_frame.png",
)

PAIR_OCC_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_top10_pair_occupancy.png",
)

RESIDUE_OCC_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_top10_residue_occupancy.png",
)

RESNAME_OCC_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_residue_type_occupancy.png",
)

COUNT_SQUARE_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_per_frame_square.png",
)

PAIR_OCC_SQUARE_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_top10_pair_occupancy_square.png",
)

RESIDUE_OCC_SQUARE_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_top10_residue_occupancy_square.png",
)

RESNAME_OCC_SQUARE_PNG = os.path.join(
    OUTDIR,
    "hbonds_PETase21_PET_dimer_residue_type_occupancy_square.png",
)


# ============================================================
# SELECTIONS
# ============================================================

# Explicit selections consistent with the PETase21–PET dimer topology used in
# the companion analyses:
#   receptor PETase21 = segid AP1, protein
#   ligand PET dimer = segid L, resname LIG
LIG_SEL = "segid L and resname LIG"
RECEPTOR_SEL = "segid AP1 and protein"

# Verified for PETase21 + PET dimer.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46

# Combined selection used only for guessing H-bond-capable atoms.
LIG_RECEPTOR_SEL = (
    f"({LIG_SEL}) or ({RECEPTOR_SEL})"
)


# ============================================================
# H-BOND SETTINGS
# ============================================================

HBOND_DIST_ANGSTROM = 3.5
HBOND_ANGLE_DEG = 150.0

# Analyze every frame.
# Keeping STEP = 1 guarantees that the first and final physical frames are
# analyzed and therefore the exported timeline spans exactly 0-500 ns.
STEP = 1

# Number of H-bonds/residues shown in occupancy figures.
TOP_N = 10


# ============================================================
# EXPLICIT TRAJECTORY TIME SCALE
# ============================================================

TOTAL_SIMULATION_TIME_NS = 500.0

TIME_TICKS_NS = np.array(
    [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def atom_label(atom):
    """Human-readable atom label."""
    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}:"
        f"{atom.name}"
    )


def residue_label(atom):
    """Human-readable residue label."""
    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}"
    )


def calculate_times(universe, step):
    """
    Return sampled GLOBAL frame numbers and a forced 0-500 ns timeline.

    Mapping:
        time_ns =
            global_frame / (trajectory_frames - 1)
            * TOTAL_SIMULATION_TIME_NS

    With STEP=1:
        first frame = 0 ns
        final frame = exactly 500 ns
    """

    n_frames = len(
        universe.trajectory
    )

    if n_frames < 2:
        raise RuntimeError(
            "At least two combined trajectory frames are required "
            "to construct the 0-500 ns time axis."
        )

    frames = list(
        range(
            0,
            n_frames,
            step,
        )
    )

    # For the default STEP=1 this is already true.
    # If STEP is changed later, explicitly include the physical final frame in
    # the exported time table. The H-bond analysis itself is intended to be
    # run with STEP=1 for exact correspondence with this final-frame inclusion.
    if frames[-1] != n_frames - 1:
        frames.append(
            n_frames - 1
        )

    denominator = float(
        n_frames - 1
    )

    times_ns = [
        (
            float(frame)
            / denominator
            * TOTAL_SIMULATION_TIME_NS
        )
        for frame in frames
    ]

    return frames, times_ns


def frame_to_time_ns(
    frame,
    n_frames,
):
    """Convert a global chained-trajectory frame to the forced 0-500 ns scale."""

    return (
        float(frame)
        / float(n_frames - 1)
        * TOTAL_SIMULATION_TIME_NS
    )


def empty_outputs(
    frames,
    times_ns,
):
    """Create outputs even if zero H-bonds are found."""

    per_frame = pd.DataFrame({
        "global_frame": frames,
        "time_ns": times_ns,
        "hbonds_count": 0,
        "has_hbond": 0,
    })

    per_frame.to_csv(
        PER_FRAME_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(columns=[
        "global_frame", "donor_idx", "hydrogen_idx", "acceptor_idx",
        "distance_A", "angle_deg", "time_ns", "donor_atom",
        "hydrogen_atom", "acceptor_atom", "direction",
        "receptor_residue", "receptor_resname",
    ]).to_csv(
        EVENTS_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(columns=["donor_idx", "acceptor_idx", "donor_atom", "acceptor_atom", "direction", "receptor_residue", "receptor_resname", "frames_with_hbond", "distance_mean_A", "distance_sd_A", "angle_mean_deg", "angle_sd_deg", "sampled_frames", "occupancy_fraction", "occupancy_percent", "donor_short", "acceptor_short"]).to_csv(
        PAIR_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(columns=["receptor_residue", "receptor_resname", "frames_with_hbond", "sampled_frames", "occupancy_fraction", "occupancy_percent"]).to_csv(
        RESIDUE_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(columns=["receptor_resname", "frames_with_hbond", "sampled_frames", "occupancy_fraction", "occupancy_percent"]).to_csv(
        RESNAME_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8",
    ) as summary_file:

        summary_file.write(
            "No PET dimer–PETase21 hydrogen bonds detected.\n"
        )

        summary_file.write(
            f"Rescaled plotting timeline (not physical MD duration): "
            f"0-{TOTAL_SIMULATION_TIME_NS:g} ns\n"
        )

    save_count_plot(
        per_frame
    )


# ============================================================
# GRAPHPAD PRISM-LIKE PLOTTING STYLE
# ============================================================

def set_plot_style():
    """Apply the original GraphPad Prism-like typography/export style."""

    plt.rcParams.update({
        "font.family": [
            "Josefin Sans",
            "Arial",
            "DejaVu Sans",
        ],
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
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
    })


GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"

# Exact original color scheme retained.
GRAPH_COLORS = [
    "#58094F",
    "#841E62",
    "#F3AA20",
    "#2A445E",
    "#346B6D",
]


def style_graphpad_axis(
    axis,
    show_grid=True,
):
    """Gray plot field, black frame, mirrored inward ticks and white grid."""

    axis.set_facecolor(
        GRAPH_BACKGROUND
    )
    axis.set_axisbelow(
        True
    )

    for spine in axis.spines.values():
        spine.set_visible(
            True
        )
        spine.set_color(
            GRAPH_FRAME_COLOR
        )
        spine.set_linewidth(
            1.5
        )

    axis.xaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    axis.yaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.5,
        color=GRAPH_FRAME_COLOR,
        labelcolor=GRAPH_FRAME_COLOR,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3.5,
        width=1.1,
        color=GRAPH_FRAME_COLOR,
    )

    if show_grid:
        axis.grid(
            visible=True,
            which="major",
            axis="both",
            color=GRAPH_GRID_COLOR,
            linestyle=(0, (4, 4)),
            linewidth=1.35,
            alpha=1.0,
        )

        axis.grid(
            visible=False,
            which="minor",
            axis="both",
        )

    for tick_label in (
        list(axis.get_xticklabels())
        + list(axis.get_yticklabels())
    ):
        tick_label.set_fontweight(
            "bold"
        )


def set_plot_area_size(
    figure,
    axis,
    square=False,
):
    """
    Preserve exact original plotting areas:
        wide   = 7.62 x 5.08 cm
        square = 5.08 x 5.08 cm
    """

    width = (
        5.08 if square else 7.62
    ) / 2.54

    height = 5.08 / 2.54

    figure.canvas.draw()

    old_size = (
        figure.get_size_inches()
    )

    boxes = {
        item:
            item.get_position().get_points()
            * old_size
        for item in figure.axes
    }

    left, right = boxes[
        axis
    ][:, 0]

    rows = sorted({
        tuple(
            np.round(
                box[:, 1],
                9,
            )
        )
        for box in boxes.values()
    })

    def resize_x(value):
        return (
            value
            + (
                width
                - (right - left)
            )
            * np.clip(
                (value - left)
                / (right - left),
                0.0,
                1.0,
            )
        )

    def resize_y(value):
        return (
            value
            + sum(
                (
                    height
                    - (top - bottom)
                )
                * np.clip(
                    (value - bottom)
                    / (top - bottom),
                    0.0,
                    1.0,
                )
                for bottom, top in rows
            )
        )

    new_width = resize_x(
        old_size[0]
    )

    new_height = resize_y(
        old_size[1]
    )

    figure.set_size_inches(
        new_width,
        new_height,
        forward=True,
    )

    for item, box in boxes.items():

        x0, x1 = (
            resize_x(value)
            for value in box[:, 0]
        )

        y0, y1 = (
            resize_y(value)
            for value in box[:, 1]
        )

        item.set_position([
            x0 / new_width,
            y0 / new_height,
            (x1 - x0) / new_width,
            (y1 - y0) / new_height,
        ])


def _save_count_plot(
    per_frame,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (
        (5.4, 5.4)
        if square
        else (6.6, 4.2)
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    style_graphpad_axis(
        axis
    )

    if square:
        axis.set_box_aspect(
            1
        )

    axis.plot(
        per_frame["time_ns"],
        per_frame["hbonds_count"],
        linewidth=1.35,
        color=GRAPH_COLORS[3],
    )

    axis.set_xlabel(
        "Time (ns; rescaled)"
    )

    axis.set_xlim(
        0.0,
        TOTAL_SIMULATION_TIME_NS,
    )

    axis.set_xticks(
        TIME_TICKS_NS
    )

    axis.set_ylabel(
        "H-bonds (PETase21 ↔ PET dimer)"
    )

    axis.set_title(
        "PETase21–PET dimer hydrogen bonds"
    )

    figure.tight_layout()

    set_plot_area_size(
        figure,
        axis,
        square=square,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_count_plot(
    per_frame,
):
    _save_count_plot(
        per_frame,
        COUNT_PNG,
        square=False,
    )

    _save_count_plot(
        per_frame,
        COUNT_SQUARE_PNG,
        square=True,
    )


def _save_horizontal_bar(
    labels,
    values,
    title,
    output_path,
    color,
    square=False,
):
    set_plot_style()

    figure_size = (
        (5.6, 5.6)
        if square
        else (7.0, 4.8)
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    style_graphpad_axis(
        axis
    )

    if square:
        axis.set_box_aspect(
            1
        )

    y = np.arange(
        len(values)
    )

    axis.barh(
        y,
        values,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        labels
    )

    axis.set_xlabel(
        "Occupancy (%)"
    )

    axis.set_title(
        title
    )

    axis.set_xlim(
        0,
        max(
            100.0,
            float(
                np.max(values)
            )
            * 1.10,
        ),
    )

    axis.invert_yaxis()

    figure.tight_layout()

    set_plot_area_size(
        figure,
        axis,
        square=square,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_top_pair_plot(
    occupancy,
):
    """Top-N donor-acceptor H-bond occupancies."""

    if occupancy.empty:
        return

    top = occupancy.head(
        TOP_N
    ).copy()

    labels = (
        top["donor_short"]
        + " → "
        + top["acceptor_short"]
    ).to_numpy()

    values = top[
        "occupancy_percent"
    ].to_numpy()

    title = (
        "Top PETase21–PET dimer H-bond occupancies"
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        PAIR_OCC_PNG,
        GRAPH_COLORS[1],
        False,
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        PAIR_OCC_SQUARE_PNG,
        GRAPH_COLORS[1],
        True,
    )


def save_top_residue_plot(
    residue_occ,
):
    """Top-N PETase21 residues interacting with PET dimer by H-bond."""

    if residue_occ.empty:
        return

    top = residue_occ.head(
        TOP_N
    ).copy()

    labels = top[
        "receptor_residue"
    ].to_numpy()

    values = top[
        "occupancy_percent"
    ].to_numpy()

    title = (
        "Top PETase21 residues forming H-bonds with PET dimer"
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        RESIDUE_OCC_PNG,
        GRAPH_COLORS[4],
        False,
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        RESIDUE_OCC_SQUARE_PNG,
        GRAPH_COLORS[4],
        True,
    )


def save_residue_type_plot(
    residue_type_occ,
):
    """Occupancy by PETase21 amino-acid/residue type."""

    if residue_type_occ.empty:
        return

    labels = residue_type_occ[
        "receptor_resname"
    ].to_numpy()

    values = residue_type_occ[
        "occupancy_percent"
    ].to_numpy()

    title = (
        "H-bond occupancy by PETase21 residue type"
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        RESNAME_OCC_PNG,
        GRAPH_COLORS[2],
        False,
    )

    _save_horizontal_bar(
        labels,
        values,
        title,
        RESNAME_OCC_SQUARE_PNG,
        GRAPH_COLORS[2],
        True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "PET dimer <-> PETase21 HYDROGEN-BOND ANALYSIS"
    )

    print(
        f"{len(TRAJECTORIES)} selected DCD file(s); rescaled plotting timeline = 0-500 ns"
    )

    print(
        "=" * 78
    )

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if STEP != 1:
        raise ValueError(
            "This adapted script is intentionally configured with STEP=1 "
            "so that the physical final frame is included and corresponds "
            "exactly to 500 ns. Keep STEP = 1 for this analysis."
        )

    if not os.path.isfile(
        TOPOLOGY
    ):
        raise FileNotFoundError(
            f"PSF not found:\n{TOPOLOGY}"
        )

    for trajectory in TRAJECTORIES:

        if not os.path.isfile(
            trajectory
        ):
            raise FileNotFoundError(
                f"DCD not found:\n{trajectory}"
            )

    print(
        "\n[INFO] Topology:"
    )

    print(
        TOPOLOGY
    )

    print(
        "\n[INFO] Trajectory folder:"
    )

    for trajectory_dir in TRAJECTORY_DIRS:
        print(
            f"  {trajectory_dir}"
        )

    print(
        "\n[INFO] Discovered trajectory files (chronological order):"
    )

    for index, trajectory in enumerate(
        TRAJECTORIES,
        start=1,
    ):
        print(
            f"  DCD {index}: {trajectory}"
        )

    print(
        "\n[INFO] Energy logs "
        "(provenance only; not used for H-bond calculations):"
    )

    for energy_log in ENERGY_LOGS:
        print(
            f"  {energy_log}"
        )

    print(
        "\n[INFO] Output directory:"
    )

    print(
        OUTDIR
    )

    # ========================================================
    # LOAD SYSTEM
    # ========================================================

    print(
        "\n[INFO] Loading the discovered root-level DCD trajectory..."
    )

    u = mda.Universe(
        TOPOLOGY,
        *TRAJECTORIES,
    )

    n_total_frames = len(
        u.trajectory
    )

    print(
        f"[INFO] Atoms: "
        f"{u.atoms.n_atoms:,}"
    )

    print(
        f"[INFO] Combined trajectory frames: "
        f"{n_total_frames:,}"
    )

    print(
        "[INFO] Global frame numbering follows the selected root-level DCD."
    )

    if n_total_frames < 2:
        raise RuntimeError(
            "At least two combined trajectory frames are required."
        )

    # ========================================================
    # CHECK SELECTIONS
    # ========================================================

    ligand = u.select_atoms(
        LIG_SEL
    )

    receptor = u.select_atoms(
        RECEPTOR_SEL
    )

    print(
        f"\n[INFO] PET dimer atoms: "
        f"{ligand.n_atoms:,}"
    )

    print(
        f"[INFO] PETase21 atoms: "
        f"{receptor.n_atoms:,}"
    )

    print(
        f"[INFO] PETase21 residues: "
        f"{receptor.n_residues:,}"
    )

    if ligand.n_atoms == 0:
        raise RuntimeError(
            "No atoms found for PET dimer with selection: "
            f"{LIG_SEL}"
        )

    if ligand.n_atoms and ligand.n_residues != EXPECTED_LIGAND_RESIDUES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_LIGAND_RESIDUES} PET-dimer LIG residue, "
            f"but found {ligand.n_residues}. Refine LIG_SEL."
        )

    if ligand.n_atoms != EXPECTED_LIGAND_ATOMS:
        raise RuntimeError(
            f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET-dimer LIG residue, "
            f"but found {ligand.n_atoms}. Check ionized.psf / LIG_SEL."
        )

    if receptor.n_atoms == 0:
        raise RuntimeError(
            "No atoms found for PETase21 with selection: "
            f"{RECEPTOR_SEL}"
        )

    if receptor.n_residues != EXPECTED_RECEPTOR_RESIDUES:
        raise RuntimeError(
            f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues, "
            f"but found {receptor.n_residues}. Check ionized.psf."
        )

    overlap = np.intersect1d(
        ligand.indices,
        receptor.indices,
    )

    if overlap.size > 0:
        raise RuntimeError(
            "Ligand and PETase21 selections overlap. "
            "Check LIG_SEL and RECEPTOR_SEL."
        )

    print(
        "\n[INFO] PETase21 residue names detected:"
    )

    detected_resnames = sorted(
        set(
            receptor.resnames
        )
    )

    print(
        ", ".join(
            detected_resnames
        )
    )

    # ========================================================
    # GET FRAME TIMES
    # ========================================================

    frames_sampled, times_ns = (
        calculate_times(
            u,
            STEP,
        )
    )

    sampled_frames = len(
        frames_sampled
    )

    print(
        f"\n[INFO] STEP: {STEP}"
    )

    print(
        f"[INFO] Frames analyzed: "
        f"{sampled_frames:,}"
    )

    print(
        "[INFO] Rescaled plotting time range: "
        f"{times_ns[0]:.6f} to {times_ns[-1]:.6f} ns"
    )

    if not np.isclose(
        times_ns[0],
        0.0,
    ):
        raise RuntimeError(
            "First reconstructed time is not 0 ns."
        )

    if not np.isclose(
        times_ns[-1],
        TOTAL_SIMULATION_TIME_NS,
    ):
        raise RuntimeError(
            "Final reconstructed time is not 500 ns."
        )

    # ========================================================
    # BUILD H-BOND ANALYSIS
    # ========================================================

    print(
        "\n[INFO] Preparing H-bond selections..."
    )

    hba = HydrogenBondAnalysis(
        universe=u,
        between=[
            LIG_SEL,
            RECEPTOR_SEL,
        ],
        d_a_cutoff=
            HBOND_DIST_ANGSTROM,
        d_h_a_angle_cutoff=
            HBOND_ANGLE_DEG,
        update_selections=False,
    )

    # Guess H-bond-capable atoms only from ligand + receptor.
    # The PSF provides charges, masses and bond connectivity.
    hydrogen_sel = (
        hba.guess_hydrogens(
            LIG_RECEPTOR_SEL
        )
    )

    acceptor_sel = (
        hba.guess_acceptors(
            LIG_RECEPTOR_SEL
        )
    )

    # Conservative fallbacks if charge-based guessing returns nothing.
    if not hydrogen_sel:

        hydrogen_sel = (
            f"({LIG_RECEPTOR_SEL}) "
            "and name H*"
        )

    if not acceptor_sel:

        acceptor_sel = (
            f"({LIG_RECEPTOR_SEL}) "
            "and (name O* N* S*)"
        )

    hba.hydrogens_sel = (
        hydrogen_sel
    )

    hba.acceptors_sel = (
        acceptor_sel
    )

    try:

        n_h = u.select_atoms(
            hydrogen_sel
        ).n_atoms

        n_a = u.select_atoms(
            acceptor_sel
        ).n_atoms

        print(
            f"[INFO] Candidate H atoms: "
            f"{n_h:,}"
        )

        print(
            f"[INFO] Candidate acceptors: "
            f"{n_a:,}"
        )

    except Exception:
        pass

    # ========================================================
    # RUN H-BOND ANALYSIS
    # ========================================================

    print(
        "\n[INFO] Running H-bond analysis..."
    )

    print(
        f"[INFO] D-A cutoff: "
        f"{HBOND_DIST_ANGSTROM:.2f} Å"
    )

    print(
        f"[INFO] D-H-A angle cutoff: "
        f"{HBOND_ANGLE_DEG:.1f}°"
    )

    hba.run(
        step=STEP
    )

    hbonds = (
        hba.results.hbonds
    )

    # ========================================================
    # NO H-BONDS
    # ========================================================

    if (
        hbonds is None
        or len(hbonds) == 0
    ):

        print(
            "\n[INFO] No PET dimer–PETase21 "
            "H-bonds detected."
        )

        empty_outputs(
            frames_sampled,
            times_ns,
        )

        return

    # ========================================================
    # CREATE EVENT TABLE
    # ========================================================

    events = pd.DataFrame(
        hbonds[:, :6],
        columns=[
            "global_frame",
            "donor_idx",
            "hydrogen_idx",
            "acceptor_idx",
            "distance_A",
            "angle_deg",
        ],
    )

    for col in [
        "global_frame",
        "donor_idx",
        "hydrogen_idx",
        "acceptor_idx",
    ]:

        events[col] = (
            events[col]
            .astype(int)
        )

    # ========================================================
    # EXTRA SAFETY:
    # retain strictly PET dimer <-> PETase21 events
    # ========================================================

    lig_indices = set(
        ligand.indices.tolist()
    )

    receptor_indices = set(
        receptor.indices.tolist()
    )

    mask = [
        (
            row.donor_idx
            in lig_indices
            and
            row.acceptor_idx
            in receptor_indices
        )
        or
        (
            row.donor_idx
            in receptor_indices
            and
            row.acceptor_idx
            in lig_indices
        )
        for row in events.itertuples()
    ]

    events = (
        events.loc[mask]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if events.empty:

        print(
            "\n[INFO] H-bonds existed in the raw analysis, "
            "but no strict PET dimer–PETase21 events remained."
        )

        empty_outputs(
            frames_sampled,
            times_ns,
        )

        return

    # ========================================================
    # TIME FOR EACH EVENT
    # ========================================================

    events["time_ns"] = [
        frame_to_time_ns(
            frame,
            n_total_frames,
        )
        for frame in events[
            "global_frame"
        ]
    ]

    # ========================================================
    # HUMAN-READABLE ATOM LABELS
    # ========================================================

    events[
        "donor_atom"
    ] = [
        atom_label(
            u.atoms[i]
        )
        for i in events[
            "donor_idx"
        ]
    ]

    events[
        "hydrogen_atom"
    ] = [
        atom_label(
            u.atoms[i]
        )
        for i in events[
            "hydrogen_idx"
        ]
    ]

    events[
        "acceptor_atom"
    ] = [
        atom_label(
            u.atoms[i]
        )
        for i in events[
            "acceptor_idx"
        ]
    ]

    # ========================================================
    # DIRECTION + PETase21 RESIDUE
    # ========================================================

    directions = []
    receptor_residues = []
    receptor_resnames = []

    for row in events.itertuples():

        donor = u.atoms[
            row.donor_idx
        ]

        acceptor = u.atoms[
            row.acceptor_idx
        ]

        if donor.index in lig_indices:

            directions.append(
                "PET dimer donor -> PETase21 acceptor"
            )

            receptor_atom = acceptor

        else:

            directions.append(
                "PETase21 donor -> PET dimer acceptor"
            )

            receptor_atom = donor

        receptor_residues.append(
            residue_label(
                receptor_atom
            )
        )

        receptor_resnames.append(
            receptor_atom.resname
        )

    events[
        "direction"
    ] = directions

    events[
        "receptor_residue"
    ] = receptor_residues

    events[
        "receptor_resname"
    ] = receptor_resnames

    # ========================================================
    # SAVE RAW EVENTS
    # ========================================================

    events.to_csv(
        EVENTS_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[OK] Raw H-bond events:\n"
        f"{EVENTS_OUT}"
    )

    # ========================================================
    # NUMBER OF H-BONDS PER FRAME
    #
    # Unique donor-H-acceptor triplet per frame.
    # ========================================================

    unique_triplets = (
        events[
            [
                "global_frame",
                "donor_idx",
                "hydrogen_idx",
                "acceptor_idx",
            ]
        ]
        .drop_duplicates()
    )

    counts_by_frame = (
        unique_triplets
        .groupby(
            "global_frame"
        )
        .size()
    )

    per_frame = pd.DataFrame({
        "global_frame":
            frames_sampled,
        "time_ns":
            times_ns,
    })

    per_frame[
        "hbonds_count"
    ] = (
        per_frame[
            "global_frame"
        ]
        .map(
            counts_by_frame
        )
        .fillna(0)
        .astype(int)
    )

    per_frame[
        "has_hbond"
    ] = (
        per_frame[
            "hbonds_count"
        ]
        > 0
    ).astype(int)

    per_frame.to_csv(
        PER_FRAME_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    save_count_plot(
        per_frame
    )

    # ========================================================
    # OVERALL OCCUPANCY
    # ========================================================

    overall_occupancy = (
        100.0
        * per_frame[
            "has_hbond"
        ].mean()
    )

    mean_hbonds = (
        per_frame[
            "hbonds_count"
        ].mean()
    )

    sd_hbonds = (
        per_frame[
            "hbonds_count"
        ].std()
    )

    median_hbonds = (
        per_frame[
            "hbonds_count"
        ].median()
    )

    max_hbonds = (
        per_frame[
            "hbonds_count"
        ].max()
    )

    # ========================================================
    # PAIR OCCUPANCY
    #
    # Unique donor <-> acceptor heavy-atom pair.
    # ========================================================

    pair_events = (
        events
        .drop_duplicates(
            subset=[
                "global_frame",
                "donor_idx",
                "acceptor_idx",
            ]
        )
    )

    pair_group = (
        pair_events.groupby(
            [
                "donor_idx",
                "acceptor_idx",
            ]
        )
    )

    pair_occ = (
        pair_group
        .agg(
            donor_atom=(
                "donor_atom",
                "first",
            ),
            acceptor_atom=(
                "acceptor_atom",
                "first",
            ),
            direction=(
                "direction",
                "first",
            ),
            receptor_residue=(
                "receptor_residue",
                "first",
            ),
            receptor_resname=(
                "receptor_resname",
                "first",
            ),
            frames_with_hbond=(
                "global_frame",
                "nunique",
            ),
            distance_mean_A=(
                "distance_A",
                "mean",
            ),
            distance_sd_A=(
                "distance_A",
                "std",
            ),
            angle_mean_deg=(
                "angle_deg",
                "mean",
            ),
            angle_sd_deg=(
                "angle_deg",
                "std",
            ),
        )
        .reset_index()
    )

    pair_occ[
        "sampled_frames"
    ] = sampled_frames

    pair_occ[
        "occupancy_fraction"
    ] = (
        pair_occ[
            "frames_with_hbond"
        ]
        / sampled_frames
    )

    pair_occ[
        "occupancy_percent"
    ] = (
        100.0
        * pair_occ[
            "occupancy_fraction"
        ]
    )

    pair_occ[
        "donor_short"
    ] = (
        pair_occ[
            "donor_atom"
        ]
        .apply(
            lambda x:
                ":".join(
                    x.split(":")[1:]
                )
        )
    )

    pair_occ[
        "acceptor_short"
    ] = (
        pair_occ[
            "acceptor_atom"
        ]
        .apply(
            lambda x:
                ":".join(
                    x.split(":")[1:]
                )
        )
    )

    pair_occ = (
        pair_occ
        .sort_values(
            [
                "occupancy_percent",
                "distance_mean_A",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    pair_occ.to_csv(
        PAIR_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    save_top_pair_plot(
        pair_occ
    )

    # ========================================================
    # OCCUPANCY BY INDIVIDUAL PETase21 RESIDUE
    #
    # At least one ligand H-bond with that residue in a frame.
    # ========================================================

    residue_frame = (
        events[
            [
                "global_frame",
                "receptor_residue",
                "receptor_resname",
            ]
        ]
        .drop_duplicates()
    )

    residue_occ = (
        residue_frame
        .groupby(
            [
                "receptor_residue",
                "receptor_resname",
            ]
        )
        .agg(
            frames_with_hbond=(
                "global_frame",
                "nunique",
            )
        )
        .reset_index()
    )

    residue_occ[
        "sampled_frames"
    ] = sampled_frames

    residue_occ[
        "occupancy_fraction"
    ] = (
        residue_occ[
            "frames_with_hbond"
        ]
        / sampled_frames
    )

    residue_occ[
        "occupancy_percent"
    ] = (
        100.0
        * residue_occ[
            "occupancy_fraction"
        ]
    )

    residue_occ = (
        residue_occ
        .sort_values(
            "occupancy_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    residue_occ.to_csv(
        RESIDUE_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    save_top_residue_plot(
        residue_occ
    )

    # ========================================================
    # OCCUPANCY BY PETase21 RESIDUE TYPE
    #
    # Percentage of sampled frames containing >=1 H-bond to ANY
    # PETase21 residue of that amino-acid/residue type.
    # ========================================================

    residue_type_frame = (
        events[
            [
                "global_frame",
                "receptor_resname",
            ]
        ]
        .drop_duplicates()
    )

    residue_type_occ = (
        residue_type_frame
        .groupby(
            "receptor_resname"
        )
        .agg(
            frames_with_hbond=(
                "global_frame",
                "nunique",
            )
        )
        .reset_index()
    )

    residue_type_occ[
        "sampled_frames"
    ] = sampled_frames

    residue_type_occ[
        "occupancy_fraction"
    ] = (
        residue_type_occ[
            "frames_with_hbond"
        ]
        / sampled_frames
    )

    residue_type_occ[
        "occupancy_percent"
    ] = (
        100.0
        * residue_type_occ[
            "occupancy_fraction"
        ]
    )

    residue_type_occ = (
        residue_type_occ
        .sort_values(
            "occupancy_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    residue_type_occ.to_csv(
        RESNAME_OCC_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    save_residue_type_plot(
        residue_type_occ
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    unique_pairs = len(
        pair_occ
    )

    total_events = len(
        events
    )

    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8",
    ) as summary_file:

        summary_file.write(
            "PET dimer <-> PETase21 HYDROGEN-BOND ANALYSIS\n"
        )

        summary_file.write(
            "=" * 70
            + "\n\n"
        )

        summary_file.write(
            f"Topology: {TOPOLOGY}\n\n"
        )

        summary_file.write(
            "Trajectory handling: MDAnalysis chains all discovered DCD files "
            "into one global frame sequence in the listed order.\n\n"
        )

        summary_file.write(
            "Trajectory folder:\n"
        )

        for trajectory_dir in TRAJECTORY_DIRS:
            summary_file.write(
                f"  {trajectory_dir}\n"
            )

        summary_file.write(
            "\nTrajectory file:\n"
        )

        for trajectory in TRAJECTORIES:
            summary_file.write(
                f"  {trajectory}\n"
            )

        summary_file.write(
            "\nEnergy logs "
            "(provenance only; not used in H-bond calculation):\n"
        )

        for energy_log in ENERGY_LOGS:
            summary_file.write(
                f"  {energy_log}\n"
            )

        summary_file.write(
            "\n"
        )

        summary_file.write(
            "Selections:\n"
        )

        summary_file.write(
            f"  Ligand: {LIG_SEL}\n"
        )

        summary_file.write(
            f"  PETase21: {RECEPTOR_SEL}\n\n"
        )

        summary_file.write(
            "Rescaled plotting-time mapping (not verified physical time):\n"
        )

        summary_file.write(
            "  time_ns = global_frame / "
            "(trajectory_frames - 1) * 500\n"
        )

        summary_file.write(
            f"  Range: "
            f"0-{TOTAL_SIMULATION_TIME_NS:g} ns\n\n"
        )

        summary_file.write(
            "NOTICE: The 0-500 ns labels are a frame-index rescaling, not "
            "a verification of physical simulation duration.\n\n"
        )

        summary_file.write(
            f"D-A cutoff: "
            f"{HBOND_DIST_ANGSTROM:.2f} Å\n"
        )

        summary_file.write(
            f"D-H-A angle cutoff: "
            f"{HBOND_ANGLE_DEG:.1f} degrees\n"
        )

        summary_file.write(
            f"STEP: {STEP}\n\n"
        )

        summary_file.write(
            f"Total combined trajectory frames: "
            f"{n_total_frames}\n"
        )

        summary_file.write(
            f"Sampled frames: "
            f"{sampled_frames}\n"
        )

        summary_file.write(
            f"Total detected H-bond events: "
            f"{total_events}\n"
        )

        summary_file.write(
            f"Unique donor-acceptor pairs: "
            f"{unique_pairs}\n\n"
        )

        summary_file.write(
            "H-BONDS PER FRAME\n"
        )

        summary_file.write(
            "-" * 40
            + "\n"
        )

        summary_file.write(
            f"Mean ± SD: "
            f"{mean_hbonds:.4f} "
            f"± {sd_hbonds:.4f}\n"
        )

        summary_file.write(
            f"Median: "
            f"{median_hbonds:.4f}\n"
        )

        summary_file.write(
            f"Maximum: "
            f"{max_hbonds}\n\n"
        )

        summary_file.write(
            "OVERALL OCCUPANCY\n"
        )

        summary_file.write(
            "-" * 40
            + "\n"
        )

        summary_file.write(
            "Frames containing at least one "
            "PET dimer–PETase21 H-bond:\n"
        )

        summary_file.write(
            f"{overall_occupancy:.3f}%\n\n"
        )

        summary_file.write(
            "TOP H-BOND PAIRS\n"
        )

        summary_file.write(
            "-" * 40
            + "\n"
        )

        top_pairs = (
            pair_occ
            .head(TOP_N)
        )

        for _, row in top_pairs.iterrows():

            summary_file.write(
                f"{row['donor_atom']} -> "
                f"{row['acceptor_atom']}\n"
            )

            summary_file.write(
                f"  Direction: "
                f"{row['direction']}\n"
            )

            summary_file.write(
                f"  PETase21 residue: "
                f"{row['receptor_residue']}\n"
            )

            summary_file.write(
                f"  Occupancy: "
                f"{row['occupancy_percent']:.3f}%\n"
            )

            summary_file.write(
                f"  Distance: "
                f"{row['distance_mean_A']:.3f} Å\n"
            )

            summary_file.write(
                f"  Angle: "
                f"{row['angle_mean_deg']:.3f}°\n\n"
            )

    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 78
    )

    print(
        f"\nMean H-bonds/frame: "
        f"{mean_hbonds:.4f} "
        f"± {sd_hbonds:.4f}"
    )

    print(
        f"Median H-bonds/frame: "
        f"{median_hbonds:.4f}"
    )

    print(
        f"Maximum H-bonds/frame: "
        f"{max_hbonds}"
    )

    print(
        f"\nOverall H-bond occupancy: "
        f"{overall_occupancy:.3f}%"
    )

    print(
        f"\nUnique donor-acceptor pairs: "
        f"{unique_pairs}"
    )

    print(
        "\nTop 10 H-bond pairs:"
    )

    display_columns = [
        "donor_atom",
        "acceptor_atom",
        "direction",
        "receptor_residue",
        "occupancy_percent",
        "distance_mean_A",
        "angle_mean_deg",
    ]

    print(
        pair_occ[
            display_columns
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop PETase21 residues:"
    )

    print(
        residue_occ
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    print(
        "\nOccupancy by PETase21 residue type:"
    )

    print(
        residue_type_occ
        .to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nRescaled time-axis validation:"
    )

    print(
        f"  First point: "
        f"{per_frame['time_ns'].iloc[0]:.6f} ns"
    )

    print(
        f"  Final point: "
        f"{per_frame['time_ns'].iloc[-1]:.6f} ns"
    )

    print(
        f"\nAll outputs saved in:\n"
        f"{OUTDIR}"
    )

    print(
        """
Files:

  hbonds_PETase21_PET_dimer_all_events.csv
  hbonds_PETase21_PET_dimer_per_frame.csv
  hbonds_PETase21_PET_dimer_pair_occupancy.csv
  hbonds_PETase21_PET_dimer_residue_occupancy.csv
  hbonds_PETase21_PET_dimer_residue_type_occupancy.csv
  hbonds_PETase21_PET_dimer_summary.txt

  hbonds_PETase21_PET_dimer_per_frame.png
  hbonds_PETase21_PET_dimer_top10_pair_occupancy.png
  hbonds_PETase21_PET_dimer_top10_residue_occupancy.png
  hbonds_PETase21_PET_dimer_residue_type_occupancy.png

Square versions of all plots are also generated.
"""
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
