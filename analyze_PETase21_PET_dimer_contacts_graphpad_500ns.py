# -*- coding: utf-8 -*-
"""
General atom-contact analysis for PETase21–PET dimer.

Contact definition preserved from the supplied workflow:
    every ligand–protein atom pair with minimum-image distance < 4.5 Å.
    Counts represent ATOM PAIRS, not numbers of contacting residues.

The verified root-level trajectory is:
    ionized.dcd

All selected trajectory frames are mapped linearly to 0–500 ns for plotting:
    displayed_time_ns = global_frame / (combined_frames - 1) * 500

IMPORTANT: The 500 ns time axis is a REQUESTED NORMALIZATION, NOT proof
that the trajectory contains 500 ns of physical simulation time.
The DCD time metadata is not used for the displayed axis.

The named energy log:
    energia-petase21-pet-dimer-coo-and-oh-ends.log
is included for provenance reporting only and is NOT used for the contact
calculation itself.

GraphPad-like style and the established dark-blue line color (#2A445E)
are preserved.

Required packages:
    numpy, matplotlib, MDAnalysis
"""

import os
import re

import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import AutoMinorLocator
from MDAnalysis.lib.distances import distance_array


# ============================================================
# WINDOWS PATHS AND DCD INPUTS
# ============================================================

PROJECT_ROOT = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica"
)
BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "petase21-PET-02-dimer-coo-and-oh-ends",
)

PSF = os.path.join(BASE_DIR, "ionized.psf")

TRAJECTORY_DIRS = [
    BASE_DIR,
]

ENERGY_LOG_NAMES = [
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
]


def _natural_key(path):
    """Natural filename ordering: segment2.dcd before segment10.dcd."""
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
            f"Trajectory directory not found:\n{run_dir}"
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
        dirs[:] = [d for d in dirs if not d.upper().startswith("CONTACTS_")]
        for name in files:
            if name.lower().endswith(".dcd"):
                recursive.append(os.path.join(root, name))

    recursive = sorted(recursive, key=_natural_key)

    if not recursive:
        raise FileNotFoundError(
            "No DCD trajectory was found in or below:\n"
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
        for name in files:
            if name == filename:
                recursive.append(os.path.join(root, name))

    recursive = sorted(recursive, key=_natural_key)
    if recursive:
        return recursive[0]

    return f"NOT FOUND: {filename}"


DCD_FILES = []
for trajectory_dir in TRAJECTORY_DIRS:
    DCD_FILES.extend(discover_dcd_files(trajectory_dir))

ENERGY_LOGS = [
    discover_energy_log(filename)
    for filename in ENERGY_LOG_NAMES
]


# ============================================================
# OUTPUT DIRECTORY / FILES
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "CONTACTS_PETase21_PET_dimer_500ns",
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_contacts_500ns.txt",
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_contacts_500ns.png",
)

OUTPUT_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_contacts_500ns_square.png",
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

CONTACT_CUTOFF_A = 4.5
STEP = 1
TOTAL_SIMULATION_TIME_NS = 500.0
TIME_TICKS_NS = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])


# ============================================================
# SYSTEM SELECTIONS
# ============================================================

RECEPTOR_SEL = "segid AP1 and protein"
LIGAND_SEL = "segid L and resname LIG"

# Verified for PETase21 + PET dimer.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46


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

CONTACT_LINE_COLOR = GRAPH_COLORS[3]


def set_plot_style():
    """Apply the GraphPad Prism-like typography/export standard."""
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
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
    })


def style_graphpad_axis(axis, show_grid=True):
    """Gray plotting field, black frame, inward mirrored ticks, white grid."""
    axis.set_facecolor(GRAPH_BACKGROUND)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRAPH_FRAME_COLOR)
        spine.set_linewidth(1.5)

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

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
        axis.grid(visible=False, which="minor", axis="both")

    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")


def set_plot_area_size(figure, axis, square=False):
    """
    Set exact axes plotting area:
        wide   = 7.62 x 5.08 cm
        square = 5.08 x 5.08 cm
    """
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


def save_contact_plot(times_ns, contact_counts, output_path, square=False):
    """Save one GraphPad-like contact-vs-time plot."""
    set_plot_style()

    figure_size = (5.4, 5.4) if square else (6.6, 4.2)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    if square:
        axis.set_box_aspect(1)

    axis.plot(
        times_ns,
        contact_counts,
        color=CONTACT_LINE_COLOR,
        linewidth=1.35,
    )

    axis.set_xlabel("Time (ns; rescaled)")
    axis.set_ylabel("Number of contacts")
    axis.set_xlim(0.0, TOTAL_SIMULATION_TIME_NS)
    axis.set_xticks(TIME_TICKS_NS)
    axis.set_title(
        "PETase21–PET dimer contacts\n"
        f"cutoff = {CONTACT_CUTOFF_A:.1f} Å"
    )

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# VERIFY INPUT FILES
# ============================================================

if STEP < 1:
    raise ValueError("STEP must be at least 1.")

if not os.path.isfile(PSF):
    raise FileNotFoundError(f"PSF file not found:\n{PSF}")

for dcd in DCD_FILES:
    if not os.path.isfile(dcd):
        raise FileNotFoundError(f"DCD file not found:\n{dcd}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD SYSTEM / CONCATENATED TRAJECTORY
# ============================================================

print("=" * 78)
print("PETase21 <-> PET dimer GENERAL CONTACT ANALYSIS")
print(f"{len(DCD_FILES)} discovered DCD(s); normalized DISPLAY timeline = 0–500 ns")
print("NOTE: 500 ns is a frame-normalized DISPLAY axis, not verified physical MD time.")
print("=" * 78)

print("\nLoading topology and concatenated trajectory...")

u = mda.Universe(PSF, *DCD_FILES)
n_total_frames = len(u.trajectory)

if n_total_frames < 2:
    raise RuntimeError(
        "At least two combined trajectory frames are required to construct "
        "the 0-500 ns time axis."
    )

print(f"\nTopology:\n{PSF}")

print("\nTrajectory directories (chronological):")
for run_dir in TRAJECTORY_DIRS:
    print(f"  {run_dir}")

print("\nSequential trajectories:")
for index, dcd in enumerate(DCD_FILES, start=1):
    print(f"  DCD {index}: {dcd}")

print("\nEnergy logs (provenance only; not used for contacts):")
for energy_log in ENERGY_LOGS:
    print(f"  {energy_log}")


# ============================================================
# DEFINE SELECTIONS
# ============================================================

receptor = u.select_atoms(RECEPTOR_SEL)
ligand = u.select_atoms(LIGAND_SEL)


# ============================================================
# VALIDATE SELECTIONS
# ============================================================

if receptor.n_atoms == 0:
    available_segids = sorted(set(u.atoms.segids))
    raise ValueError(
        "No PETase21 receptor atoms were found with "
        f"selection: {RECEPTOR_SEL}\n"
        f"Available segids: {available_segids}"
    )

if ligand.n_atoms == 0:
    available_resnames = sorted(set(u.atoms.resnames))
    available_segids = sorted(set(u.atoms.segids))
    raise ValueError(
        "No PET dimer ligand atoms were found with "
        f"selection: {LIGAND_SEL}\n"
        f"Available segids: {available_segids}\n"
        f"Available resnames: {available_resnames}"
    )

if receptor.n_residues != EXPECTED_RECEPTOR_RESIDUES:
    raise ValueError(
        f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues, but found "
        f"{receptor.n_residues}. Check ionized.psf."
    )

if ligand.n_residues != EXPECTED_LIGAND_RESIDUES:
    raise ValueError(
        f"Expected exactly {EXPECTED_LIGAND_RESIDUES} PET-dimer LIG residue, "
        f"but found {ligand.n_residues}. Inspect the PSF residue names/segment IDs."
    )

if ligand.n_atoms != EXPECTED_LIGAND_ATOMS:
    raise ValueError(
        f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET-dimer LIG residue, "
        f"but found {ligand.n_atoms}. Check ionized.psf / ligand selection."
    )

overlap = np.intersect1d(receptor.indices, ligand.indices)
if overlap.size > 0:
    raise RuntimeError(
        "PETase21 and ligand selections overlap. "
        "Check RECEPTOR_SEL and LIGAND_SEL."
    )

print(f"\nTotal system atoms: {u.atoms.n_atoms:,}")
print(f"PETase21 atoms: {receptor.n_atoms:,}")
print(f"PETase21 residues: {receptor.n_residues:,}")
print(f"PET dimer atoms: {ligand.n_atoms:,}")
print(f"Combined trajectory frames: {n_total_frames:,}")
print(f"Contact cutoff: {CONTACT_CUTOFF_A:.1f} Å")


# ============================================================
# BUILD GLOBAL SAMPLED FRAME LIST
# ============================================================

sampled_frame_indices = list(range(0, n_total_frames, STEP))

if sampled_frame_indices[-1] != n_total_frames - 1:
    sampled_frame_indices.append(n_total_frames - 1)

time_denominator = float(n_total_frames - 1)


# ============================================================
# CALCULATE CONTACTS FRAME BY FRAME
# ============================================================

frames = []
times_ns = []
contact_counts = []

print("\nCalculating PETase21–PET dimer contacts...")

for sampled_index, global_frame in enumerate(sampled_frame_indices):
    ts = u.trajectory[global_frame]

    box = ts.dimensions
    if box is not None and (
        not np.all(np.isfinite(box)) or np.any(np.asarray(box)[:3] <= 0)
    ):
        box = None

    distances = distance_array(
        ligand.positions,
        receptor.positions,
        box=box,
    )

    n_contacts = int(
        np.count_nonzero(distances < CONTACT_CUTOFF_A)
    )

    time_ns = (
        float(global_frame) / time_denominator * TOTAL_SIMULATION_TIME_NS
    )

    frames.append(int(global_frame))
    times_ns.append(float(time_ns))
    contact_counts.append(n_contacts)

    if sampled_index % 100 == 0 or global_frame == n_total_frames - 1:
        print(
            f"Processed global frame {global_frame + 1:,}/{n_total_frames:,} "
            f"({time_ns:.3f} ns)"
        )


# ============================================================
# VALIDATE FORCED TIME SCALE
# ============================================================

if not np.isclose(times_ns[0], 0.0):
    raise RuntimeError("First reconstructed trajectory time is not 0 ns.")

if not np.isclose(times_ns[-1], TOTAL_SIMULATION_TIME_NS):
    raise RuntimeError("Final reconstructed trajectory time is not exactly 500 ns.")

print("\nNormalized display-axis validation (not physical-time validation):")
print(f"  First sampled point: {times_ns[0]:.6f} ns")
print(f"  Final sampled point: {times_ns[-1]:.6f} ns")


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

results = np.column_stack((frames, times_ns, contact_counts))

np.savetxt(
    OUTPUT_TXT,
    results,
    fmt=["%d", "%.6f", "%d"],
    header="Global_Frame Rescaled_time_ns Contacts",
    comments="",
)


# ============================================================
# SAVE FIGURES
# ============================================================

save_contact_plot(
    times_ns,
    contact_counts,
    OUTPUT_PNG,
    square=False,
)

save_contact_plot(
    times_ns,
    contact_counts,
    OUTPUT_SQUARE_PNG,
    square=True,
)


# ============================================================
# FINAL REPORT
# ============================================================

contact_array = np.asarray(contact_counts, dtype=float)

print("\n" + "=" * 78)
print("ANALYSIS COMPLETED")
print("=" * 78)

print(f"\nFrames analyzed: {len(contact_counts):,}")
print(f"Mean contacts/frame: {contact_array.mean():.3f}")

if len(contact_array) > 1:
    print(f"SD contacts/frame: {contact_array.std(ddof=1):.3f}")
else:
    print("SD contacts/frame: n/a")

print(f"Median contacts/frame: {np.median(contact_array):.3f}")
print(f"Minimum contacts/frame: {int(contact_array.min())}")
print(f"Maximum contacts/frame: {int(contact_array.max())}")

print(f"\nResults saved in:\n{OUTPUT_DIR}")
print(f"\nNumerical results:\n{OUTPUT_TXT}")
print(f"\nWide figure:\n{OUTPUT_PNG}")
print(f"\nSquare figure:\n{OUTPUT_SQUARE_PNG}")
