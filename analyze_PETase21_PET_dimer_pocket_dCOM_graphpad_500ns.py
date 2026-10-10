# -*- coding: utf-8 -*-

"""
PBC-aware center-of-mass distance analysis between PET dimer and the
PETase21 binding pocket.

Metric
------
For each frame:

    dCOM-COM = | COM(PET dimer) - COM(PETase21 binding pocket) |

Both centers of mass are mass-weighted.

Binding-pocket definition
-------------------------
The binding pocket is defined ONCE from the first trajectory frame:
all PETase21 residues containing at least one atom within 5.0 Å of PET dimer.

Selections:
    PETase21 = segid AP1 and protein
    PET dimer = segid L and resname LIG

Periodic boundaries
-------------------
PETase21 and PET-dimer fragments are reconstructed with PSF bond connectivity
when valid periodic-cell data are available. The ligand-pocket COM vector is
then converted to its minimum-image periodic vector before its magnitude is
calculated. If no valid box is available in a frame, the raw Cartesian COM
vector is retained and a warning is emitted.

Trajectory handling
-------------------
The verified root-level trajectory is:
    ionized.dcd

Forced time scale
-----------------
The global frame sequence is mapped to:

    time_ns = global_frame / (total_frames - 1) * 500

Thus the first frame is plotted at 0 ns and the final frame at 500 ns.
This is a requested display/export rescaling and does not establish the true
physical simulated duration.

Energy log
----------
The named NAMD energy log:
    energia-petase21-pet-dimer-coo-and-oh-ends.log
is located and recorded for provenance only; it is not used mathematically in
the dCOM calculation.

GraphPad-like format
--------------------
- gray plotting field
- complete black frame
- mirrored inward major/minor ticks
- white dashed major grid
- bold typography
- established five-color palette
- magenta dCOM trace (#841E62)
- exact axes plotting areas:
    wide   = 7.62 x 5.08 cm
    square = 5.08 x 5.08 cm

Required packages:
    numpy, pandas, matplotlib, MDAnalysis
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array, minimize_vectors

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

PSF_FILE = os.path.join(
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

    Files directly inside the supplied PET-dimer folder are preferred. If none
    are found there, the directory is searched recursively.
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

        # Avoid analysis output folders if the script is rerun.
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.lower().startswith("dcom_")
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
            "No DCD trajectory was found in or below:\n"
            f"{run_dir}"
        )

    return recursive


def discover_energy_log(filename):
    """Locate one named NAMD energy log for provenance reporting only."""
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


DCD_FILES = []

for trajectory_dir in TRAJECTORY_DIRS:
    DCD_FILES.extend(
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
# OUTPUTS
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dCOM_PETase21_PET_dimer_pocket_500ns",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_pocket_dCOM_500ns.csv",
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_pocket_dCOM_500ns.txt",
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_pocket_dCOM_summary.csv",
)

OUTPUT_POCKET = os.path.join(
    OUTPUT_DIR,
    "PETase21_binding_pocket_residues_5A.txt",
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_pocket_dCOM_500ns.png",
)

OUTPUT_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_pocket_dCOM_500ns_square.png",
)


# ============================================================
# SETTINGS
# ============================================================

RECEPTOR_SELECTION = "segid AP1 and protein"
LIGAND_SELECTION = "segid L and resname LIG"

# Verified for PETase21 + PET dimer.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46

POCKET_CUTOFF_A = 5.0
STEP = 1

TOTAL_SIMULATION_TIME_NS = 500.0
TIME_TICKS_NS = np.array(
    [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
)


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

DCOM_COLOR = GRAPH_COLORS[1]


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
    width = (5.08 if square else 7.62) / 2.54
    height = 5.08 / 2.54

    figure.canvas.draw()
    old_size = figure.get_size_inches()

    boxes = {
        item: item.get_position().get_points() * old_size
        for item in figure.axes
    }

    left, right = boxes[axis][:, 0]
    rows = sorted({
        tuple(np.round(box[:, 1], 9))
        for box in boxes.values()
    })

    def resize_x(value):
        return value + (width - (right - left)) * np.clip(
            (value - left) / (right - left),
            0.0,
            1.0,
        )

    def resize_y(value):
        return value + sum(
            (height - (top - bottom)) * np.clip(
                (value - bottom) / (top - bottom),
                0.0,
                1.0,
            )
            for bottom, top in rows
        )

    new_width = resize_x(old_size[0])
    new_height = resize_y(old_size[1])

    figure.set_size_inches(new_width, new_height, forward=True)

    for item, box in boxes.items():
        x0, x1 = (resize_x(value) for value in box[:, 0])
        y0, y1 = (resize_y(value) for value in box[:, 1])

        item.set_position([
            x0 / new_width,
            y0 / new_height,
            (x1 - x0) / new_width,
            (y1 - y0) / new_height,
        ])


# ============================================================
# PBC HELPERS
# ============================================================

def valid_periodic_box(dimensions):
    if dimensions is None:
        return False

    dimensions = np.asarray(dimensions, dtype=float)

    if dimensions.size < 6:
        return False

    if not np.all(np.isfinite(dimensions)):
        return False

    return bool(np.all(dimensions[:3] > 0.0))


_UNWRAP_WARNING_FLAGS = {
    "receptor": False,
    "ligand": False,
}


def unwrap_groups(
    receptor,
    ligand,
    *,
    have_valid_box,
):
    """
    Reconstruct bonded fragments when valid periodic cell data are available.

    If the current DCD frame has no usable box, leave coordinates unchanged.
    """
    if not have_valid_box:
        return

    try:
        receptor.unwrap(
            compound="fragments"
        )
    except Exception as exc:

        if not _UNWRAP_WARNING_FLAGS[
            "receptor"
        ]:
            warnings.warn(
                "Could not unwrap PETase21 fragments; "
                f"stored coordinates will be used. Details: {exc}"
            )

            _UNWRAP_WARNING_FLAGS[
                "receptor"
            ] = True

    try:
        ligand.unwrap(
            compound="fragments"
        )
    except Exception as exc:

        if not _UNWRAP_WARNING_FLAGS[
            "ligand"
        ]:
            warnings.warn(
                "Could not unwrap PET dimer; "
                f"stored coordinates will be used. Details: {exc}"
            )

            _UNWRAP_WARNING_FLAGS[
                "ligand"
            ] = True


def minimum_image_vector(vector, dimensions):
    vector = np.asarray(vector, dtype=float)

    if not valid_periodic_box(dimensions):
        return vector

    return minimize_vectors(
        vector.reshape(1, 3),
        dimensions,
    )[0]


# ============================================================
# POCKET DEFINITION
# ============================================================

def define_binding_pocket(universe, receptor, ligand, cutoff):
    universe.trajectory[0]

    first_dimensions = (
        universe.trajectory.ts.dimensions
    )

    have_valid_box = (
        valid_periodic_box(
            first_dimensions
        )
    )

    unwrap_groups(
        receptor,
        ligand,
        have_valid_box=have_valid_box,
    )

    box = (
        first_dimensions
        if have_valid_box
        else None
    )

    distances = distance_array(
        receptor.positions,
        ligand.positions,
        box=box,
    )

    contacting_receptor_atom_local_indices = np.where(
        np.any(
            distances <= cutoff,
            axis=1,
        )
    )[0]

    if contacting_receptor_atom_local_indices.size == 0:
        raise RuntimeError(
            "No PETase21 atoms were found within "
            f"{cutoff:.1f} Å of PET dimer in the first frame."
        )

    contacting_atoms = receptor[
        contacting_receptor_atom_local_indices
    ]

    pocket_residues = contacting_atoms.residues
    pocket_atoms = pocket_residues.atoms

    return pocket_residues, pocket_atoms


# ============================================================
# PLOTTING
# ============================================================

def save_dcom_plot(dataframe, output_path, square=False):
    set_plot_style()

    figure_size = (5.4, 5.4) if square else (6.6, 4.2)
    figure, axis = plt.subplots(figsize=figure_size)

    style_graphpad_axis(axis)

    if square:
        axis.set_box_aspect(1)

    axis.plot(
        dataframe["time_ns"],
        dataframe["ligand_pocket_dCOM_A"],
        color=DCOM_COLOR,
        linewidth=1.35,
    )

    axis.set_xlabel("Time (ns; rescaled)")
    axis.set_ylabel("Ligand–pocket dCOM (Å)")
    axis.set_xlim(0.0, TOTAL_SIMULATION_TIME_NS)
    axis.set_xticks(TIME_TICKS_NS)
    axis.set_title("PETase21–PET dimer pocket dCOM")

    figure.tight_layout()
    set_plot_area_size(
        figure,
        axis,
        square=square,
    )

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("PETase21 <-> PET dimer BINDING-POCKET dCOM-COM ANALYSIS")
    print(f"{len(DCD_FILES)} discovered DCD file(s); PBC-aware; rescaled combined timeline = 0-500 ns")
    print("=" * 78)

    if STEP < 1:
        raise ValueError("STEP must be >= 1.")

    if not os.path.isfile(PSF_FILE):
        raise FileNotFoundError(
            f"PSF file not found:\n{PSF_FILE}"
        )

    for dcd_file in DCD_FILES:
        if not os.path.isfile(dcd_file):
            raise FileNotFoundError(
                f"DCD file not found:\n{dcd_file}"
            )

    print("\nLoading topology and the verified root-level DCD trajectory...")
    print("[INFO] Trajectory folder:")
    for trajectory_dir in TRAJECTORY_DIRS:
        print(f"  {trajectory_dir}")

    print("[INFO] Discovered DCD:")
    for index, dcd_file in enumerate(DCD_FILES, start=1):
        print(f"  DCD {index}: {dcd_file}")

    print(
        "[INFO] Energy logs "
        "(provenance only; not used mathematically):"
    )
    for energy_log in ENERGY_LOGS:
        print(f"  {energy_log}")

    universe = mda.Universe(
        PSF_FILE,
        *DCD_FILES,
    )

    total_frames = len(universe.trajectory)  # total frames in the selected DCD

    if total_frames < 2:
        raise RuntimeError(
            "At least two trajectory frames are required."
        )

    receptor = universe.select_atoms(
        RECEPTOR_SELECTION
    )

    ligand = universe.select_atoms(
        LIGAND_SELECTION
    )

    if receptor.n_atoms == 0:
        raise RuntimeError(
            "No PETase21 atoms found with selection:\n"
            f"{RECEPTOR_SELECTION}"
        )

    if ligand.n_atoms == 0:
        raise RuntimeError(
            "No PET dimer atoms found with selection:\n"
            f"{LIGAND_SELECTION}"
        )

    if receptor.n_residues != EXPECTED_RECEPTOR_RESIDUES:
        raise RuntimeError(
            f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues, "
            f"but found {receptor.n_residues}. Check ionized.psf."
        )

    if ligand.n_residues != EXPECTED_LIGAND_RESIDUES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_LIGAND_RESIDUES} PET-dimer LIG residue, "
            f"but found {ligand.n_residues}. Refine LIGAND_SELECTION."
        )

    if ligand.n_atoms != EXPECTED_LIGAND_ATOMS:
        raise RuntimeError(
            f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET-dimer LIG residue, "
            f"but found {ligand.n_atoms}. Check ionized.psf / LIGAND_SELECTION."
        )

    if np.intersect1d(receptor.indices, ligand.indices).size:
        raise RuntimeError("Protein and ligand selections overlap; check the PSF.")

    print(f"[INFO] PETase21 segment IDs: {sorted(set(receptor.segids))}")
    print(f"[INFO] PET dimer segment IDs: {sorted(set(ligand.segids))}")

    print(f"\nTotal system atoms: {universe.atoms.n_atoms:,}")
    print(f"PETase21 atoms: {receptor.n_atoms:,}")
    print(f"PETase21 residues: {receptor.n_residues:,}")
    print(f"PET dimer atoms: {ligand.n_atoms:,}")
    print(f"Combined trajectory frames: {total_frames:,}")
    print(
        "[IMPORTANT] Plot time is reconstructed from the frame index: "
        "t = combined_global_frame/(N_combined-1) * 500 ns. Confirm actual elapsed simulation time "
        "from the NAMD log and DCD saving interval before reporting physical ns."
    )

    pocket_residues, pocket = define_binding_pocket(
        universe=universe,
        receptor=receptor,
        ligand=ligand,
        cutoff=POCKET_CUTOFF_A,
    )

    print(
        f"\nBinding-pocket definition: PETase21 residues with >=1 atom within "
        f"{POCKET_CUTOFF_A:.1f} Å of PET dimer at the first trajectory frame"
    )
    print(f"Pocket residues: {pocket_residues.n_residues:,}")
    print(f"Pocket atoms: {pocket.n_atoms:,}")

    with open(
        OUTPUT_POCKET,
        "w",
        encoding="utf-8",
    ) as pocket_file:

        pocket_file.write("PETase21 binding-pocket residues\n")
        pocket_file.write("Reference frame: first trajectory frame\n")
        pocket_file.write(
            f"Definition: residue has >=1 atom within "
            f"{POCKET_CUTOFF_A:.1f} Å of PET dimer\n"
        )
        pocket_file.write(
            "COM calculation: mass-weighted over all atoms of these residues\n\n"
        )
        pocket_file.write("SegID\tResid\tResname\tN_atoms\n")

        for residue in pocket_residues:
            pocket_file.write(
                f"{residue.segid}\t"
                f"{residue.resid}\t"
                f"{residue.resname}\t"
                f"{residue.atoms.n_atoms}\n"
            )

    sampled_frames = list(
        range(
            0,
            total_frames,
            STEP,
        )
    )

    if sampled_frames[-1] != total_frames - 1:
        sampled_frames.append(
            total_frames - 1
        )

    time_denominator = float(
        total_frames - 1
    )  # spans the full selected trajectory

    rows = []

    missing_box_warning_reported = False

    print(
        "\nCalculating mass-weighted ligand-pocket COM distances..."
    )

    for sample_index, global_frame in enumerate(
        sampled_frames
    ):

        ts = universe.trajectory[
            global_frame
        ]

        have_valid_box = valid_periodic_box(
            ts.dimensions
        )

        if (
            not have_valid_box
            and not missing_box_warning_reported
        ):
            warnings.warn(
                "At least one DCD frame has no valid periodic cell. "
                "For such frames, the raw Cartesian ligand-pocket COM vector "
                "is retained because minimum-image correction cannot be applied."
            )

            missing_box_warning_reported = True

        unwrap_groups(
            receptor,
            ligand,
            have_valid_box=have_valid_box,
        )

        ligand_com = ligand.center_of_mass()
        pocket_com = pocket.center_of_mass()

        raw_vector = ligand_com - pocket_com

        raw_distance = float(
            np.linalg.norm(
                raw_vector
            )
        )

        corrected_vector = minimum_image_vector(
            raw_vector,
            ts.dimensions,
        )

        corrected_distance = float(
            np.linalg.norm(
                corrected_vector
            )
        )

        time_ns = (
            float(global_frame)
            / time_denominator
            * TOTAL_SIMULATION_TIME_NS
        )

        rows.append({
            "sample_index": int(sample_index),
            "global_frame": int(global_frame),
            "time_ns": float(time_ns),
            "ligand_COM_x_A": float(ligand_com[0]),
            "ligand_COM_y_A": float(ligand_com[1]),
            "ligand_COM_z_A": float(ligand_com[2]),
            "pocket_COM_x_A": float(pocket_com[0]),
            "pocket_COM_y_A": float(pocket_com[1]),
            "pocket_COM_z_A": float(pocket_com[2]),
            "raw_COM_distance_A": raw_distance,
            "ligand_pocket_dCOM_A": corrected_distance,
        })

        if (
            sample_index % 100 == 0
            or global_frame == total_frames - 1
        ):
            print(
                f"Processed global frame "
                f"{global_frame + 1:,}/{total_frames:,} "
                f"({time_ns:.3f} ns): "
                f"dCOM = {corrected_distance:.3f} Å"
            )

    # These times rescale the saved frames to a requested 0-500 ns axis.
    # Do not treat this remapping as evidence that the NAMD run lasted 500 ns.
    dataframe = pd.DataFrame(rows)

    if not np.isclose(
        dataframe["time_ns"].iloc[0],
        0.0,
    ):
        raise RuntimeError(
            "First reconstructed time is not 0 ns."
        )

    if not np.isclose(
        dataframe["time_ns"].iloc[-1],
        TOTAL_SIMULATION_TIME_NS,
    ):
        raise RuntimeError(
            "Final reconstructed time is not exactly 500 ns."
        )

    with open(
        os.path.join(OUTPUT_DIR, "time_axis_method.txt"),
        "w",
        encoding="utf-8",
    ) as metadata:
        metadata.write(
            "Timeline: normalized from 0 to 500 ns using the combined global "
            "frame index / (combined_total_frames - 1) * 500.\n"
            "The root-level ionized.dcd is used as the trajectory.\n"
            "This does not verify the actual simulated duration. For physical "
            "time, cross-check the NAMD timestep, first/last step and DCD output "
            "frequency against the simulation log.\n"
            f"Combined trajectory frames: {total_frames}\n"
            f"Sample interval in frames: {STEP}\n"
            f"Topology: {PSF_FILE}\n"
        )

        metadata.write(
            "Trajectory folder:\n"
        )

        for trajectory_dir in TRAJECTORY_DIRS:
            metadata.write(
                f"  {trajectory_dir}\n"
            )

        for index, dcd_file in enumerate(DCD_FILES, start=1):
            metadata.write(
                f"DCD {index}: {dcd_file}\n"
            )

        metadata.write(
            "Energy logs (provenance only; not used in dCOM):\n"
        )

        for energy_log in ENERGY_LOGS:
            metadata.write(
                f"  {energy_log}\n"
            )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as text_file:

        text_file.write(
            "Global_Frame\t"
            "Time_ns\t"
            "Ligand_Pocket_dCOM_A\t"
            "Raw_COM_Distance_A\t"
            "Ligand_COM_x_A\t"
            "Ligand_COM_y_A\t"
            "Ligand_COM_z_A\t"
            "Pocket_COM_x_A\t"
            "Pocket_COM_y_A\t"
            "Pocket_COM_z_A\n"
        )

        for row in dataframe.itertuples(index=False):
            text_file.write(
                f"{row.global_frame}\t"
                f"{row.time_ns:.6f}\t"
                f"{row.ligand_pocket_dCOM_A:.6f}\t"
                f"{row.raw_COM_distance_A:.6f}\t"
                f"{row.ligand_COM_x_A:.6f}\t"
                f"{row.ligand_COM_y_A:.6f}\t"
                f"{row.ligand_COM_z_A:.6f}\t"
                f"{row.pocket_COM_x_A:.6f}\t"
                f"{row.pocket_COM_y_A:.6f}\t"
                f"{row.pocket_COM_z_A:.6f}\n"
            )

    distances = dataframe[
        "ligand_pocket_dCOM_A"
    ]

    summary = pd.DataFrame({
        "Metric": [
            "Ligand-pocket dCOM (Å)"
        ],
        "N": [
            int(distances.count())
        ],
        "Mean": [
            float(distances.mean())
        ],
        "SD": [
            float(distances.std())
        ],
        "Median": [
            float(distances.median())
        ],
        "Minimum": [
            float(distances.min())
        ],
        "Maximum": [
            float(distances.max())
        ],
    })

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    save_dcom_plot(
        dataframe,
        OUTPUT_PNG,
        square=False,
    )

    save_dcom_plot(
        dataframe,
        OUTPUT_SQUARE_PNG,
        square=True,
    )

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)

    print("\nPBC-corrected ligand-pocket dCOM:")
    print(
        f"  Mean ± SD: "
        f"{distances.mean():.3f} ± {distances.std():.3f} Å"
    )
    print(f"  Median: {distances.median():.3f} Å")
    print(f"  Minimum: {distances.min():.3f} Å")
    print(f"  Maximum: {distances.max():.3f} Å")

    print("\nRescaled time-axis validation (not a physical-time verification):")
    print(
        f"  First point: "
        f"{dataframe['time_ns'].iloc[0]:.6f} ns"
    )
    print(
        f"  Final point: "
        f"{dataframe['time_ns'].iloc[-1]:.6f} ns"
    )

    print(f"\nAll outputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
