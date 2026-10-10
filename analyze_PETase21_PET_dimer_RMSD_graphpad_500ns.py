# -*- coding: utf-8 -*-

"""
RMSD analysis for PETase21 + PET dimer across sequential NAMD trajectories.

Analyses
--------
1. PETase21 backbone RMSD
2. PET dimer RMSD relative to backbone-aligned PETase21
3. PET dimer + binding-pocket RMSD

Methodological details
----------------------
- PETase21 is selected as: segid AP1 and protein
- PETase21 backbone is selected as: segid AP1 and protein and backbone
- PET dimer is selected as: segid L and resname LIG
- The binding pocket is defined ONCE from the first combined trajectory frame
  as all PETase21 residues containing at least one atom within 5.0 Å of
  PET dimer.
- Every sampled frame is aligned to the first frame using the PETase21 backbone
  before RMSD is calculated. This removes global translation/rotation and makes
  ligand RMSD interpretable relative to the receptor.
- Basic PBC handling reconstructs receptor/ligand bonded fragments and places
  PET dimer in the periodic image nearest PETase21 before alignment.

Trajectory handling
-------------------
The root-level PETase21/PET-dimer DCD is loaded from:

    1) NAMD_PETase21_PET_dimer_310K_ALL_IN_ONE
    2) NAMD_PETase21_PET_dimer_310K_NPT_RESTART

The root-level ionized.dcd is discovered from the supplied PETase21/PET-dimer folder.

Forced time scale
-----------------
The combined global frame sequence is deliberately remapped to:

    time_ns =
        combined_global_frame
        / (combined_total_frames - 1)
        * 500

Therefore:
    first combined frame =   0 ns
    final combined frame  = 500 ns

This does not modify coordinates or RMSD values. It only defines the requested
display/export time axis and does NOT independently establish 500 ns of
physical MD time.

Energy logs
-----------
The supplied NAMD energy logs are located and recorded for provenance only.
They are not used mathematically in the RMSD calculation.

GraphPad-like format
--------------------
- gray plotting area
- full black frame
- inward mirrored major/minor ticks
- white dashed major grid
- bold typography
- established palette
- exact axes plotting areas:
    wide   = 7.62 x 5.08 cm
    square = 5.08 x 5.08 cm

Curve colors
------------
- PETase21 backbone            = #2A445E (navy)
- PET dimer                 = #841E62 (magenta)
- PET dimer + binding pocket = #F3AA20 (gold)

Required packages:
    numpy, pandas, matplotlib, MDAnalysis
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.lib.distances import minimize_vectors

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

    DCDs directly inside the supplied run directory are preferred. If none
    are present there, the directory is searched recursively.
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
            if not directory.upper().startswith("RMSD_")
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
    "RMSD_PETase21_PET_dimer_500ns",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_RMSD_500ns.txt",
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_RMSD_500ns.csv",
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_RMSD_summary.csv",
)

OUTPUT_POCKET = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_binding_pocket_residues.txt",
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_RMSD_500ns.png",
)

OUTPUT_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PETase21_PET_dimer_RMSD_500ns_square.png",
)


# ============================================================
# SETTINGS
# ============================================================

RECEPTOR_SELECTION = "segid AP1 and protein"
BACKBONE_SELECTION = "segid AP1 and protein and backbone"
LIGAND_SELECTION = "segid L and resname LIG"

# Verified for the PETase21 + PET-dimer ionized.psf.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46

# Static binding-pocket definition from first trajectory frame.
DISTANCE_CUTOFF_A = 5.0

# Analyze every frame.
STEP = 1

# Explicit requested final display duration.
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

# Consistent colors for the three RMSD curves.
COLOR_PROTEIN = GRAPH_COLORS[3]   # navy
COLOR_LIGAND = GRAPH_COLORS[1]    # magenta
COLOR_POCKET = GRAPH_COLORS[2]    # gold


def set_plot_style():
    """Apply the same GraphPad-like typography used in related scripts."""

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
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 150,
    })


def style_graphpad_axis(
    axis,
    show_grid=True,
):
    """Gray plotting field, black frame, mirrored ticks, white dashed grid."""

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
    Set exact axes plotting area:
        wide   = 7.62 x 5.08 cm
        square = 5.08 x 5.08 cm
    """

    width = (
        5.08 if square else 7.62
    ) / 2.54

    height = 5.08 / 2.54

    figure.canvas.draw()

    old_size = figure.get_size_inches()

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


# ============================================================
# RMSD / PBC HELPERS
# ============================================================

def rmsd_to_reference(
    reference_positions,
    current_positions,
):
    """
    Conventional unweighted Cartesian RMSD after the receptor-based alignment.
    """

    difference = (
        current_positions
        - reference_positions
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    difference * difference,
                    axis=1,
                )
            )
        )
    )


def valid_periodic_box(
    dimensions,
):
    """Return True when trajectory unit-cell dimensions are usable."""

    if dimensions is None:
        return False

    dimensions = np.asarray(
        dimensions,
        dtype=float,
    )

    if dimensions.size < 6:
        return False

    if not np.all(
        np.isfinite(
            dimensions
        )
    ):
        return False

    return bool(
        np.all(
            dimensions[:3]
            > 0.0
        )
    )


def prepare_complex_pbc(receptor, ligand, dimensions):
    """Reassemble bonded fragments and place ligand near the receptor under PBC."""
    if not valid_periodic_box(dimensions):
        warnings.warn(
            "No usable PBC box in the DCD frame; coordinates will not be unwrapped. "
            "Check the resulting RMSD for boundary artifacts.",
            stacklevel=2,
        )
        return

    # For a receptor split over a periodic boundary, rebuilding is essential
    # before the backbone fit.  A missing bond topology should not silently
    # generate convincing but incorrect RMSD plots.
    try:
        receptor.unwrap(compound="fragments", reference="com", inplace=True)
        ligand.unwrap(compound="fragments", reference="com", inplace=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to reconstruct PETase21/PET dimer across PBC. "
            "Check ionized.psf bonds and ionized.dcd box dimensions."
        ) from exc

    receptor_com = receptor.center_of_mass()
    ligand_com = ligand.center_of_mass()
    displacement = ligand_com - receptor_com
    nearest_displacement = minimize_vectors(displacement[None, :], dimensions)[0]
    ligand.translate((receptor_com + nearest_displacement) - ligand_com)


def align_positions_to_reference(
    mobile_backbone,
    reference_backbone,
    mobile_positions,
    reference_center,
):
    """
    Return arbitrary mobile coordinates transformed using the rigid-body fit
    derived from mobile_backbone -> reference_backbone.
    """

    mobile_center = (
        mobile_backbone.mean(
            axis=0
        )
    )

    mobile_centered = (
        mobile_backbone
        - mobile_center
    )

    reference_centered = (
        reference_backbone
        - reference_center
    )

    rotation, _ = rotation_matrix(
        mobile_centered,
        reference_centered,
    )

    if not np.all(
        np.isfinite(
            rotation
        )
    ):
        raise RuntimeError(
            "Backbone alignment returned a non-finite rotation matrix."
        )

    aligned = (
        np.dot(
            mobile_positions
            - mobile_center,
            rotation.T,
        )
        + reference_center
    )

    return aligned


# ============================================================
# PLOTTING
# ============================================================

def save_rmsd_plot(dataframe, output_path, square=False):
    """Produce plots with precisely sized AXES, independent of outer margins."""
    set_plot_style()
    plot_w = (5.08 if square else 7.62) / 2.54
    plot_h = 5.08 / 2.54
    left, bottom, right, top = 0.85, 0.65, 0.25, 0.55
    fig_w = left + plot_w + right
    fig_h = bottom + plot_h + top
    figure = plt.figure(figsize=(fig_w, fig_h))
    axis = figure.add_axes([
        left / fig_w, bottom / fig_h, plot_w / fig_w, plot_h / fig_h
    ])
    style_graphpad_axis(axis)
    axis.plot(
        dataframe["time_ns"], dataframe["protein_backbone_rmsd_A"],
        label="PETase21 backbone", color=COLOR_PROTEIN, linewidth=1.25,
    )
    axis.plot(
        dataframe["time_ns"], dataframe["ligand_rmsd_A"],
        label="PET dimer", color=COLOR_LIGAND, linewidth=1.25,
    )
    axis.plot(
        dataframe["time_ns"], dataframe["ligand_pocket_rmsd_A"],
        label="PET dimer + pocket", color=COLOR_POCKET, linewidth=1.25,
    )
    axis.set_xlabel("Time (ns; rescaled)")
    axis.set_ylabel("RMSD (Å)")
    axis.set_xlim(0.0, TOTAL_SIMULATION_TIME_NS)
    axis.set_xticks(TIME_TICKS_NS)
    axis.set_title("PETase21–PET dimer RMSD")
    axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "PETase21 / PET dimer RMSD ANALYSIS"
    )

    print(
        f"Backbone-aligned; {len(DCD_FILES)} discovered DCD file(s); combined rescaled timeline = 0-500 ns"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if STEP < 1:
        raise ValueError(
            "STEP must be >= 1."
        )

    if not os.path.isfile(
        PSF_FILE
    ):
        raise FileNotFoundError(
            f"PSF file not found:\n{PSF_FILE}"
        )

    for dcd_file in DCD_FILES:

        if not os.path.isfile(
            dcd_file
        ):
            raise FileNotFoundError(
                f"DCD file not found:\n{dcd_file}"
            )

    # --------------------------------------------------------
    # Load all discovered DCDs as one logical trajectory
    # --------------------------------------------------------

    print(
        "\nLoading topology and all discovered DCDs as one chained trajectory..."
    )

    print("[INFO] Trajectory folders in chronological order:")
    for trajectory_dir in TRAJECTORY_DIRS:
        print(
            f"  {trajectory_dir}"
        )

    print("[INFO] Discovered DCD order:")
    for index, dcd_file in enumerate(DCD_FILES, start=1):
        print(
            f"  DCD {index}: "
            f"{dcd_file}"
        )

    print(
        "[INFO] Energy logs "
        "(provenance only; not used mathematically):"
    )
    for energy_log in ENERGY_LOGS:
        print(
            f"  {energy_log}"
        )

    universe = mda.Universe(
        PSF_FILE,
        *DCD_FILES,
    )

    total_frames = len(
        universe.trajectory
    )  # total frames across all chained DCD files

    if total_frames < 2:
        raise RuntimeError(
            "At least two trajectory frames are required."
        )

    receptor = universe.select_atoms(
        RECEPTOR_SELECTION
    )

    backbone = universe.select_atoms(
        BACKBONE_SELECTION
    )

    ligand = universe.select_atoms(
        LIGAND_SELECTION
    )

    if receptor.n_atoms == 0:
        raise RuntimeError(
            "No PETase21 atoms were found with selection:\n"
            f"{RECEPTOR_SELECTION}"
        )

    if backbone.n_atoms == 0:
        raise RuntimeError(
            "No PETase21 backbone atoms were found with selection:\n"
            f"{BACKBONE_SELECTION}"
        )

    if ligand.n_atoms == 0:
        raise RuntimeError(
            "No PET dimer atoms were found with selection:\n"
            f"{LIGAND_SELECTION}"
        )

    if receptor.n_residues != EXPECTED_RECEPTOR_RESIDUES:
        raise RuntimeError(
            f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues; "
            f"found {receptor.n_residues}. Check ionized.psf."
        )

    if ligand.n_residues != EXPECTED_LIGAND_RESIDUES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_LIGAND_RESIDUES} PET-dimer LIG residue; "
            f"found {ligand.n_residues}. Inspect ligand selection and PSF."
        )

    if ligand.n_atoms != EXPECTED_LIGAND_ATOMS:
        raise RuntimeError(
            f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET-dimer LIG residue; "
            f"found {ligand.n_atoms}. Check ionized.psf / ligand selection."
        )

    if np.intersect1d(receptor.indices, ligand.indices).size:
        raise RuntimeError("Protein and ligand selections overlap.")
    if backbone.n_atoms < 3:
        raise RuntimeError("Backbone alignment requires at least three protein atoms.")

    print(
        f"\nTotal system atoms: "
        f"{universe.atoms.n_atoms:,}"
    )

    print(
        f"PETase21 atoms: "
        f"{receptor.n_atoms:,}"
    )

    print(
        f"PETase21 residues: "
        f"{receptor.n_residues:,}"
    )

    print(
        f"PETase21 backbone atoms: "
        f"{backbone.n_atoms:,}"
    )

    print(
        f"PET dimer atoms: "
        f"{ligand.n_atoms:,}"
    )

    print(
        f"Combined trajectory frames: "
        f"{total_frames:,}"
    )

    # --------------------------------------------------------
    # Reference frame + PBC preparation
    # --------------------------------------------------------

    universe.trajectory[
        0
    ]

    prepare_complex_pbc(
        receptor,
        ligand,
        universe.trajectory.ts.dimensions,
    )

    # Static pocket definition:
    # identify protein atoms within cutoff, then expand to all atoms of those
    # PETase21 residues.
    pocket_seed = universe.select_atoms(
        (
            f"({RECEPTOR_SELECTION}) "
            f"and around {DISTANCE_CUTOFF_A:.6f} group lig"
        ),
        lig=ligand,
        periodic=False,
    )

    if pocket_seed.n_atoms == 0:
        raise RuntimeError(
            "No PETase21 binding-pocket atoms were found within "
            f"{DISTANCE_CUTOFF_A:.1f} Å of PET dimer in the first combined trajectory frame."
        )

    pocket_residues = (
        pocket_seed.residues
    )

    binding_pocket = (
        pocket_residues.atoms
    )

    print(
        f"\nStatic pocket cutoff: "
        f"{DISTANCE_CUTOFF_A:.1f} Å"
    )

    print(
        f"Pocket residues: "
        f"{pocket_residues.n_residues:,}"
    )

    print(
        f"Pocket atoms after whole-residue expansion: "
        f"{binding_pocket.n_atoms:,}"
    )

    # --------------------------------------------------------
    # Save pocket residue identities
    # --------------------------------------------------------

    with open(
        OUTPUT_POCKET,
        "w",
        encoding="utf-8",
    ) as pocket_file:

        pocket_file.write(
            "PETase21 binding-pocket residues\n"
        )

        pocket_file.write(
            f"Reference: first combined trajectory frame\n"
        )

        pocket_file.write(
            f"Definition: residue contains >=1 atom within "
            f"{DISTANCE_CUTOFF_A:.1f} Å of PET dimer\n\n"
        )

        pocket_file.write(
            "SegID\tResid\tResname\n"
        )

        for residue in pocket_residues:

            pocket_file.write(
                f"{residue.segid}\t"
                f"{residue.resid}\t"
                f"{residue.resname}\n"
            )

    # --------------------------------------------------------
    # Store reference coordinates
    # --------------------------------------------------------

    reference_backbone = (
        backbone.positions.copy()
    )

    reference_center = (
        reference_backbone.mean(
            axis=0
        )
    )

    reference_ligand = (
        ligand.positions.copy()
    )

    reference_pocket = (
        binding_pocket.positions.copy()
    )

    reference_ligand_pocket = np.vstack(
        [
            reference_ligand,
            reference_pocket,
        ]
    )

    # --------------------------------------------------------
    # Global sampled frame indices across all chained DCD files
    # --------------------------------------------------------

    sampled_frames = list(
        range(
            0,
            total_frames,
            STEP,
        )
    )

    if (
        sampled_frames[-1]
        != total_frames - 1
    ):

        sampled_frames.append(
            total_frames - 1
        )

    time_denominator = float(
        total_frames - 1
    )  # denominator spans the full chained trajectory

    # --------------------------------------------------------
    # RMSD loop
    # --------------------------------------------------------

    rows = []

    print(
        "\nCalculating backbone-aligned RMSD..."
    )

    for sampled_index, global_frame in enumerate(
        sampled_frames
    ):

        ts = universe.trajectory[
            global_frame
        ]

        prepare_complex_pbc(
            receptor,
            ligand,
            ts.dimensions,
        )

        current_backbone = (
            backbone.positions.copy()
        )

        current_ligand = (
            ligand.positions.copy()
        )

        current_pocket = (
            binding_pocket.positions.copy()
        )

        # Rigid-body alignment is derived ONLY from PETase21 backbone.
        aligned_backbone = align_positions_to_reference(
            mobile_backbone=current_backbone,
            reference_backbone=reference_backbone,
            mobile_positions=current_backbone,
            reference_center=reference_center,
        )

        aligned_ligand = align_positions_to_reference(
            mobile_backbone=current_backbone,
            reference_backbone=reference_backbone,
            mobile_positions=current_ligand,
            reference_center=reference_center,
        )

        aligned_pocket = align_positions_to_reference(
            mobile_backbone=current_backbone,
            reference_backbone=reference_backbone,
            mobile_positions=current_pocket,
            reference_center=reference_center,
        )

        aligned_ligand_pocket = np.vstack(
            [
                aligned_ligand,
                aligned_pocket,
            ]
        )

        protein_rmsd = rmsd_to_reference(
            reference_backbone,
            aligned_backbone,
        )

        ligand_rmsd = rmsd_to_reference(
            reference_ligand,
            aligned_ligand,
        )

        ligand_pocket_rmsd = rmsd_to_reference(
            reference_ligand_pocket,
            aligned_ligand_pocket,
        )

        time_ns = (
            float(global_frame)
            / time_denominator
            * TOTAL_SIMULATION_TIME_NS
        )

        rows.append({
            "sample_index":
                sampled_index,
            "global_frame":
                int(global_frame),
            "time_ns":
                float(time_ns),
            "protein_backbone_rmsd_A":
                float(protein_rmsd),
            "ligand_rmsd_A":
                float(ligand_rmsd),
            "ligand_pocket_rmsd_A":
                float(ligand_pocket_rmsd),
        })

        if (
            sampled_index % 100 == 0
            or global_frame
            == total_frames - 1
        ):

            print(
                f"Processed trajectory frame "
                f"{global_frame + 1:,}/"
                f"{total_frames:,} "
                f"({time_ns:.3f} ns)"
            )

    dataframe = pd.DataFrame(
        rows
    )

    # Save explicit metadata documenting the two-file time-axis treatment.
    with open(
        os.path.join(OUTPUT_DIR, "time_axis_method.txt"),
        "w",
        encoding="utf-8",
    ) as metadata:
        metadata.write(
            "Timeline: combined global frame index normalized to 0-500 ns.\n"
            "Formula: time_ns = combined_global_frame / "
            "(combined_total_frames - 1) * 500.\n"
            "This is a display/export rescaling and does not independently "
            "verify the physical MD duration.\n"
            f"Combined trajectory frames: {total_frames}\n"
            f"Sampling step in frames: {STEP}\n"
            f"Topology: {PSF_FILE}\n"
        )
        metadata.write(
            "Trajectory folders (chronological order):\n"
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
            "Energy logs (provenance only; not used in RMSD):\n"
        )

        for energy_log in ENERGY_LOGS:
            metadata.write(
                f"  {energy_log}\n"
            )

    # --------------------------------------------------------
    # Validate 0-500 ns axis
    # --------------------------------------------------------

    if not np.isclose(
        dataframe[
            "time_ns"
        ].iloc[0],
        0.0,
    ):
        raise RuntimeError(
            "First reconstructed time is not 0 ns."
        )

    if not np.isclose(
        dataframe[
            "time_ns"
        ].iloc[-1],
        TOTAL_SIMULATION_TIME_NS,
    ):
        raise RuntimeError(
            "Final reconstructed time is not exactly 500 ns."
        )

    # --------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------

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
            "PETase21_Backbone_RMSD_A\t"
            "PET dimer_RMSD_A\t"
            "PET dimer_Plus_Pocket_RMSD_A\n"
        )

        for row in dataframe.itertuples(
            index=False
        ):

            text_file.write(
                f"{row.global_frame}\t"
                f"{row.time_ns:.6f}\t"
                f"{row.protein_backbone_rmsd_A:.6f}\t"
                f"{row.ligand_rmsd_A:.6f}\t"
                f"{row.ligand_pocket_rmsd_A:.6f}\n"
            )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    rmsd_columns = [
        "protein_backbone_rmsd_A",
        "ligand_rmsd_A",
        "ligand_pocket_rmsd_A",
    ]

    summary = pd.DataFrame({
        "N":
            dataframe[
                rmsd_columns
            ].count(),
        "Mean_A":
            dataframe[
                rmsd_columns
            ].mean(),
        "SD_A":
            dataframe[
                rmsd_columns
            ].std(),
        "Median_A":
            dataframe[
                rmsd_columns
            ].median(),
        "Minimum_A":
            dataframe[
                rmsd_columns
            ].min(),
        "Maximum_A":
            dataframe[
                rmsd_columns
            ].max(),
    })

    summary.to_csv(
        OUTPUT_SUMMARY,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Save GraphPad-like figures
    # --------------------------------------------------------

    save_rmsd_plot(
        dataframe,
        OUTPUT_PNG,
        square=False,
    )

    save_rmsd_plot(
        dataframe,
        OUTPUT_SQUARE_PNG,
        square=True,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

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
        "\nForced time-axis validation:"
    )

    print(
        f"  First point: "
        f"{dataframe['time_ns'].iloc[0]:.6f} ns"
    )

    print(
        f"  Final point: "
        f"{dataframe['time_ns'].iloc[-1]:.6f} ns"
    )

    print("NOTE: 0–500 ns is a combined-frame-index rescaling across all chained DCDs, "
          "not independent proof of 500 ns of physical MD.")
    print(
        "\nMean RMSD values:"
    )

    print(
        f"  PETase21 backbone: "
        f"{dataframe['protein_backbone_rmsd_A'].mean():.3f} Å"
    )

    print(
        f"  PET dimer: "
        f"{dataframe['ligand_rmsd_A'].mean():.3f} Å"
    )

    print(
        f"  PET dimer + pocket: "
        f"{dataframe['ligand_pocket_rmsd_A'].mean():.3f} Å"
    )

    print(
        f"\nAll outputs saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nMain files:"
    )

    print(
        f"  {OUTPUT_TXT}"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        f"  {OUTPUT_POCKET}"
    )

    print(
        f"  {OUTPUT_PNG}"
    )

    print(
        f"  {OUTPUT_SQUARE_PNG}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
