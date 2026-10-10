# -*- coding: utf-8 -*-
"""
PETase21 residue RMSF and RMSF-convergence analysis for the PET dimer-bound system.

Method retained from the supplied workflow
------------------------------------------
1. PETase21 is rigid-body aligned to the first sampled frame using its backbone.
2. Each PETase21 residue is represented by the arithmetic/geometric centroid of
   ALL atoms in that residue, including hydrogens.
3. Per-residue RMSF is calculated as:

       RMSF_i = sqrt( < |r_i(t) - <r_i>|^2 >_t )

   equivalently:

       sqrt( sum_xyz( <(r_xyz(t)-<r_xyz>)^2>_t ) )

   No extra division by 3 is applied.
4. PET dimer is validated in the topology but is NOT included in the RMSF
   calculation.

Trajectory handling
-------------------
The root-level production DCD in the PETase21/PET-dimer folder is used:
    1) ionized.dcd

The root-level ionized.dcd is discovered from the supplied PETase21/PET-dimer folder.

Forced display-time mapping
---------------------------
The separate RMSF-convergence plot uses:

    time_ns = global_frame / (combined_frames - 1) * 500

Therefore:
    first combined frame =   0 ns
    last combined frame  = 500 ns

This 0-500 ns axis is a USER-REQUESTED RESCALING of the combined frame index,
not an independent measurement of physical MD duration. Original DCD-reported
time is exported separately for verification.

The per-residue RMSF plot uses residue number on the x-axis; it is not a
time-series plot.

Energy logs
-----------
The two supplied NAMD energy logs are located and reported for provenance only.
They are not used mathematically in the RMSF calculation.

GraphPad-like plotting
----------------------
- exact plotting areas:
    wide   = 7.62 x 5.08 cm
    square = 5.08 x 5.08 cm
- gray plotting field
- black full frame
- mirrored inward ticks
- white dashed major grid
- navy residue-RMSF curve
- teal RMSF-convergence curve
- bold typography

Required packages:
    MDAnalysis, numpy, pandas, matplotlib
"""

from pathlib import Path
import re
import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# =============================================================================
# INPUT / OUTPUT PATHS
# =============================================================================

BASE_DIR = Path(
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica"
    r"\petase21-PET-02-dimer-coo-and-oh-ends"
)

PSF_FILE = BASE_DIR / "ionized.psf"

TRAJECTORY_DIRS = [
    BASE_DIR,
]

ENERGY_LOG_NAMES = [
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
]


def natural_key(path):
    """Natural filename order: part2.dcd before part10.dcd."""
    name = Path(path).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


def discover_dcd_files(run_dir):
    """
    Discover DCDs without guessing exact filenames.

    Prefer DCD files directly inside the supplied run directory. If none are
    present there, search recursively.
    """
    run_dir = Path(run_dir)

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Trajectory directory not found:\n{run_dir}"
        )

    direct = sorted(
        [
            path
            for path in run_dir.glob("*.dcd")
            if path.is_file()
        ],
        key=natural_key,
    )

    if direct:
        return direct

    recursive = sorted(
        [
            path
            for path in run_dir.rglob("*.dcd")
            if path.is_file()
            and not any(
                part.upper().startswith("RMSF_")
                for part in path.parts
            )
        ],
        key=natural_key,
    )

    if not recursive:
        raise FileNotFoundError(
            "No DCD trajectory was found in or below:\n"
            f"{run_dir}"
        )

    return recursive


def discover_energy_log(filename):
    """Locate one named NAMD log for provenance reporting only."""
    direct_candidates = [
        BASE_DIR / filename,
        *(run_dir / filename for run_dir in TRAJECTORY_DIRS),
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return str(candidate)

    matches = sorted(
        [
            path
            for path in BASE_DIR.rglob(filename)
            if path.is_file()
        ],
        key=natural_key,
    )

    if matches:
        return str(matches[0])

    return f"NOT FOUND: {filename}"


DCD_FILES = []

for run_dir in TRAJECTORY_DIRS:
    DCD_FILES.extend(
        discover_dcd_files(run_dir)
    )


ENERGY_LOGS = [
    discover_energy_log(filename)
    for filename in ENERGY_LOG_NAMES
]


OUTPUT_DIR = (
    BASE_DIR
    / "RMSF_PETase21_PET_dimer_500ns"
)


# =============================================================================
# SETTINGS
# =============================================================================

RECEPTOR_SEL = "segid AP1 and protein"

# PET dimer is validated but NOT included in RMSF.
LIGAND_SEL = "segid L and resname LIG"

# Verified against the uploaded PETase21 + PET-dimer ionized.psf.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46

# Applied to the already selected PETase21 receptor.
ALIGNMENT_SEL = "backbone"

# Analyze every frame by default.
# If STEP is increased, the physical final combined frame is still included.
STEP = 1

FORCED_TOTAL_NS = 500.0

TIME_TICKS_NS = np.arange(
    0.0,
    501.0,
    100.0,
)

REQUESTED_RESIDUE_TICK_STEP = 30

MAX_RESIDUE_TICKS = 12

# Source method:
# - all residue atoms including H
# - arithmetic/geometric residue centroid
UNWRAP_PROTEIN_WITH_BONDS = True


# =============================================================================
# GRAPHPAD-LIKE STYLE
# =============================================================================

GRAPH_COLORS = [
    "#58094F",
    "#841E62",
    "#F3AA20",
    "#2A445E",
    "#346B6D",
]

GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"

RMSF_COLOR = GRAPH_COLORS[3]          # navy
CONVERGENCE_COLOR = GRAPH_COLORS[4]   # teal


def set_plot_style():
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


def style_graphpad_axis(axis):
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

    for label in (
        list(axis.get_xticklabels())
        + list(axis.get_yticklabels())
    ):
        label.set_fontweight(
            "bold"
        )


def new_figure(square=False):
    """
    Create an axes with an EXACT plotting area:

        wide   = 7.62 x 5.08 cm
        square = 5.08 x 5.08 cm

    Margins are added outside the requested plotting area. Do not use
    tight_layout() or bbox_inches="tight", which would alter exact geometry.
    """
    plot_w = (
        5.08 if square else 7.62
    ) / 2.54

    plot_h = 5.08 / 2.54

    left = 0.82
    right = 0.23
    bottom = 0.68
    top = 0.58

    fig_w = (
        left
        + plot_w
        + right
    )

    fig_h = (
        bottom
        + plot_h
        + top
    )

    figure = plt.figure(
        figsize=(
            fig_w,
            fig_h,
        )
    )

    axis = figure.add_axes([
        left / fig_w,
        bottom / fig_h,
        plot_w / fig_w,
        plot_h / fig_h,
    ])

    style_graphpad_axis(
        axis
    )

    return figure, axis


def save_residue_plot(
    result,
    filename,
    square=False,
):
    """
    Plot final per-residue RMSF.

    The x-axis is residue position/number, NOT simulation time.
    """
    figure, axis = new_figure(
        square
    )

    x = result[
        "residue_index"
    ].to_numpy()

    rmsf = result[
        "RMSF_A"
    ].to_numpy()

    resids = result[
        "resid"
    ].to_numpy()

    axis.plot(
        x,
        rmsf,
        color=RMSF_COLOR,
        linewidth=1.35,
    )

    axis.set_xlim(
        float(x[0]),
        float(x[-1]),
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.set_xlabel(
        "PETase21 residue"
    )

    axis.set_ylabel(
        "RMSF (Å)"
    )

    axis.set_title(
        "PETase21 residue RMSF"
    )

    # Retain 30-residue spacing as the minimum requested interval, while
    # thinning labels when necessary to avoid overlap in the compact figure.
    requested = max(
        1,
        int(
            REQUESTED_RESIDUE_TICK_STEP
        ),
    )

    multiplier = max(
        1,
        int(
            np.ceil(
                len(x)
                / (
                    MAX_RESIDUE_TICKS
                    * requested
                )
            )
        ),
    )

    tick_step = (
        requested
        * multiplier
    )

    tick_idx = np.arange(
        0,
        len(x),
        tick_step,
    )

    axis.set_xticks(
        x[
            tick_idx
        ]
    )

    axis.set_xticklabels(
        [
            str(value)
            for value in resids[
                tick_idx
            ]
        ]
    )

    figure.savefig(
        filename,
        dpi=300,
    )

    plt.close(
        figure
    )


def save_convergence_plot(
    convergence,
    filename,
    square=False,
):
    """
    Plot cumulative mean residue RMSF versus the forced 0-500 ns display axis.
    """
    figure, axis = new_figure(
        square
    )

    axis.plot(
        convergence[
            "time_ns"
        ],
        convergence[
            "mean_cumulative_RMSF_A"
        ],
        color=CONVERGENCE_COLOR,
        linewidth=1.35,
        label="Mean cumulative residue RMSF",
    )

    axis.set_xlim(
        0.0,
        FORCED_TOTAL_NS,
    )

    axis.set_ylim(
        bottom=0.0
    )

    axis.set_xticks(
        TIME_TICKS_NS
    )

    axis.set_xlabel(
        "Time (ns; rescaled)"
    )

    axis.set_ylabel(
        "Mean cumulative RMSF (Å)"
    )

    axis.set_title(
        "PETase21 RMSF convergence"
    )

    figure.savefig(
        filename,
        dpi=300,
    )

    plt.close(
        figure
    )


# =============================================================================
# RMSF / TRAJECTORY HELPERS
# =============================================================================

def kabsch_rotate(
    mobile,
    reference,
):
    """
    Rotation for row-vector coordinates:

        aligned = (mobile - mobile_COM) @ R + reference_COM
    """
    mobile_center = mobile.mean(
        axis=0
    )

    reference_center = reference.mean(
        axis=0
    )

    mobile_centered = (
        mobile
        - mobile_center
    )

    reference_centered = (
        reference
        - reference_center
    )

    u_matrix, _, vt_matrix = (
        np.linalg.svd(
            mobile_centered.T
            @ reference_centered
        )
    )

    correction = np.eye(3)

    correction[
        2,
        2,
    ] = np.sign(
        np.linalg.det(
            u_matrix
            @ vt_matrix
        )
    )

    rotation = (
        u_matrix
        @ correction
        @ vt_matrix
    )

    return (
        rotation,
        mobile_center,
        reference_center,
    )


def valid_periodic_box(
    timestep,
):
    box = timestep.dimensions

    return (
        box is not None
        and len(box) >= 6
        and np.all(
            np.isfinite(
                box
            )
        )
        and np.all(
            np.asarray(
                box[:3]
            )
            > 0.0
        )
    )


def build_dcd_frame_map(
    dcd_files,
):
    """Return cumulative frame bounds for source-DCD provenance."""
    from MDAnalysis.coordinates.DCD import DCDReader

    bounds = []

    cumulative = 0

    for dcd_index, dcd_path in enumerate(
        dcd_files,
        start=1,
    ):
        with DCDReader(
            str(
                dcd_path
            )
        ) as reader:
            n_frames = len(
                reader
            )

        bounds.append({
            "dcd_index":
                dcd_index,
            "path":
                str(dcd_path),
            "start_global":
                cumulative,
            "stop_global_exclusive":
                cumulative
                + n_frames,
            "n_frames":
                n_frames,
        })

        cumulative += (
            n_frames
        )

    return bounds


def locate_source_frame(
    global_frame,
    bounds,
):
    """Map one chained global frame to its source DCD and local frame."""
    for info in bounds:

        if (
            info[
                "start_global"
            ]
            <= global_frame
            < info[
                "stop_global_exclusive"
            ]
        ):
            return (
                info[
                    "dcd_index"
                ],
                info[
                    "path"
                ],
                global_frame
                - info[
                    "start_global"
                ],
            )

    raise IndexError(
        f"Global frame {global_frame} "
        "is outside the DCD frame map."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if STEP < 1:
        raise ValueError(
            "STEP must be at least 1."
        )

    if not PSF_FILE.is_file():
        raise FileNotFoundError(
            f"PSF not found:\n{PSF_FILE}"
        )

    if not DCD_FILES:
        raise FileNotFoundError(
            "No DCD files were discovered."
        )

    for index, path in enumerate(
        DCD_FILES,
        start=1,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"DCD {index} not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_plot_style()

    print(
        "=" * 78
    )

    print(
        "PETase21 + PET dimer | residue RMSF + RMSF convergence"
    )

    print(
        "Forced display timeline for convergence = 0–500 ns"
    )

    print(
        "=" * 78
    )

    print(
        f"\nTopology:\n{PSF_FILE}"
    )

    print(
        "\nTrajectory directories "
        "(chronological run order):"
    )

    for run_dir in TRAJECTORY_DIRS:
        print(
            f"  {run_dir}"
        )

    print(
        "\nDiscovered DCD files:"
    )

    for index, path in enumerate(
        DCD_FILES,
        start=1,
    ):
        print(
            f"  DCD {index}: {path}"
        )

    print(
        "\nEnergy logs "
        "(provenance only; not used for RMSF):"
    )

    for energy_log in ENERGY_LOGS:
        print(
            f"  {energy_log}"
        )

    print(
        "\nLoading topology and chained trajectories..."
    )

    universe = mda.Universe(
        str(
            PSF_FILE
        ),
        *(
            str(path)
            for path in DCD_FILES
        ),
    )

    n_total = len(
        universe.trajectory
    )

    if n_total < 2:
        raise ValueError(
            "At least two combined DCD frames are required."
        )

    dcd_bounds = build_dcd_frame_map(
        DCD_FILES
    )

    mapped_total = sum(
        item[
            "n_frames"
        ]
        for item in dcd_bounds
    )

    if mapped_total != n_total:
        raise RuntimeError(
            f"DCD frame-map total ({mapped_total}) "
            "differs from chained trajectory "
            f"frame total ({n_total})."
        )

    print(
        "\nDCD frame counts:"
    )

    for item in dcd_bounds:
        print(
            f"  DCD {item['dcd_index']}: "
            f"{item['n_frames']:,} frames — "
            f"{item['path']}"
        )

    # -------------------------------------------------------------------------
    # Selections
    # -------------------------------------------------------------------------

    protein = universe.select_atoms(
        RECEPTOR_SEL
    )

    ligand = universe.select_atoms(
        LIGAND_SEL
    )

    if (
        protein.n_atoms == 0
        or ligand.n_atoms == 0
    ):
        raise ValueError(
            f"Could not select PETase21 "
            f"({protein.n_atoms} atoms) or "
            f"PET dimer ({ligand.n_atoms} atoms).\n"
            f"Available segids: "
            f"{sorted(set(universe.atoms.segids))}\n"
            f"Available resnames: "
            f"{sorted(set(universe.atoms.resnames))}"
        )

    if protein.n_residues != EXPECTED_RECEPTOR_RESIDUES:
        raise ValueError(
            f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues, "
            f"but found {protein.n_residues}. Check ionized.psf."
        )

    if ligand.n_residues != EXPECTED_LIGAND_RESIDUES:
        raise ValueError(
            f"Expected exactly {EXPECTED_LIGAND_RESIDUES} PET-dimer LIG residue, "
            f"but found {ligand.n_residues}."
        )

    if ligand.n_atoms != EXPECTED_LIGAND_ATOMS:
        raise ValueError(
            f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET-dimer LIG residue, "
            f"but found {ligand.n_atoms}. Check ionized.psf / ligand selection."
        )

    if np.intersect1d(
        protein.indices,
        ligand.indices,
    ).size:
        raise ValueError(
            "PETase21 and PET dimer selections overlap."
        )

    backbone = protein.select_atoms(
        ALIGNMENT_SEL
    )

    if backbone.n_atoms < 3:
        raise ValueError(
            "Too few PETase21 backbone atoms for 3-D alignment."
        )

    try:
        has_bonds = len(
            universe.atoms.bonds
        ) > 0
    except (
        mda.exceptions.NoDataError,
        AttributeError,
    ):
        has_bonds = False

    if not has_bonds:
        warnings.warn(
            "No bond connectivity detected; "
            "protein PBC unwrapping may not work."
        )

    # -------------------------------------------------------------------------
    # Vectorized all-atom residue centroids
    # -------------------------------------------------------------------------

    residue_ids, inverse = np.unique(
        protein.resindices,
        return_inverse=True,
    )

    residues = protein.residues

    if len(
        residue_ids
    ) != len(
        residues
    ):
        raise ValueError(
            "Could not match PETase21 atoms to PETase21 residues."
        )

    counts = np.bincount(
        inverse
    ).astype(
        float
    )

    protein_indices = (
        protein.indices
    )

    backbone_in_protein = (
        np.searchsorted(
            protein_indices,
            backbone.indices,
        )
    )

    if not np.array_equal(
        protein_indices[
            backbone_in_protein
        ],
        backbone.indices,
    ):
        raise ValueError(
            "Backbone atoms could not be mapped to PETase21 atoms."
        )

    # -------------------------------------------------------------------------
    # Sampled global frames
    # -------------------------------------------------------------------------

    frames = list(
        range(
            0,
            n_total,
            STEP,
        )
    )

    if frames[-1] != (
        n_total - 1
    ):
        frames.append(
            n_total - 1
        )

    # -------------------------------------------------------------------------
    # Reference backbone
    # -------------------------------------------------------------------------

    universe.trajectory[
        frames[0]
    ]

    if (
        UNWRAP_PROTEIN_WITH_BONDS
        and valid_periodic_box(
            universe.trajectory.ts
        )
    ):
        try:
            protein.unwrap(
                compound="fragments",
                reference="com",
                inplace=True,
            )
        except (
            ValueError,
            AttributeError,
            mda.exceptions.NoDataError,
        ) as error:
            warnings.warn(
                "Reference-frame protein unwrapping skipped: "
                f"{error}"
            )

    reference_backbone = np.asarray(
        backbone.positions,
        dtype=np.float64,
    ).copy()

    # -------------------------------------------------------------------------
    # Online RMSF accumulation
    # -------------------------------------------------------------------------

    mean = np.zeros(
        (
            len(residues),
            3,
        ),
        dtype=np.float64,
    )

    m2 = np.zeros(
        (
            len(residues),
            3,
        ),
        dtype=np.float64,
    )

    convergence_rows = []

    n_seen = 0

    unwrap_failure_reported = False

    print(
        f"\nPETase21 atoms: "
        f"{protein.n_atoms:,}; "
        f"residues: "
        f"{len(residues):,}"
    )

    print(
        f"PET dimer atoms: "
        f"{ligand.n_atoms:,}; "
        f"global frames: "
        f"{n_total:,}"
    )

    print(
        f"Sampling "
        f"{len(frames):,} frames; "
        "assigned convergence timeline 0–500 ns"
    )

    for sampled_index, global_frame in enumerate(
        frames
    ):
        timestep = universe.trajectory[
            global_frame
        ]

        if (
            UNWRAP_PROTEIN_WITH_BONDS
            and valid_periodic_box(
                timestep
            )
        ):
            try:
                protein.unwrap(
                    compound="fragments",
                    reference="com",
                    inplace=True,
                )

            except (
                ValueError,
                AttributeError,
                mda.exceptions.NoDataError,
            ) as error:

                if not unwrap_failure_reported:
                    warnings.warn(
                        "Protein PBC unwrapping skipped "
                        "for at least one frame: "
                        f"{error}"
                    )

                    unwrap_failure_reported = (
                        True
                    )

        coords = np.asarray(
            protein.positions,
            dtype=np.float64,
        )

        rotation, mobile_com, reference_com = (
            kabsch_rotate(
                coords[
                    backbone_in_protein
                ],
                reference_backbone,
            )
        )

        aligned = (
            (coords - mobile_com)
            @ rotation
            + reference_com
        )

        # Per-residue arithmetic/geometric center of ALL atoms, including H.
        centroids = np.column_stack([
            np.bincount(
                inverse,
                weights=aligned[
                    :,
                    coordinate_index
                ],
                minlength=len(
                    residues
                ),
            )
            / counts
            for coordinate_index in range(
                3
            )
        ])

        # Welford online variance per coordinate.
        # RMSF = sqrt(sum_xyz(M2_xyz / n_frames)).
        n_seen += 1

        delta = (
            centroids
            - mean
        )

        mean += (
            delta
            / n_seen
        )

        m2 += (
            delta
            * (
                centroids
                - mean
            )
        )

        cumulative_rmsf = np.sqrt(
            np.maximum(
                m2.sum(
                    axis=1
                )
                / n_seen,
                0.0,
            )
        )

        assigned_ns = (
            float(
                global_frame
            )
            / float(
                n_total - 1
            )
            * FORCED_TOTAL_NS
        )

        (
            source_dcd_index,
            source_dcd_file,
            source_local_frame,
        ) = locate_source_frame(
            global_frame,
            dcd_bounds,
        )

        # MDAnalysis DCD time may reset or have metadata quirks at chained-file
        # boundaries, so retain it only as a verification/provenance column.
        try:
            dcd_time_ns = (
                float(
                    timestep.time
                )
                / 1000.0
            )
        except Exception:
            dcd_time_ns = np.nan

        convergence_rows.append({
            "global_frame":
                int(
                    global_frame
                ),
            "time_ns":
                float(
                    assigned_ns
                ),
            "dcd_time_ns":
                dcd_time_ns,
            "source_dcd_index":
                int(
                    source_dcd_index
                ),
            "source_local_frame":
                int(
                    source_local_frame
                ),
            "source_dcd_file":
                source_dcd_file,
            "sampled_frames_so_far":
                int(
                    n_seen
                ),
            "mean_cumulative_RMSF_A":
                float(
                    cumulative_rmsf.mean()
                ),
            "median_cumulative_RMSF_A":
                float(
                    np.median(
                        cumulative_rmsf
                    )
                ),
        })

        if (
            sampled_index % 100 == 0
            or sampled_index
            == len(frames) - 1
        ):
            print(
                f"Frame "
                f"{global_frame + 1:,}/"
                f"{n_total:,} — "
                f"mapped "
                f"{assigned_ns:.3f} ns"
            )

    # -------------------------------------------------------------------------
    # Final RMSF
    # -------------------------------------------------------------------------

    rmsf = np.sqrt(
        np.maximum(
            m2.sum(
                axis=1
            )
            / n_seen,
            0.0,
        )
    )

    results = pd.DataFrame({
        "residue_index":
            np.arange(
                1,
                len(residues) + 1,
                dtype=int,
            ),
        "resid":
            [
                int(
                    residue.resid
                )
                for residue in residues
            ],
        "resname":
            [
                str(
                    residue.resname
                )
                for residue in residues
            ],
        "segid":
            [
                str(
                    residue.segid
                )
                for residue in residues
            ],
        "RMSF_A":
            rmsf,
    })

    convergence = pd.DataFrame(
        convergence_rows
    )

    if (
        not np.isclose(
            convergence[
                "time_ns"
            ].iloc[0],
            0.0,
        )
        or not np.isclose(
            convergence[
                "time_ns"
            ].iloc[-1],
            FORCED_TOTAL_NS,
        )
    ):
        raise RuntimeError(
            "The rescaled timeline does not cover exactly 0–500 ns."
        )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    prefix = (
        "PETase21_PET_dimer"
    )

    residue_txt = (
        OUTPUT_DIR
        / f"{prefix}_RMSF_by_residue.txt"
    )

    residue_csv = (
        OUTPUT_DIR
        / f"{prefix}_RMSF_by_residue.csv"
    )

    convergence_csv = (
        OUTPUT_DIR
        / f"{prefix}_RMSF_convergence_500ns.csv"
    )

    results[
        [
            "resid",
            "resname",
            "RMSF_A",
        ]
    ].to_csv(
        residue_txt,
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    results.to_csv(
        residue_csv,
        index=False,
        encoding="utf-8-sig",
    )

    convergence.to_csv(
        convergence_csv,
        index=False,
        encoding="utf-8-sig",
    )

    save_residue_plot(
        results,
        OUTPUT_DIR
        / f"{prefix}_RMSF_by_residue.png",
    )

    save_residue_plot(
        results,
        OUTPUT_DIR
        / f"{prefix}_RMSF_by_residue_square.png",
        True,
    )

    save_convergence_plot(
        convergence,
        OUTPUT_DIR
        / f"{prefix}_RMSF_convergence_500ns.png",
    )

    save_convergence_plot(
        convergence,
        OUTPUT_DIR
        / f"{prefix}_RMSF_convergence_500ns_square.png",
        True,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = (
        OUTPUT_DIR
        / f"{prefix}_RMSF_summary.txt"
    )

    summary_lines = [
        "PETase21 / PET dimer RMSF analysis",
        "=" * 70,
        "",
        f"Receptor: {RECEPTOR_SEL}",
        f"Ligand: {LIGAND_SEL} (validated but not included in RMSF selection)",
        f"PSF: {PSF_FILE}",
        "",
        "Trajectory folders in chronological order:",
    ]

    summary_lines.extend(
        [
            f"  {run_dir}"
            for run_dir in TRAJECTORY_DIRS
        ]
    )

    summary_lines.extend([
        "",
        "Discovered DCD files in chained order:",
    ])

    summary_lines.extend(
        [
            f"  DCD {index}: {path}"
            for index, path in enumerate(
                DCD_FILES,
                start=1,
            )
        ]
    )

    summary_lines.extend([
        "",
        "Energy logs (provenance only; not used for RMSF):",
    ])

    summary_lines.extend(
        [
            f"  {energy_log}"
            for energy_log in ENERGY_LOGS
        ]
    )

    summary_lines.extend([
        "",
        f"Total combined DCD frames: {n_total}",
        f"Sampled frames: {n_seen}",
        f"PETase21 residue count: {len(residues)}",
        "",
        "Method:",
        "  Backbone-fitted per-residue RMSF.",
        "  Residue representation = arithmetic/geometric centroid of all residue atoms, including H.",
        "  RMSF = sqrt(sum_xyz(mean_t[(r_xyz(t)-mean_t(r_xyz))**2])).",
        "  No extra division by 3 is applied.",
        "",
        f"Mean residue RMSF: {rmsf.mean():.6f} Å",
        f"Median residue RMSF: {np.median(rmsf):.6f} Å",
        "",
        "Mapped convergence-time axis:",
        "  first combined frame = 0 ns",
        "  final combined frame = 500 ns",
        "  formula = global_frame/(total_frames-1)*500",
        "",
        "IMPORTANT:",
        "  500 ns is a forced plotting/data rescaling, not verified physical simulation duration.",
        "  Original DCD-reported time is exported in the convergence CSV for verification.",
        "  The residue-RMSF plot uses residue number on the x-axis, not time.",
    ])

    summary.write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final console report
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RMSF ANALYSIS COMPLETED"
    )

    print(
        "=" * 78
    )

    print(
        f"\nFiles saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nFinal mean residue RMSF: "
        f"{rmsf.mean():.4f} Å"
    )

    print(
        f"Median residue RMSF: "
        f"{np.median(rmsf):.4f} Å"
    )

    print(
        "Mapped convergence-time endpoints: "
        "0 ns and 500 ns "
        "(not physical-time validation)"
    )


if __name__ == "__main__":
    main()
