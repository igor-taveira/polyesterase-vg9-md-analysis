# -*- coding: utf-8 -*-
"""
PETase21 protein radius-of-gyration analysis for the PET-dimer-bound system.

Scientific definition
---------------------
Rg is the mass-weighted radius of gyration of the PETase21 protein only:

    Rg = sqrt[
        sum_i m_i |r_i - r_COM|^2
        /
        sum_i m_i
    ]

PET dimer and solvent are excluded from the Rg calculation.

Periodic boundaries
-------------------
When a valid periodic cell is available, the bonded PETase21 protein is
reconstructed across PBC before calculating its mass-weighted center of mass
and radius of gyration. If a frame lacks a valid unit cell, the stored protein
coordinates are used and a warning is issued once.

Trajectory handling
-------------------
The verified root-level trajectory is:
    ionized.dcd

Forced 0-500 ns display axis
----------------------------
The global frame sequence is mapped to:

    time_ns =
        global_frame
        / (trajectory_frames - 1)
        * 500

Thus:
    first frame =   0 ns
    final frame  = 500 ns

This is a USER-REQUESTED presentation/export rescaling. It does NOT
independently establish 500 ns of physical MD simulation time.

Energy log
----------
The named NAMD energy log:
    energia-petase21-pet-dimer-coo-and-oh-ends.log
is located and reported for provenance only; it is not used mathematically
in the Rg calculation.

GraphPad-like style
-------------------
- original gray Rg trace (#808080) retained
- gray plotting field
- complete black frame
- mirrored inward major/minor ticks
- white dashed major grid
- bold typography
- exact plotting areas:
    wide   = 7.62 x 5.08 cm
    square = 5.08 x 5.08 cm

Requires:
    MDAnalysis, numpy, matplotlib
"""

from pathlib import Path
import re
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError


# =============================================================================
# INPUT PATHS
# =============================================================================

BASE_DIR = Path(
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica"
    r"\petase21-PET-02-dimer-coo-and-oh-ends"
)

TOPOLOGY = (
    BASE_DIR
    / "ionized.psf"
)

TRAJECTORY_DIRS = [
    BASE_DIR,
]

ENERGY_LOG_NAMES = [
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
]


def natural_key(path):
    """Natural filename ordering: part2.dcd before part10.dcd."""
    name = Path(path).name

    return [
        int(token)
        if token.isdigit()
        else token.lower()
        for token in re.split(
            r"(\d+)",
            name,
        )
    ]


def discover_dcd_files(run_dir):
    """
    Discover DCD files without guessing exact filenames.

    DCDs directly inside the supplied PET-dimer directory are preferred. If none
    are present there, the directory is searched recursively.
    """
    run_dir = Path(
        run_dir
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            "Trajectory directory not found:\n"
            f"{run_dir}"
        )

    direct = sorted(
        [
            path
            for path in run_dir.glob(
                "*.dcd"
            )
            if path.is_file()
        ],
        key=natural_key,
    )

    if direct:
        return direct

    recursive = sorted(
        [
            path
            for path in run_dir.rglob(
                "*.dcd"
            )
            if path.is_file()
            and not any(
                part.upper().startswith(
                    "RG_"
                )
                for part in path.parts
            )
        ],
        key=natural_key,
    )

    if not recursive:
        raise FileNotFoundError(
            "No DCD trajectory was found "
            "in or below:\n"
            f"{run_dir}"
        )

    return recursive


def discover_energy_log(filename):
    """
    Locate one named NAMD energy log for provenance reporting only.
    """
    direct_candidates = [
        BASE_DIR
        / filename,
        *(
            run_dir
            / filename
            for run_dir
            in TRAJECTORY_DIRS
        ),
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return str(
                candidate
            )

    matches = sorted(
        [
            path
            for path
            in BASE_DIR.rglob(
                filename
            )
            if path.is_file()
        ],
        key=natural_key,
    )

    if matches:
        return str(
            matches[0]
        )

    return (
        f"NOT FOUND: {filename}"
    )


DCD_FILES = []

for run_dir in TRAJECTORY_DIRS:
    DCD_FILES.extend(
        discover_dcd_files(
            run_dir
        )
    )


ENERGY_LOGS = [
    discover_energy_log(
        filename
    )
    for filename
    in ENERGY_LOG_NAMES
]


OUTPUT_DIR = (
    BASE_DIR
    / "RG_PETase21_PET_dimer_500ns"
)


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

PROTEIN_SELECTION = (
    "segid AP1 and protein"
)

# Validation only; excluded from Rg.
LIGAND_SELECTION = (
    "segid L and resname LIG"
)

# Verified for PETase21 + PET dimer.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46

# Analyze every frame by default.
# If STEP > 1, the final frame is still explicitly included.
STEP = 1

TOTAL_PLOTTED_TIME_NS = (
    500.0
)

TIME_TICKS_NS = np.arange(
    0.0,
    501.0,
    100.0,
)

UNWRAP_PROTEIN = True


# =============================================================================
# GRAPH STYLE
# =============================================================================

GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"

# Original Rg line color retained.
RG_LINE_COLOR = "#808080"

# Companion GraphPad-format palette retained for consistency/provenance.
GRAPH_COLORS = [
    "#58094F",
    "#841E62",
    "#F3AA20",
    "#2A445E",
    "#346B6D",
]


OUTPUT_TXT = (
    OUTPUT_DIR
    / "Rg_PETase21_PET_dimer_500ns.txt"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "Rg_PETase21_PET_dimer_500ns.csv"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "Rg_PETase21_PET_dimer_500ns.png"
)

OUTPUT_SQUARE_PNG = (
    OUTPUT_DIR
    / "Rg_PETase21_PET_dimer_500ns_square.png"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "Rg_PETase21_PET_dimer_500ns_summary.txt"
)


def set_plot_style():
    """Apply the established GraphPad-like style."""
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


def style_graphpad_axis(
    axis,
):
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

    for label in [
        *axis.get_xticklabels(),
        *axis.get_yticklabels(),
    ]:
        label.set_fontweight(
            "bold"
        )


def save_rg_plot(
    times_ns,
    rg_values,
    output_path,
    *,
    square,
):
    """
    Save Rg plot with exact axes size:

        wide   = 7.62 x 5.08 cm
        square = 5.08 x 5.08 cm

    Margins are outside the requested plotting area.
    """
    set_plot_style()

    plot_w_in = (
        5.08 if square else 7.62
    ) / 2.54

    plot_h_in = (
        5.08 / 2.54
    )

    left = 0.85
    bottom = 0.65
    right = 0.28
    top = 0.54

    fig_w = (
        left
        + plot_w_in
        + right
    )

    fig_h = (
        bottom
        + plot_h_in
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
        plot_w_in / fig_w,
        plot_h_in / fig_h,
    ])

    style_graphpad_axis(
        axis
    )

    axis.plot(
        times_ns,
        rg_values,
        color=RG_LINE_COLOR,
        linestyle="-",
        linewidth=1.35,
    )

    axis.set_xlabel(
        "Time (ns; rescaled)"
    )

    axis.set_ylabel(
        "Radius of gyration (Å)"
    )

    axis.set_title(
        "PETase21 radius of gyration"
    )

    axis.set_xlim(
        0.0,
        TOTAL_PLOTTED_TIME_NS,
    )

    axis.set_xticks(
        TIME_TICKS_NS
    )

    axis.margins(
        y=0.06
    )

    if square:
        axis.tick_params(
            axis="x",
            labelsize=8,
        )

    # Do not use bbox_inches="tight"; this preserves the exact axes geometry.
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        figure
    )


# =============================================================================
# PBC / TRAJECTORY HELPERS
# =============================================================================

def valid_box(
    dimensions,
):
    """
    Validate DCD unit-cell dimensions.

    Lengths must be finite and >0. Angles must be finite and in (0,180).
    """
    if dimensions is None:
        return False

    box = np.asarray(
        dimensions,
        dtype=float,
    )

    if box.size < 6:
        return False

    if not np.all(
        np.isfinite(
            box[:6]
        )
    ):
        return False

    if not np.all(
        box[:3] > 0.0
    ):
        return False

    if not np.all(
        (
            box[3:6] > 0.0
        )
        & (
            box[3:6] < 180.0
        )
    ):
        return False

    return True


def build_dcd_frame_map(
    dcd_files,
):
    """
    Build source-DCD provenance map for the selected trajectory.
    """
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
                str(
                    dcd_path
                ),
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
    """
    Map one global frame to its source DCD/local frame.
    """
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

    if (
        not isinstance(
            STEP,
            int,
        )
        or STEP < 1
    ):
        raise ValueError(
            "STEP must be an integer >= 1."
        )

    if not TOPOLOGY.is_file():
        raise FileNotFoundError(
            "Required topology not found:\n"
            f"{TOPOLOGY}"
        )

    if not DCD_FILES:
        raise FileNotFoundError(
            "No DCD files were discovered."
        )

    for path in DCD_FILES:

        if not path.is_file():
            raise FileNotFoundError(
                "Required DCD file not found:\n"
                f"{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 78
    )

    print(
        "PETase21 radius of gyration | PET dimer-bound complex"
    )

    print(
        "Forced display timeline = 0–500 ns"
    )

    print(
        "=" * 78
    )

    print(
        f"\nPSF:\n{TOPOLOGY}"
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
        "(provenance only; not used for Rg):"
    )

    for energy_log in ENERGY_LOGS:
        print(
            f"  {energy_log}"
        )

    # -------------------------------------------------------------------------
    # Load topology + trajectory
    # -------------------------------------------------------------------------

    universe = mda.Universe(
        str(
            TOPOLOGY
        ),
        *(
            str(path)
            for path
            in DCD_FILES
        ),
    )

    protein = universe.select_atoms(
        PROTEIN_SELECTION
    )

    ligand = universe.select_atoms(
        LIGAND_SELECTION
    )

    if protein.n_atoms == 0:
        raise ValueError(
            "Empty PETase21 selection:\n"
            f"{PROTEIN_SELECTION}\n"
            f"Available segids: "
            f"{sorted(set(universe.atoms.segids))}"
        )

    if ligand.n_atoms == 0:
        raise ValueError(
            "Empty PET dimer selection:\n"
            f"{LIGAND_SELECTION}\n"
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
            f"but found {ligand.n_atoms}. Check ionized.psf / LIGAND_SELECTION."
        )

    if np.intersect1d(
        protein.indices,
        ligand.indices,
    ).size:
        raise ValueError(
            "Protein and ligand selections overlap."
        )

    masses = protein.masses.astype(
        np.float64
    )

    if (
        not np.all(
            np.isfinite(
                masses
            )
        )
        or np.any(
            masses < 0
        )
        or masses.sum() <= 0
    ):
        raise ValueError(
            "Invalid PETase21 atomic masses in PSF."
        )

    total_mass = float(
        masses.sum()
    )

    n_frames = len(
        universe.trajectory
    )

    if n_frames < 2:
        raise ValueError(
            "The trajectory must have at least 2 frames."
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

    if mapped_total != n_frames:
        raise RuntimeError(
            f"Frame-map total ({mapped_total}) differs from "
            f"selected trajectory total ({n_frames})."
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
    # Sampling
    # -------------------------------------------------------------------------

    indices = list(
        range(
            0,
            n_frames,
            STEP,
        )
    )

    if indices[-1] != (
        n_frames - 1
    ):
        indices.append(
            n_frames - 1
        )

    frames = []
    times_ns = []
    rg_values = []

    source_dcd_indices = []
    source_local_frames = []
    source_dcd_files = []

    missing_box_warned = False

    print(
        f"\nPETase21: "
        f"{protein.n_atoms:,} atoms; "
        f"{protein.n_residues:,} residues"
    )

    print(
        f"PET dimer "
        f"(excluded from Rg): "
        f"{ligand.n_atoms:,} atoms"
    )

    print(
        f"DCD frames: "
        f"{n_frames:,}; "
        f"sampled: "
        f"{len(indices):,}"
    )

    # -------------------------------------------------------------------------
    # Frame loop
    # -------------------------------------------------------------------------

    for sampled_index, frame in enumerate(
        indices
    ):
        timestep = universe.trajectory[
            frame
        ]

        if UNWRAP_PROTEIN:

            if valid_box(
                timestep.dimensions
            ):
                try:
                    protein.unwrap(
                        compound="fragments",
                        reference="com",
                        inplace=True,
                    )

                except (
                    NoDataError,
                    ValueError,
                ) as exc:
                    raise RuntimeError(
                        "Cannot reconstruct PETase21 across PBC. "
                        "Inspect the PSF bond topology and DCD unit-cell "
                        "dimensions."
                    ) from exc

            elif not missing_box_warned:

                warnings.warn(
                    "DCD contains no valid periodic box in at least one "
                    "frame. PETase21 was not unwrapped in those frames; "
                    "inspect Rg for possible PBC artifacts.",
                    stacklevel=2,
                )

                missing_box_warned = (
                    True
                )

        # Mass-weighted Rg:
        # sqrt[sum_i m_i |r_i-r_COM|^2 / sum_i m_i]
        coordinates = (
            protein.positions.astype(
                np.float64
            )
        )

        center = (
            np.sum(
                coordinates
                * masses[
                    :,
                    None,
                ],
                axis=0,
            )
            / total_mass
        )

        squared_displacements = np.sum(
            (
                coordinates
                - center
            ) ** 2,
            axis=1,
        )

        rg = float(
            np.sqrt(
                np.dot(
                    masses,
                    squared_displacements,
                )
                / total_mass
            )
        )

        if not np.isfinite(
            rg
        ):
            raise RuntimeError(
                "Non-finite Rg at trajectory frame "
                f"{frame}."
            )

        forced_ns = (
            float(
                frame
            )
            * TOTAL_PLOTTED_TIME_NS
            / float(
                n_frames - 1
            )
        )

        (
            source_dcd_index,
            source_dcd_file,
            source_local_frame,
        ) = locate_source_frame(
            frame,
            dcd_bounds,
        )

        frames.append(
            int(
                frame
            )
        )

        times_ns.append(
            float(
                forced_ns
            )
        )

        rg_values.append(
            rg
        )

        source_dcd_indices.append(
            int(
                source_dcd_index
            )
        )

        source_local_frames.append(
            int(
                source_local_frame
            )
        )

        source_dcd_files.append(
            source_dcd_file
        )

        if (
            sampled_index % 100 == 0
            or frame == n_frames - 1
        ):
            print(
                f"Frame "
                f"{frame + 1:,}/"
                f"{n_frames:,} | "
                f"rescaled "
                f"{forced_ns:.3f} ns | "
                f"Rg "
                f"{rg:.3f} Å"
            )

    # -------------------------------------------------------------------------
    # Validate forced time axis
    # -------------------------------------------------------------------------

    if not (
        np.isclose(
            times_ns[0],
            0.0,
        )
        and np.isclose(
            times_ns[-1],
            TOTAL_PLOTTED_TIME_NS,
        )
    ):
        raise RuntimeError(
            "The 0–500 ns rescaled time axis failed validation."
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR
        / "time_axis_method.txt"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Timeline: global frame index "
            "normalized to 0-500 ns.\n"
        )

        handle.write(
            "Formula: time_ns = global_frame / "
            "(total_DCD_frames - 1) * 500.\n"
        )

        handle.write(
            "This is a presentation/export rescaling and does not "
            "independently verify physical MD duration.\n"
        )

        handle.write(
            f"Combined trajectory frames: {n_frames}\n"
        )

        handle.write(
            f"Sampling step in frames: {STEP}\n"
        )

        handle.write(
            f"Topology: {TOPOLOGY}\n"
        )

        handle.write(
            "Trajectory folder:\n"
        )

        for run_dir in TRAJECTORY_DIRS:
            handle.write(
                f"  {run_dir}\n"
            )

        for index, path in enumerate(
            DCD_FILES,
            start=1,
        ):
            handle.write(
                f"DCD {index}: {path}\n"
            )

        handle.write(
            "Energy logs (provenance only; not used in Rg):\n"
        )

        for energy_log in ENERGY_LOGS:
            handle.write(
                f"  {energy_log}\n"
            )

    # -------------------------------------------------------------------------
    # Tabular outputs
    # -------------------------------------------------------------------------

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Global_Frame\t"
            "Time_ns_rescaled\t"
            "Radius_of_Gyration_A\t"
            "Source_DCD_Index\t"
            "Source_Local_Frame\t"
            "Source_DCD_File\n"
        )

        for (
            frame,
            time_ns,
            rg,
            source_dcd_index,
            source_local_frame,
            source_dcd_file,
        ) in zip(
            frames,
            times_ns,
            rg_values,
            source_dcd_indices,
            source_local_frames,
            source_dcd_files,
        ):
            handle.write(
                f"{frame}\t"
                f"{time_ns:.6f}\t"
                f"{rg:.6f}\t"
                f"{source_dcd_index}\t"
                f"{source_local_frame}\t"
                f"{source_dcd_file}\n"
            )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
    ) as handle:

        handle.write(
            "global_frame,"
            "time_ns_rescaled,"
            "Rg_A,"
            "source_dcd_index,"
            "source_local_frame,"
            "source_dcd_file\n"
        )

        for (
            frame,
            time_ns,
            rg,
            source_dcd_index,
            source_local_frame,
            source_dcd_file,
        ) in zip(
            frames,
            times_ns,
            rg_values,
            source_dcd_indices,
            source_local_frames,
            source_dcd_files,
        ):
            safe_file = str(
                source_dcd_file
            ).replace(
                '"',
                '""',
            )

            handle.write(
                f"{frame},"
                f"{time_ns:.6f},"
                f"{rg:.6f},"
                f"{source_dcd_index},"
                f"{source_local_frame},"
                f"\"{safe_file}\"\n"
            )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    save_rg_plot(
        times_ns,
        rg_values,
        OUTPUT_PNG,
        square=False,
    )

    save_rg_plot(
        times_ns,
        rg_values,
        OUTPUT_SQUARE_PNG,
        square=True,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    values = np.asarray(
        rg_values,
        dtype=float,
    )

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "PETase21 radius of gyration "
            "(PET dimer excluded)\n"
        )

        handle.write(
            "=" * 70
            + "\n\n"
        )

        handle.write(
            f"Topology: {TOPOLOGY}\n"
        )

        handle.write(
            "Trajectory directories "
            "(chronological order):\n"
        )

        for run_dir in TRAJECTORY_DIRS:
            handle.write(
                f"  {run_dir}\n"
            )

        handle.write(
            "\nDCD files "
            "(chained order):\n"
        )

        for index, path in enumerate(
            DCD_FILES,
            start=1,
        ):
            handle.write(
                f"  DCD {index}: {path}\n"
            )

        handle.write(
            "\nEnergy logs "
            "(provenance only; not used for Rg):\n"
        )

        for energy_log in ENERGY_LOGS:
            handle.write(
                f"  {energy_log}\n"
            )

        handle.write(
            f"\nProtein selection: "
            f"{PROTEIN_SELECTION}\n"
        )

        handle.write(
            f"Ligand selection "
            f"(validation only, excluded): "
            f"{LIGAND_SELECTION}\n"
        )

        handle.write(
            f"Protein unwrapping across PBC: "
            f"{UNWRAP_PROTEIN}\n"
        )

        handle.write(
            f"DCD frames: "
            f"{n_frames}\n"
        )

        handle.write(
            f"Sampled frames: "
            f"{len(frames)}\n"
        )

        handle.write(
            "Time axis: FORCED 0–500 ns; "
            "NOT verified physical MD time\n"
        )

        handle.write(
            "Time formula: global_frame / "
            "(total_DCD_frames - 1) * 500 ns\n"
        )

        handle.write(
            "Rg definition: mass-weighted PETase21 protein Rg only; "
            "PET dimer/solvent excluded.\n"
        )

        handle.write(
            f"Mean ± SD Rg: "
            f"{values.mean():.6f} ± "
            f"{values.std(ddof=1):.6f} Å\n"
        )

        handle.write(
            f"Median Rg: "
            f"{np.median(values):.6f} Å\n"
        )

        handle.write(
            f"Minimum Rg: "
            f"{values.min():.6f} Å\n"
        )

        handle.write(
            f"Maximum Rg: "
            f"{values.max():.6f} Å\n"
        )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Rg ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nRg mean ± SD: "
        f"{values.mean():.3f} ± "
        f"{values.std(ddof=1):.3f} Å"
    )

    print(
        f"Median Rg: "
        f"{np.median(values):.3f} Å"
    )

    print(
        f"Rescaled times: "
        f"{times_ns[0]:.3f}–"
        f"{times_ns[-1]:.3f} ns"
    )

    print(
        "NOTE: 0–500 ns is a combined-frame-index rescaling; "
        "it does not independently establish 500 ns of physical MD."
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )

    for path in [
        OUTPUT_TXT,
        OUTPUT_CSV,
        OUTPUT_PNG,
        OUTPUT_SQUARE_PNG,
        OUTPUT_SUMMARY,
        metadata_path,
    ]:
        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()
