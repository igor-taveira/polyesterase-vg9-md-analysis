# -*- coding: utf-8 -*-
"""PETase21–PET dimer C-alpha dynamic cross-correlation analysis.

Uses the PETase21–PET dimer ionized.psf and the root-level ionized.dcd only.
The NAMD energy log is validated and recorded as reproducibility metadata but
is not used to calculate DCCM, which is derived from aligned C-alpha
coordinate fluctuations.

Every selected trajectory frame is mapped linearly to 0–500 ns FOR DISPLAY
ONLY. This mapping does not make the underlying MD duration 500 ns. Verify
NAMD timesteps, output stride and trajectory continuity before interpreting
the split windows as physical 250-ns intervals.

PETase21 is backbone-aligned to the first selected frame and its C-alpha 3D
displacement cross-correlations are calculated. PET dimer, solvent and ions
are excluded from the DCCM matrix.

Outputs: full/first-half/second-half/second-minus-first DCCMs (NPY/CSV/PNG),
residue map, strong-pair table, mean absolute long-range correlation profiles,
audit CSV, and analysis metadata. Retains the original matplotlib viridis
palette, Josefin Sans/Arial/DejaVu Sans fallback, and GraphPad-like geometry.

Requires Python, numpy, pandas, matplotlib and MDAnalysis; no GPU needed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import font_manager

import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDReader
from MDAnalysis.exceptions import NoDataError


# -----------------------------------------------------------------------------
# INPUTS — PETase21 + PET dimer
# -----------------------------------------------------------------------------
BASE_DIR = Path(
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella\dados\docking-e-dinamica"
    r"\petase21-PET-02-dimer-coo-and-oh-ends"
)

PSF_FILE = BASE_DIR / "ionized.psf"
DCD_FILE = BASE_DIR / "ionized.dcd"
ENERGY_LOG_FILE = BASE_DIR / "energia-petase21-pet-dimer-coo-and-oh-ends.log"


def discover_dcd_files():
    """Use only the verified root-level PETase21/PET-dimer trajectory."""
    if not DCD_FILE.is_file():
        raise FileNotFoundError(
            "PETase21–PET dimer trajectory was not found:\\n"
            f"{DCD_FILE}\\nCheck its actual name/path in PowerShell."
        )
    return [DCD_FILE]


OUTPUT_DIR = BASE_DIR / "DCCM_PETase21_PET_dimer_viridis_500ns"

# Analyze every fifth combined frame; include both ends of every DCD boundary.
# Reduce STEP for finer temporal sampling, at the expense of compute time.
STEP = 5
RECEPTOR_SELECTION = "segid AP1 and protein"
BACKBONE_SELECTION = f"({RECEPTOR_SELECTION}) and backbone"
CA_SELECTION = f"({RECEPTOR_SELECTION}) and name CA"
LIGAND_SELECTION = "segid L and resname LIG"  # PET dimer
UNWRAP_PROTEIN = True
TOTAL_PLOTTED_TIME_NS = 500.0  # VISUALIZATION, not measured MD duration

# A sequence-neighbor exclusion for the separate summary/profile (not DCCM).
MIN_SEQUENCE_SEPARATION = 5
PAIR_ABS_CORRELATION_CUTOFF = 0.70
MAX_EXPORTED_PAIRS = 1000
CORRELATION_BLOCK_FRAMES = 128               # memory-conservative covariance
MIN_FRAMES_PER_WINDOW = 3

# -----------------------------------------------------------------------------
# GRAPHPAD-LIKE GEOMETRY AND THE REQUESTED ORIGINAL MATPLOTLIB VIRIDIS
# -1 (purple) -> 0 (teal) -> +1 (yellow) in the DCCM heatmaps.
# Viridis is sequential, NOT a zero-centered diverging colour map.
# -----------------------------------------------------------------------------
GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"
DCCM_CMAP = plt.get_cmap("viridis")
DELTA_CMAP = DCCM_CMAP
PROFILE_COLOR = DCCM_CMAP(0.72)  # viridis green for the auxiliary profile


def set_plot_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in ("Josefin Sans", "Arial", "DejaVu Sans")
                   if f in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": chosen,
        "font.weight": "bold",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.labelweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def exact_axes(*, square: bool, colorbar: bool = False):
    """Figure with EXACT inner axis of 5.08x5.08 or 7.62x5.08 cm.

    The colorbar is outside that plotting area and does not resize it.
    """
    ax_w = (5.08 if square else 7.62) / 2.54
    ax_h = 5.08 / 2.54
    left, bottom, top = 0.72, 0.67, 0.52
    right = 0.93 if colorbar else 0.30
    fig_w, fig_h = left + ax_w + right, bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       ax_w / fig_w, ax_h / fig_h])
    cbax = None
    if colorbar:
        cbax = fig.add_axes([(left + ax_w + 0.12) / fig_w,
                             bottom / fig_h, 0.115 / fig_w, ax_h / fig_h])
    ax.set_facecolor(GRAPH_BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(GRAPH_FRAME_COLOR)
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    ax.tick_params(axis="both", which="major", direction="in", top=True,
                   right=True, length=4, width=1.15, color=GRAPH_FRAME_COLOR)
    return fig, ax, cbax


def save_map(matrix, title: str, path: Path, *, difference=False):
    set_plot_style()
    fig, ax, cbax = exact_axes(square=True, colorbar=True)
    norm = Normalize(vmin=-2.0, vmax=2.0) if difference else Normalize(vmin=-1.0, vmax=1.0)
    im = ax.imshow(matrix, cmap=DELTA_CMAP if difference else DCCM_CMAP,
                   norm=norm, origin="upper", interpolation="nearest",
                   aspect="equal", rasterized=True)
    n = matrix.shape[0]
    positions = np.unique(np.rint(np.linspace(0, n - 1, min(6, n))).astype(int))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels((positions + 1).astype(str))
    ax.set_yticklabels((positions + 1).astype(str))
    ax.set_xlabel("PETase21 residue index (Cα)")
    ax.set_ylabel("PETase21 residue index (Cα)")
    ax.set_title(title, fontsize=9, weight="bold")
    cbar = fig.colorbar(im, cax=cbax)
    cbar.set_ticks([-2, -1, 0, 1, 2] if difference else [-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(direction="in", labelsize=7, length=3)
    cbar.set_label("Δ correlation" if difference else "Correlation", size=8,
                   weight="bold", labelpad=4)
    fig.savefig(path, dpi=300)  # no bbox_inches='tight': axes dimensions remain exact
    plt.close(fig)


def save_profile(values: np.ndarray, path: Path, *, square: bool):
    set_plot_style()
    fig, ax, _ = exact_axes(square=square)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRAPH_GRID_COLOR, linestyle=(0, (4, 4)),
            linewidth=1.35, axis="both")
    ax.plot(np.arange(1, len(values) + 1), values, color=PROFILE_COLOR,
            linewidth=1.20)
    ax.set_xlabel("PETase21 residue index (Cα)")
    ax.set_ylabel("Mean |DCCM| (long-range)")
    ax.set_title("PETase21 mean absolute DCCM", fontsize=10)
    ax.set_xlim(1, max(2, len(values)))
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", which="major", top=True, right=True,
                   direction="in", length=5, width=1.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def is_valid_box(dimensions):
    if dimensions is None:
        return False
    box = np.asarray(dimensions, dtype=float)
    return (box.size >= 6 and np.all(np.isfinite(box[:6]))
            and np.all(box[:3] > 0) and np.all((box[3:6] > 0) & (box[3:6] < 180)))


def align_ca_to_reference(backbone_coords, ca_coords,
                          reference_backbone, reference_center):
    """Backbone Kabsch fit, then apply SAME rigid transformation to C-alpha.

    Coordinates are row vectors: aligned_CA = (CA - mobile_center) @ R + ref_center.
    """
    mobile_center = backbone_coords.mean(axis=0)
    x = backbone_coords - mobile_center
    y = reference_backbone - reference_center
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rot = u @ correction @ vt
    return (ca_coords - mobile_center) @ rot + reference_center


def compute_dccm(aligned_ca: np.ndarray):
    """3-D displacement DCCM; blockwise GEMM avoids N x N x frames arrays."""
    n_frames, n_residues, dims = aligned_ca.shape
    if n_frames < MIN_FRAMES_PER_WINDOW or dims != 3:
        raise ValueError("At least 3 aligned 3-D frames are required per DCCM.")

    # float64 centering: prevents cancellation on absolute coordinate scales.
    mean = np.mean(aligned_ca, axis=0, dtype=np.float64)
    numerator = np.zeros((n_residues, n_residues), dtype=np.float64)
    for begin in range(0, n_frames, CORRELATION_BLOCK_FRAMES):
        xyz = aligned_ca[begin:begin + CORRELATION_BLOCK_FRAMES].astype(np.float64)
        xyz -= mean[None, :, :]
        for axis in range(3):
            fluctuations = xyz[:, :, axis]
            numerator += fluctuations.T @ fluctuations

    variance = np.diag(numerator).copy()
    denom = np.sqrt(np.outer(variance, variance))
    output = np.zeros_like(numerator)
    np.divide(numerator, denom, out=output, where=denom > 1e-20)
    np.clip(output, -1.0, 1.0, out=output)
    output = 0.5 * (output + output.T)  # guard floating-point roundoff
    np.fill_diagonal(output, np.where(variance > 1e-20, 1.0, 0.0))
    return output


def save_matrix(mat: np.ndarray, key: str, residue_labels: list[str]):
    np.save(OUTPUT_DIR / f"{key}.npy", mat.astype(np.float32))
    pd.DataFrame(mat, index=residue_labels, columns=residue_labels).to_csv(
        OUTPUT_DIR / f"{key}.csv", encoding="utf-8-sig", float_format="%.6f")


def dcd_counts(dcd_files):
    counts = []
    for f in dcd_files:
        with DCDReader(str(f)) as reader:
            counts.append(len(reader))
    return counts


def main():
    DCD_FILES = discover_dcd_files()
    if not isinstance(STEP, int) or STEP < 1:
        raise ValueError("STEP must be an integer >= 1.")
    if len(DCD_FILES) < 1:
        raise ValueError("At least one DCD file must be supplied.")
    if len({str(path).casefold() for path in DCD_FILES}) != len(DCD_FILES):
        raise ValueError("All DCD paths must be distinct.")
    for path in [PSF_FILE, *DCD_FILES]:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found:\n{path}")
    if not ENERGY_LOG_FILE.is_file():
        raise FileNotFoundError(
            "Requested PETase21–PET dimer energy log was not found:\n"
            f"{ENERGY_LOG_FILE}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = dcd_counts(DCD_FILES)
    if any(n < 1 for n in counts):
        raise ValueError(f"All DCDs must contain at least one frame: {counts}")
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    u = mda.Universe(str(PSF_FILE), *(str(f) for f in DCD_FILES))
    total_frames = len(u.trajectory)
    if sum(counts) != total_frames:
        raise RuntimeError("DCD frame counts are inconsistent with chained trajectory.")

    protein = u.select_atoms(RECEPTOR_SELECTION)
    backbone = u.select_atoms(BACKBONE_SELECTION)
    ca = u.select_atoms(CA_SELECTION)
    ligand = u.select_atoms(LIGAND_SELECTION)
    if protein.n_atoms == 0 or backbone.n_atoms < 3 or ca.n_atoms < 3:
        raise ValueError("PETase21 protein/backbone/C-alpha selection is empty or too small."
                         f" Segments: {list(u.segments.segids)}")
    if ligand.n_atoms == 0:
        raise ValueError("PET dimer (resname LIG) not found in ionized.psf. "
                         "Inspect ligand resname/segid before changing LIGAND_SELECTION."
                         f" Available resnames: {sorted(set(u.atoms.resnames))}")
    if ligand.n_residues != 1:
        raise ValueError(f"Expected exactly one PET dimer residue (LIG), "
                         f"found {ligand.n_residues}; refine LIGAND_SELECTION.")
    if ligand.n_atoms != 46:
        warnings.warn(
            f"Expected 46 atoms for the uploaded PET dimer topology, but selected "
            f"{ligand.n_atoms}. Confirm that ionized.psf belongs to this system."
        )
    if np.intersect1d(protein.indices, ligand.indices).size:
        raise ValueError("Protein and ligand selections overlap.")
    if ca.n_residues != ca.n_atoms:
        raise ValueError("Expected exactly one C-alpha per selected protein residue.")
    if total_frames < 2:
        raise ValueError("The combined trajectory must contain at least two frames.")

    # Uniform sampling over all concatenated DCDs. Always include both
    # sides of every restart boundary and the very last frame.
    sampled = set(range(0, total_frames, STEP))
    sampled.add(total_frames - 1)
    for boundary in offsets[1:-1]:
        sampled.add(int(boundary - 1))
        sampled.add(int(boundary))
    sampled = sorted(sampled)
    if len(sampled) < 2 * MIN_FRAMES_PER_WINDOW:
        raise ValueError("Too few sampled frames for both split-window DCCMs. Decrease STEP.")

    print("=" * 76)
    print("PETase21 + PET dimer | C-alpha DCCM | selected DCDs | viridis")
    print(f"Topology: {PSF_FILE}")
    for i, (file, n) in enumerate(zip(DCD_FILES, counts), 1):
        print(f"DCD {i} ({n:,} frames): {file}")
    print(f"Combined: {total_frames:,} frames | sampled: {len(sampled):,} | STEP={STEP}")
    print(f"Protein: {protein.n_atoms:,} atoms / {ca.n_atoms:,} C-alpha residues")
    print(f"PET dimer validation: {ligand.n_atoms:,} atoms / {ligand.n_residues} residue")
    print(f"Energy log (audit only): {ENERGY_LOG_FILE}")
    print("WARNING: 0-500 ns is a forced DISPLAY time axis, NOT verified physical ns.")
    print("CAUTION: verify no overlapping restart steps/duplicate frames at DCD boundaries.")
    print("CAUTION: confirm that all selected DCDs contain the intended production frames before interpretation.")
    print("=" * 76)

    # Fix reference at first frame of selected DCD 1; topology atom order must be the
    # same throughout all DCDs (MDAnalysis verifies atom counts).
    first = u.trajectory[0]
    if UNWRAP_PROTEIN and is_valid_box(first.dimensions):
        protein.unwrap(compound="fragments", reference="com", inplace=True)
    ref_bb = backbone.positions.astype(np.float64).copy()
    ref_center = ref_bb.mean(axis=0)

    coords = np.empty((len(sampled), ca.n_atoms, 3), dtype=np.float32)
    global_frames = np.asarray(sampled, dtype=np.int64)
    times = global_frames.astype(np.float64) * TOTAL_PLOTTED_TIME_NS / (total_frames - 1)
    missing_box_warned = False
    for i, global_frame in enumerate(sampled):
        ts = u.trajectory[global_frame]
        if UNWRAP_PROTEIN:
            if is_valid_box(ts.dimensions):
                try:
                    protein.unwrap(compound="fragments", reference="com", inplace=True)
                except (NoDataError, ValueError) as exc:
                    raise RuntimeError("Cannot reconstruct PETase21 across PBC; check PSF bonds. ") from exc
            elif not missing_box_warned:
                warnings.warn("Some frames have no valid periodic cell: protein could not "
                              "be unwrapped there. Inspect DCCM for PBC artifacts.")
                missing_box_warned = True
        coords[i] = align_ca_to_reference(
            backbone.positions.astype(np.float64),
            ca.positions.astype(np.float64), ref_bb, ref_center,
        ).astype(np.float32)
        if i % 100 == 0 or i == len(sampled) - 1:
            print(f"Processed {i + 1:,}/{len(sampled):,} sampled frames "
                  f"(combined frame {global_frame:,}; display {times[i]:.2f} ns)")

    # The split is by the FORCED 0--500 display axis, and each window is
    # centered about its own mean. Not a physical 250-ns window unless verified.
    in_first = times < TOTAL_PLOTTED_TIME_NS / 2
    in_second = ~in_first
    if min(np.count_nonzero(in_first), np.count_nonzero(in_second)) < MIN_FRAMES_PER_WINDOW:
        raise ValueError("A display-time half has fewer than 3 sampled frames.")

    labels = [f"{a.segid}:{a.resname}:{a.resid}" for a in ca]
    residue_index = pd.DataFrame({
        "matrix_index_1based": np.arange(1, ca.n_atoms + 1),
        "segid": ca.segids,
        "resid": ca.resids,
        "resname": ca.resnames,
        "ca_atom_index_0based": ca.indices,
        "matrix_label": labels,
    })
    residue_index.to_csv(OUTPUT_DIR / "PETase21_DCCM_residue_index.csv",
                         index=False, encoding="utf-8-sig")
    dcd_number = np.searchsorted(offsets[1:], global_frames, side="right") + 1
    local_frames = global_frames - offsets[dcd_number - 1]
    pd.DataFrame({"global_frame_0based": global_frames,
                  "DCD_number_1based": dcd_number,
                  "DCD_local_frame_0based": local_frames,
                  "DCD_path": [str(DCD_FILES[k - 1]) for k in dcd_number],
                  "display_time_ns_rescaled": times}).to_csv(
        OUTPUT_DIR / "PETase21_DCCM_sampled_frames.csv", index=False,
        encoding="utf-8-sig")

    print("Computing full and split-window cross-correlation matrices...")
    full = compute_dccm(coords)
    first_half = compute_dccm(coords[in_first])
    second_half = compute_dccm(coords[in_second])
    delta = second_half - first_half
    for key, matrix, title, is_delta in [
        ("PETase21_DCCM_full", full, "PETase21 DCCM | 0-500 ns (rescaled)", False),
        ("PETase21_DCCM_first_half", first_half,
         "PETase21 DCCM | 0-250 ns (rescaled)", False),
        ("PETase21_DCCM_second_half", second_half,
         "PETase21 DCCM | 250-500 ns (rescaled)", False),
        ("PETase21_DCCM_second_minus_first", delta,
         "PETase21 ΔDCCM | second − first", True),
    ]:
        save_matrix(matrix, key, labels)
        save_map(matrix, title, OUTPUT_DIR / f"{key}_square.png", difference=is_delta)

    # Auxiliary per-residue mean absolute long-range correlations: exclude self
    # and near-sequence neighbours, which often correlate because of locality.
    ix = np.arange(ca.n_atoms)
    far = np.abs(ix[:, None] - ix[None, :]) >= MIN_SEQUENCE_SEPARATION
    n_far = far.sum(axis=1)
    mean_abs = np.divide((np.abs(full) * far).sum(axis=1), n_far,
                         out=np.zeros(ca.n_atoms, dtype=float), where=n_far > 0)
    residue_index.assign(mean_abs_long_range_DCCM=mean_abs).to_csv(
        OUTPUT_DIR / "PETase21_DCCM_mean_absolute_by_residue.csv",
        index=False, encoding="utf-8-sig")
    save_profile(mean_abs, OUTPUT_DIR / "PETase21_DCCM_profile_wide.png", square=False)
    save_profile(mean_abs, OUTPUT_DIR / "PETase21_DCCM_profile_square.png", square=True)

    # Report up to MAX_EXPORTED_PAIRS strongest nonlocal pairs; do not confuse
    # an arbitrary |C| threshold with a statistically significant relationship.
    pair_i, pair_j = np.where(np.triu(far &
        (np.abs(full) >= PAIR_ABS_CORRELATION_CUTOFF), k=1))
    order = np.argsort(-np.abs(full[pair_i, pair_j]))[:MAX_EXPORTED_PAIRS]
    pd.DataFrame({
        "residue_i_index_1based": pair_i[order] + 1,
        "residue_j_index_1based": pair_j[order] + 1,
        "residue_i": np.asarray(labels)[pair_i[order]],
        "residue_j": np.asarray(labels)[pair_j[order]],
        "DCCM_coefficient": full[pair_i[order], pair_j[order]],
        "sequence_index_separation": pair_j[order] - pair_i[order],
    }).to_csv(OUTPUT_DIR / "PETase21_DCCM_strong_pairs.csv",
              index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "PETase21_DCCM_analysis_metadata.txt").open(
            "w", encoding="utf-8") as f:
        f.write("PETase21/PET dimer: C-alpha residue-level DCCM\n")
        f.write("Definition: correlation of aligned 3-D C-alpha displacements "
                "from a per-window mean (dot-product normalized by variances).\n")
        f.write(f"PSF: {PSF_FILE}\n")
        for i, (path, count) in enumerate(zip(DCD_FILES, counts), 1):
            f.write(f"DCD {i}: {path} ({count} frames)\n")
        f.write(f"DCD offsets (0-based, including final end): {offsets.tolist()}\n")
        f.write(f"Combined frames: {total_frames}; sampled frames: {len(sampled)}; "
                f"STEP: {STEP}\n")
        f.write(f"PETase21 selection: {RECEPTOR_SELECTION}; backbone: "
                f"{BACKBONE_SELECTION}; C-alpha: {CA_SELECTION}\n")
        f.write(f"C-alpha residues: {ca.n_atoms}; reference: first DCD1 frame; "
                f"protein PBC unwrap: {UNWRAP_PROTEIN}\n")
        f.write(f"Display time: frame * {TOTAL_PLOTTED_TIME_NS} / "
                "(combined_frames - 1), for combined global frame.\n")
        f.write("IMPORTANT: 0-500 ns is rescaled, not validated physical MD time.\n")
        f.write("Trajectory layout: one verified root-level ionized.dcd.\n")
        f.write(f"Energy log (audit only; not used in DCCM): {ENERGY_LOG_FILE}\n")
        f.write("Check NAMD logs, output stride, chronology, independent replicates and overlapping restarts "
                "before interpreting 500 ns as physical duration.\n")
        f.write("Viridis (sequential): purple = -1, teal near 0, yellow = +1. "
                "Delta maps use -2 to +2 with the same viridis map.\n")
        f.write("Split-window maps are halves of this RESCALED display axis, "
                "each recentered on its own mean.\n")
        f.write(f"First half sampled frames: {in_first.sum()}; "
                f"second half sampled frames: {in_second.sum()}\n")
        f.write("PET dimer, solvent and ions are excluded from the receptor C-alpha DCCM.\n")
        f.write("Profile/pair sequence separation is C-alpha index difference, "
                "not necessarily the original PSF residue-number difference.\n")
        f.write("Confirm that all selected DCDs contain the intended production frames; "
                "exclude any non-production frames before making production-DCCM claims.\n")
    print("DCCM completed. Outputs:", OUTPUT_DIR)
    print("Combined rescaled endpoints:", times[0], "to", times[-1], "ns")
    assert np.isclose(times[0], 0.0) and np.isclose(times[-1], TOTAL_PLOTTED_TIME_NS)
    print("Matrix shape:", full.shape, "| full diagonal:", np.diag(full).min(),
          "to", np.diag(full).max())


if __name__ == "__main__":
    main()
