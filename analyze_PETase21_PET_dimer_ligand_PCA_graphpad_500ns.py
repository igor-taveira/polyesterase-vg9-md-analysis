# -*- coding: utf-8 -*-

"""
PCA and conformational clustering of PET dimer bound to PETase21.

Default analysis (ANALYSIS_MODE = "receptor_context"):
    1. Reconstructs PETase21 and the ligand across periodic boundaries.
    2. Places the ligand in the periodic image nearest the receptor.
    3. Aligns every frame to the first sampled PETase21 backbone conformation.
    4. Expresses ligand heavy-atom Cartesian coordinates relative to the
       aligned PETase21 center of mass.
    5. Performs PCA without per-coordinate variance scaling.
    6. Chooses a K-means cluster count using the silhouette score.

This default preserves ligand translation, orientation, and internal
conformational changes relative to the receptor while removing whole-system
translation and rotation.

Optional analysis (ANALYSIS_MODE = "internal_conformation"):
    After receptor alignment, ligand heavy atoms are additionally superposed on
    the first sampled ligand structure by the Kabsch algorithm. This removes
    ligand rigid-body motion and focuses the PCA on internal conformational
    changes.

The PETase21/PET-dimer analysis uses the explicitly selected ionized.dcd
trajectory from the system root directory. Frame indices are rescaled to
0--500 ns for display.
This does NOT establish 500 ns of physical MD; verify NAMD log/stride/steps,
and do not combine separate replicas or equilibration with production for
kinetic interpretation.

Required packages:
    numpy, pandas, matplotlib, scikit-learn, MDAnalysis
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# PATHS: PETase21 + PET dimer
# ============================================================

BASE_DIR = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2025_petase21_anna-e-marinella"
    r"\dados\docking-e-dinamica\petase21-PET-02-dimer-coo-and-oh-ends"
)

TOPOLOGY = os.path.join(BASE_DIR, "ionized.psf")
ENERGY_LOG = os.path.join(
    BASE_DIR,
    "energia-petase21-pet-dimer-coo-and-oh-ends.log",
)

# The supplied directory listing contains one trajectory DCD: ionized.dcd.
# Keep it explicit instead of recursively discovering DCDs, preventing future
# accidental inclusion of duplicate, unrelated, or independent trajectories.
TRAJECTORIES = [
    os.path.join(BASE_DIR, "ionized.dcd"),
]

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "PCA_PETase21_PET_dimer_RECEPTOR_CONTEXT_500ns",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# ENERGY_LOG is retained for reproducibility/audit only. Ligand PCA uses
# trajectory coordinates, not NAMD potential/kinetic energy values.

# "receptor_context": retains ligand position/orientation relative to PETase21
# after removing whole-receptor translation and rotation.
# "internal_conformation": additionally fits each ligand frame to the first
# sampled ligand structure, retaining only ligand internal conformational change.
ANALYSIS_MODE = "receptor_context"

# Analyze every frame. Increase to 5 or 10 for deliberate subsampling.
STEP = 1

# Maximum number of principal components retained in output tables.
MAX_PCS = 10

# K-means will test cluster counts from 2 through MAX_CLUSTERS.
MAX_CLUSTERS = 8

# Number of leading PCs used for clustering.
CLUSTER_PCS = 3

# Exploratory one-way PERMANOVA and PERMDISP are calculated in the same PC
# space used by K-means. The smallest attainable permutation p-value is
# 1 / (MULTIVARIATE_PERMUTATIONS + 1).
MULTIVARIATE_PERMUTATIONS = 9999

# Maximum observations used to calculate each silhouette score. Limiting this
# avoids the quadratic memory/time cost of silhouette analysis on long runs.
SILHOUETTE_SAMPLE_SIZE = 3000

RANDOM_STATE = 42

# The production temperature must be confirmed from the NAMD log.
TEMPERATURE_K = None

# Requested displayed duration of the trajectory.
# Time is reconstructed from the COMBINED global frame index across both
# DCDs: first frame of DCD 1 = 0 ns; last frame of the selected DCDs = 500 ns.
# This intentionally overrides DCD-internal timestamps for plots/tables.
# It does not extend the MD simulation or establish its physical duration.
TRAJECTORY_TOTAL_NS = 500.0
TIME_TICKS_NS = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

# ============================================================
# ATOM SELECTIONS
# ============================================================

# Explicit selections verified against the supplied PETase21 + PET dimer ionized.psf.
# PETase21 = segid AP1; PET dimer = one LIG residue in segid L.
RECEPTOR_SEL = "segid AP1 and protein"
RECEPTOR_FIT_SEL = "segid AP1 and protein and backbone"

LIGAND_ALL_SEL = "segid L and resname LIG"

# Heavy atoms only. Hydrogens and any massless LP halogen dummy are excluded.
LIGAND_PCA_SEL = (
    "segid L and resname LIG "
    "and not name H* and not name LP*"
)

# PSF-specific sanity checks for the uploaded PET-dimer system.
EXPECTED_RECEPTOR_RESIDUES = 257
EXPECTED_LIGAND_RESIDUES = 1
EXPECTED_LIGAND_ATOMS = 46
EXPECTED_LIGAND_PCA_HEAVY_ATOMS = 29

# Common explicit-water residue names. Water is not needed for the PCA itself,
# but the full-system representative DCD retains it through original PSF order.
WATER_SEL = (
    "resname TIP3 TIP4 TIP4P TIP5P HOH WAT SOL SPC SPCE"
)

# Display-friendly representative PDB: receptor + ligand only.
# The optional one-frame full-system DCD retains all atoms (including solvent
# and ions) in the original PSF atom order.
REPRESENTATIVE_COMPLEX_SEL = (
    f"({RECEPTOR_SEL}) or ({LIGAND_ALL_SEL})"
)
EXPORT_FULL_SYSTEM_REPRESENTATIVE_DCD = True

# ============================================================
# OUTPUT FILES
# ============================================================

SCORES_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_scores_clusters.csv",
)

VARIANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_explained_variance.csv",
)

LOADINGS_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_loadings.csv",
)

SILHOUETTE_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_silhouette_scores.csv",
)

CLUSTER_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_cluster_summary.csv",
)

REPRESENTATIVES_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_representatives.csv",
)

PERMANOVA_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_PERMANOVA.csv",
)

PERMDISP_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_PERMDISP.csv",
)

PERMDISP_GROUPS_CSV = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_PERMDISP_by_cluster.csv",
)

MULTIVARIATE_TXT = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_PERMANOVA_PERMDISP.txt",
)

SUMMARY_TXT = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_summary.txt",
)

PCA_CLUSTER_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_clusters.png",
)
PCA_CLUSTER_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_clusters_square.png",
)

PCA_TIME_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_colored_by_time.png",
)
PCA_TIME_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_colored_by_time_square.png",
)

SCREE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_scree_plot.png",
)
SCREE_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCA_scree_plot_square.png",
)

PCS_TIME_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCs_vs_time.png",
)
PCS_TIME_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_PCs_vs_time_square.png",
)

CLUSTER_OCCUPANCY_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_cluster_occupancy.png",
)
CLUSTER_OCCUPANCY_SQUARE_PNG = os.path.join(
    OUTPUT_DIR,
    "PET_dimer_ligand_cluster_occupancy_square.png",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def has_valid_box(dimensions):
    if dimensions is None or len(dimensions) < 6:
        return False

    dimensions = np.asarray(dimensions, dtype=float)

    return bool(
        np.all(np.isfinite(dimensions))
        and np.all(dimensions[:3] > 0.0)
    )


def reconstruct_receptor_ligand(
    receptor,
    ligand_all,
    dimensions,
):
    """
    Reconstruct receptor and ligand across PBC and place the ligand in the
    periodic image nearest the receptor center of mass.
    """

    for atom_group, label in (
        (receptor, "receptor"),
        (ligand_all, "ligand"),
    ):
        try:
            atom_group.unwrap(
                compound="fragments",
                reference="com",
                inplace=True,
            )
        except (NoDataError, ValueError) as exc:
            warning_key = f"{label}:{type(exc).__name__}:{exc}"
            if warning_key not in reconstruct_receptor_ligand.warned:
                warnings.warn(
                    f"{label.capitalize()} unwrapping could not be applied; "
                    f"continuing with stored coordinates. Details: {exc}"
                )
                reconstruct_receptor_ligand.warned.add(warning_key)

    receptor_com = receptor.center_of_mass()

    if has_valid_box(dimensions):
        ligand_com = ligand_all.center_of_mass()
        displacement = ligand_com - receptor_com

        minimum_displacement = minimize_vectors(
            displacement.reshape(1, 3),
            dimensions,
        )[0]

        ligand_all.translate(
            minimum_displacement - displacement
        )

    return receptor_com


reconstruct_receptor_ligand.warned = set()


def kabsch_transform(mobile, reference):
    """
    Return the rigid-body Kabsch transform that maps mobile Nx3 coordinates
    onto reference Nx3 coordinates using equal atom weights.
    """

    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if mobile.shape != reference.shape:
        raise ValueError(
            "Kabsch mobile/reference coordinate arrays must have equal shape."
        )

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)

    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center

    covariance = mobile_centered.T @ reference_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)

    rotation = u_matrix @ vt_matrix

    if np.linalg.det(rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix

    return rotation, mobile_center, reference_center


def align_system_to_receptor_reference(
    universe,
    receptor_fit,
    reference_receptor_coordinates,
):
    """
    Align the complete system using the PETase21 backbone as the fitting group.
    Returning the RMSD is useful for diagnostics; PCA itself uses the aligned
    ligand coordinates.
    """

    mobile = receptor_fit.positions.astype(np.float64).copy()

    rotation, mobile_center, reference_center = kabsch_transform(
        mobile,
        reference_receptor_coordinates,
    )

    before = mobile
    after = (
        (before - mobile_center) @ rotation
        + reference_center
    )

    receptor_fit_rmsd = float(
        np.sqrt(np.mean(np.sum(
            (after - reference_receptor_coordinates) ** 2,
            axis=1,
        )))
    )

    all_positions = universe.atoms.positions.astype(np.float64).copy()
    universe.atoms.positions = (
        (all_positions - mobile_center) @ rotation
        + reference_center
    )

    return receptor_fit_rmsd


def write_representative_structures(
    universe,
    receptor,
    receptor_fit,
    reference_receptor_coordinates,
    ligand_all,
    ligand_pca,
    representative_complex,
    representative_index,
    trajectory_frame,
    cluster_number,
    coordinate_matrix,
):
    """
    Export ligand-only and PETase21-ligand structures for one representative.

    The physical frame is reconstructed across PBC, aligned to the first sampled
    PETase21 backbone, and translated so the aligned PETase21 COM is at the origin.
    Therefore, the receptor-ligand PDB is in the same reference frame used for
    receptor-context ligand PCA.

    A one-frame full-system DCD is also written by default and retains original
    PSF atom order.
    """

    file_prefix = os.path.join(
        OUTPUT_DIR,
        f"PET_dimer_cluster_{cluster_number:02d}_representative",
    )

    ligand_only_pdb = f"{file_prefix}_ligand.pdb"
    complex_pdb = f"{file_prefix}_PETase21_ligand.pdb"
    full_system_dcd = f"{file_prefix}_full_system.dcd"

    # Ligand-only PDB contains the exact heavy-atom coordinates supplied to PCA.
    ligand_pca.positions = coordinate_matrix[
        representative_index
    ].reshape(ligand_pca.n_atoms, 3)

    with mda.Writer(
        ligand_only_pdb,
        n_atoms=ligand_pca.n_atoms,
    ) as writer:
        writer.write(ligand_pca)

    # Reload the original physical trajectory frame.
    universe.trajectory[trajectory_frame]

    reconstruct_receptor_ligand(
        receptor,
        ligand_all,
        universe.trajectory.ts.dimensions,
    )

    align_system_to_receptor_reference(
        universe,
        receptor_fit,
        reference_receptor_coordinates,
    )

    receptor_com_aligned = receptor.center_of_mass()

    physical_ligand_coordinates_relative = (
        ligand_pca.positions.astype(np.float64).copy()
        - receptor_com_aligned
    )

    if ANALYSIS_MODE == "receptor_context":
        pca_ligand_coordinates = coordinate_matrix[
            representative_index
        ].reshape(ligand_pca.n_atoms, 3)
        maximum_ligand_coordinate_difference = float(
            np.max(
                np.abs(
                    physical_ligand_coordinates_relative
                    - pca_ligand_coordinates
                )
            )
        )
    else:
        maximum_ligand_coordinate_difference = np.nan

    ligand_com_relative = (
        ligand_all.center_of_mass()
        - receptor_com_aligned
    )

    # Translate the aligned receptor COM to the origin so the representative
    # complex directly matches the receptor-relative PCA coordinate convention.
    universe.atoms.translate(-receptor_com_aligned)
    centered_receptor_com = receptor.center_of_mass()

    with mda.Writer(
        complex_pdb,
        n_atoms=representative_complex.n_atoms,
    ) as writer:
        writer.write(representative_complex)

    if EXPORT_FULL_SYSTEM_REPRESENTATIVE_DCD:
        with mda.Writer(
            full_system_dcd,
            n_atoms=universe.atoms.n_atoms,
        ) as writer:
            writer.write(universe.atoms)
    else:
        full_system_dcd = ""

    return {
        "representative_ligand_pdb": ligand_only_pdb,
        "representative_PETase21_ligand_pdb": complex_pdb,
        "representative_full_system_dcd": full_system_dcd,
        "representative_complex_atoms": representative_complex.n_atoms,
        "receptor_COM_x_before_centering_A": receptor_com_aligned[0],
        "receptor_COM_y_before_centering_A": receptor_com_aligned[1],
        "receptor_COM_z_before_centering_A": receptor_com_aligned[2],
        "receptor_COM_x_in_complex_PDB_A": centered_receptor_com[0],
        "receptor_COM_y_in_complex_PDB_A": centered_receptor_com[1],
        "receptor_COM_z_in_complex_PDB_A": centered_receptor_com[2],
        "ligand_COM_x_relative_A": ligand_com_relative[0],
        "ligand_COM_y_relative_A": ligand_com_relative[1],
        "ligand_COM_z_relative_A": ligand_com_relative[2],
        "max_ligand_coordinate_check_A": (
            maximum_ligand_coordinate_difference
        ),
    }


def kabsch_superpose(mobile, reference):
    """
    Superpose Nx3 mobile coordinates onto Nx3 reference coordinates.
    Uses equal atom weights because PCA already excludes hydrogens/dummies.
    """

    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)

    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center

    covariance = mobile_centered.T @ reference_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)

    rotation = u_matrix @ vt_matrix

    if np.linalg.det(rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix

    return (
        mobile_centered @ rotation
        + reference_center
    )


def choose_cluster_model(cluster_coordinates):
    """
    Select K by maximum silhouette score and return the fitted K-means model,
    labels, and score table.
    """

    n_observations = len(cluster_coordinates)
    highest_k = min(MAX_CLUSTERS, n_observations - 1)

    if highest_k < 2:
        raise RuntimeError(
            "At least three sampled trajectory frames are required for "
            "automatic clustering."
        )

    results = []
    models = {}

    for n_clusters in range(2, highest_k + 1):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(cluster_coordinates)

        sample_size = min(
            SILHOUETTE_SAMPLE_SIZE,
            n_observations,
        )

        score = silhouette_score(
            cluster_coordinates,
            labels,
            sample_size=(
                sample_size
                if sample_size < n_observations
                else None
            ),
            random_state=RANDOM_STATE,
        )

        results.append(
            {
                "n_clusters": n_clusters,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )

        models[n_clusters] = model

        print(
            f"K = {n_clusters}: silhouette = {score:.6f}"
        )

    score_table = pd.DataFrame(results)

    best_row = score_table.loc[
        score_table["silhouette_score"].idxmax()
    ]

    best_k = int(best_row["n_clusters"])
    best_model = models[best_k]

    # Convert zero-based sklearn labels to human-readable 1-based labels.
    best_labels = best_model.labels_.astype(int) + 1

    return best_model, best_labels, score_table


def _encode_groups(labels):
    """Return sorted group labels, zero-based codes, and group sizes."""

    labels = np.asarray(labels)
    group_labels, encoded = np.unique(
        labels,
        return_inverse=True,
    )
    group_sizes = np.bincount(encoded)

    if len(group_labels) < 2:
        raise ValueError(
            "PERMANOVA/PERMDISP require at least two groups."
        )

    if len(labels) <= len(group_labels):
        raise ValueError(
            "PERMANOVA/PERMDISP require residual degrees of freedom."
        )

    if np.any(group_sizes == 0):
        raise ValueError("An empty cluster was found.")

    return group_labels, encoded, group_sizes


def _group_coordinate_sums(coordinates, encoded, n_groups):
    """Calculate coordinate sums for every encoded group."""

    return np.column_stack(
        [
            np.bincount(
                encoded,
                weights=coordinates[:, feature_index],
                minlength=n_groups,
            )
            for feature_index in range(coordinates.shape[1])
        ]
    )


def _safe_f_ratio(
    sum_squares_between,
    df_between,
    sum_squares_within,
    df_within,
):
    """Calculate an F ratio while handling zero within-group variation."""

    mean_square_between = sum_squares_between / df_between
    mean_square_within = sum_squares_within / df_within

    if mean_square_within <= np.finfo(float).eps:
        if mean_square_between <= np.finfo(float).eps:
            return 0.0
        return float("inf")

    return float(mean_square_between / mean_square_within)


def _multivariate_statistics(
    coordinates,
    encoded,
    group_sizes,
):
    """
    Calculate Euclidean PERMANOVA and centroid-based PERMDISP statistics.

    For Euclidean PC coordinates, the coordinate-space sums of squares are
    equivalent to distance-based PERMANOVA. PERMDISP uses each frame's
    Euclidean distance to its cluster centroid in the same PC space.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    encoded = np.asarray(encoded, dtype=int)
    group_sizes = np.asarray(group_sizes, dtype=int)

    n_observations = coordinates.shape[0]
    n_groups = len(group_sizes)
    df_between = n_groups - 1
    df_within = n_observations - n_groups

    group_sums = _group_coordinate_sums(
        coordinates,
        encoded,
        n_groups,
    )
    group_centroids = group_sums / group_sizes[:, None]

    overall_centroid = coordinates.mean(axis=0)
    total_ss = float(
        np.sum((coordinates - overall_centroid) ** 2)
    )
    within_ss = float(
        np.sum(coordinates ** 2)
        - np.sum((group_sums ** 2) / group_sizes[:, None])
    )

    # Floating-point subtraction can create a negligible negative value.
    within_ss = max(within_ss, 0.0)
    between_ss = max(total_ss - within_ss, 0.0)

    permanova_f = _safe_f_ratio(
        between_ss,
        df_between,
        within_ss,
        df_within,
    )
    permanova_r_squared = (
        between_ss / total_ss
        if total_ss > np.finfo(float).eps
        else 0.0
    )

    distances_to_centroid = np.linalg.norm(
        coordinates - group_centroids[encoded],
        axis=1,
    )
    group_distance_means = (
        np.bincount(
            encoded,
            weights=distances_to_centroid,
            minlength=n_groups,
        )
        / group_sizes
    )
    overall_distance_mean = float(
        distances_to_centroid.mean()
    )

    dispersion_between_ss = float(
        np.sum(
            group_sizes
            * (group_distance_means - overall_distance_mean) ** 2
        )
    )
    dispersion_within_ss = float(
        np.sum(
            (
                distances_to_centroid
                - group_distance_means[encoded]
            ) ** 2
        )
    )
    dispersion_total_ss = (
        dispersion_between_ss + dispersion_within_ss
    )
    permdisp_f = _safe_f_ratio(
        dispersion_between_ss,
        df_between,
        dispersion_within_ss,
        df_within,
    )
    permdisp_eta_squared = (
        dispersion_between_ss / dispersion_total_ss
        if dispersion_total_ss > np.finfo(float).eps
        else 0.0
    )

    return {
        "df_between": df_between,
        "df_within": df_within,
        "permanova_F": permanova_f,
        "permanova_R_squared": permanova_r_squared,
        "permanova_SS_between": between_ss,
        "permanova_SS_within": within_ss,
        "permanova_SS_total": total_ss,
        "permdisp_F": permdisp_f,
        "permdisp_eta_squared": permdisp_eta_squared,
        "permdisp_SS_between": dispersion_between_ss,
        "permdisp_SS_within": dispersion_within_ss,
        "permdisp_SS_total": dispersion_total_ss,
        "distances_to_centroid": distances_to_centroid,
    }


def _permuted_at_least(permuted_value, observed_value):
    """Numerically stable upper-tail comparison for permutation tests."""

    if np.isinf(observed_value):
        return bool(np.isinf(permuted_value))

    tolerance = max(
        1.0,
        abs(observed_value),
    ) * 1.0e-12

    return bool(permuted_value >= observed_value - tolerance)


def run_permanova_permdisp(
    coordinates,
    labels,
    permutations,
    random_state,
):
    """
    Run one-way PERMANOVA and PERMDISP with unrestricted permutations.

    The calculations use Euclidean distances in the supplied PC score space.
    The returned p-values use the standard plus-one correction:
    (1 + number of permuted statistics >= observed) / (1 + permutations).
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)

    if coordinates.ndim != 2:
        raise ValueError(
            "Multivariate coordinates must be a two-dimensional array."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "Non-finite values were found in multivariate coordinates."
        )

    if permutations < 99:
        raise ValueError(
            "Use at least 99 permutations for PERMANOVA/PERMDISP."
        )

    group_labels, encoded, group_sizes = _encode_groups(labels)

    if len(encoded) != len(coordinates):
        raise ValueError(
            "The number of cluster labels differs from the number of frames."
        )

    observed = _multivariate_statistics(
        coordinates,
        encoded,
        group_sizes,
    )

    rng = np.random.default_rng(random_state)
    permanova_exceedances = 0
    permdisp_exceedances = 0

    for _ in range(permutations):
        permuted_encoded = rng.permutation(encoded)
        permuted = _multivariate_statistics(
            coordinates,
            permuted_encoded,
            group_sizes,
        )

        permanova_exceedances += _permuted_at_least(
            permuted["permanova_F"],
            observed["permanova_F"],
        )
        permdisp_exceedances += _permuted_at_least(
            permuted["permdisp_F"],
            observed["permdisp_F"],
        )

    permanova_p = (
        permanova_exceedances + 1
    ) / (permutations + 1)
    permdisp_p = (
        permdisp_exceedances + 1
    ) / (permutations + 1)
    minimum_p = 1.0 / (permutations + 1)
    pc_space = f"PC1-PC{coordinates.shape[1]}"

    permanova_table = pd.DataFrame(
        [
            {
                "test": "one-way PERMANOVA",
                "grouping": "K-means cluster",
                "coordinate_space": pc_space,
                "distance": "Euclidean",
                "n_frames": len(coordinates),
                "n_groups": len(group_labels),
                "df_between": observed["df_between"],
                "df_within": observed["df_within"],
                "sum_squares_between": observed[
                    "permanova_SS_between"
                ],
                "sum_squares_within": observed[
                    "permanova_SS_within"
                ],
                "sum_squares_total": observed[
                    "permanova_SS_total"
                ],
                "pseudo_F": observed["permanova_F"],
                "R_squared": observed[
                    "permanova_R_squared"
                ],
                "permutations": permutations,
                "p_value": permanova_p,
                "minimum_possible_p": minimum_p,
            }
        ]
    )

    permdisp_table = pd.DataFrame(
        [
            {
                "test": "one-way PERMDISP",
                "grouping": "K-means cluster",
                "coordinate_space": pc_space,
                "distance": "Euclidean to cluster centroid",
                "n_frames": len(coordinates),
                "n_groups": len(group_labels),
                "df_between": observed["df_between"],
                "df_within": observed["df_within"],
                "sum_squares_between": observed[
                    "permdisp_SS_between"
                ],
                "sum_squares_within": observed[
                    "permdisp_SS_within"
                ],
                "sum_squares_total": observed[
                    "permdisp_SS_total"
                ],
                "F": observed["permdisp_F"],
                "eta_squared": observed[
                    "permdisp_eta_squared"
                ],
                "permutations": permutations,
                "p_value": permdisp_p,
                "minimum_possible_p": minimum_p,
            }
        ]
    )

    dispersion_rows = []
    distances = observed["distances_to_centroid"]

    for group_index, group_label in enumerate(group_labels):
        group_distances = distances[encoded == group_index]
        dispersion_rows.append(
            {
                "cluster": group_label,
                "n_frames": len(group_distances),
                "mean_distance_to_centroid": group_distances.mean(),
                "sd_distance_to_centroid": (
                    group_distances.std(ddof=1)
                    if len(group_distances) > 1
                    else np.nan
                ),
                "median_distance_to_centroid": np.median(
                    group_distances
                ),
            }
        )

    dispersion_table = pd.DataFrame(dispersion_rows)

    return permanova_table, permdisp_table, dispersion_table


def set_plot_style():
    """Apply the requested GraphPad Prism-like global typography."""

    plt.rcParams.update(
        {
            # Matplotlib uses the first installed font in this list.
            "font.family": [
                "Josefin Sans",
                "Arial",
                "DejaVu Sans",
            ],
            "font.weight": "bold",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelweight": "bold",
            "axes.labelsize": 11,
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "figure.dpi": 150,
        }
    )


GRAPH_BACKGROUND = "#D3D3D3"
GRAPH_FRAME_COLOR = "#000000"
GRAPH_GRID_COLOR = "#FFFFFF"

# User-provided palettes.
# K-means: saturated purple -> magenta -> gold -> navy -> teal gradient.
KMEANS_GRADIENT_COLORS = [
    "#58094F",
    "#841E62",
    "#F3AA20",
    "#2A445E",
    "#346B6D",
]

# Time-colored PCA uses the standard Matplotlib viridis gradient exactly:
# dark purple -> blue -> blue-green -> green -> yellow
# reference anchors: #440154, #3B528B, #21908C, #5DC863, #FDE725.
# K-means colors remain completely independent and use the palette above.
KMEANS_CMAP = LinearSegmentedColormap.from_list(
    "PET_dimer_kmeans_gradient",
    KMEANS_GRADIENT_COLORS,
)
TIME_CMAP = plt.get_cmap("viridis")


def cluster_colors(n_clusters):
    """Return one reproducible color per K-means cluster."""
    if n_clusters <= 1:
        return [KMEANS_CMAP(0.5)]
    return [
        KMEANS_CMAP(value)
        for value in np.linspace(0.0, 1.0, n_clusters)
    ]


def style_graphpad_axis(axis, show_grid=True):
    """
    Format one axis like the supplied GraphPad Prism reference:
    gray plotting area, full black frame, mirrored inward ticks, and
    white dashed major grid.
    """

    axis.set_facecolor(GRAPH_BACKGROUND)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRAPH_FRAME_COLOR)
        spine.set_linewidth(1.5)

    # AutoMinorLocator(2) divides every major interval into two equal parts,
    # producing exactly one minor tick midway between adjacent major ticks.
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

    for tick_label in (
        list(axis.get_xticklabels())
        + list(axis.get_yticklabels())
    ):
        tick_label.set_fontweight("bold")


def set_plot_area_size(figure, axis, square=False):
    """Set each axes area in cm, preserving space for labels and legends."""
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
        tuple(np.round(box[:, 1], 9)) for box in boxes.values()
    })

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
        x0, x1 = (resize_x(value) for value in box[:, 0])
        y0, y1 = (resize_y(value) for value in box[:, 1])
        item.set_position([
            x0 / new_width, y0 / new_height,
            (x1 - x0) / new_width, (y1 - y0) / new_height,
        ])


def save_cluster_plot(
    scores,
    explained_variance,
    centers,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (5.5, 5.5) if square else (6.6, 4.2)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    clusters = sorted(scores["cluster"].unique())
    colors = cluster_colors(len(clusters))

    for color_index, cluster in enumerate(clusters):
        subset = scores[scores["cluster"] == cluster]

        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            s=24,
            alpha=0.80,
            color=colors[color_index],
            edgecolors="black",
            linewidths=0.25,
            label=f"Cluster {cluster}",
        )

    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=105,
        c="black",
        edgecolors="white",
        linewidths=1.0,
        label="Centroid",
        zorder=5,
    )

    axis.set_xlabel(
        f"PC1 ({explained_variance[0]:.2f}%)"
    )
    axis.set_ylabel(
        f"PC2 ({explained_variance[1]:.2f}%)"
    )
    axis.set_title("PETase21–PET dimer: ligand PCA clusters")
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        prop={"weight": "bold", "size": 9},
    )

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_time_colored_plot(
    scores,
    explained_variance,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (5.5, 5.5) if square else (5.6, 4.2)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    time_norm = Normalize(
        vmin=0.0,
        vmax=TRAJECTORY_TOTAL_NS,
    )

    scatter = axis.scatter(
        scores["PC1"],
        scores["PC2"],
        c=scores["time_ns"],
        cmap=TIME_CMAP,
        norm=time_norm,
        s=24,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.20,
    )

    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Time (ns; rescaled)", fontweight="bold")
    colorbar.set_ticks(TIME_TICKS_NS)
    colorbar.outline.set_linewidth(1.2)
    colorbar.outline.set_edgecolor("black")
    colorbar.ax.tick_params(
        direction="in",
        width=1.2,
        length=4,
    )
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    axis.set_xlabel(
        f"PC1 ({explained_variance[0]:.2f}%)"
    )
    axis.set_ylabel(
        f"PC2 ({explained_variance[1]:.2f}%)"
    )
    axis.set_title("PETase21–PET dimer: PCA trajectory")

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_scree_plot(
    variance_table,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (5.5, 5.5) if square else (5.8, 4.0)
    figure, axis_left = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis_left)

    x_values = variance_table["component_number"]

    axis_left.bar(
        x_values,
        variance_table["explained_variance_percent"],
        color="#841E62",
        edgecolor="#58094F",
        linewidth=1.2,
        alpha=0.95,
    )
    axis_left.set_xlabel("Principal component")
    axis_left.set_ylabel("Explained variance (%)")
    axis_left.set_xticks(x_values)

    axis_right = axis_left.twinx()
    style_graphpad_axis(axis_right, show_grid=False)
    axis_right.patch.set_alpha(0.0)
    axis_right.plot(
        x_values,
        variance_table["cumulative_variance_percent"],
        color="#F3AA20",
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=5,
        linewidth=1.8,
    )
    axis_right.set_ylabel("Cumulative variance (%)")
    axis_right.set_ylim(0.0, 105.0)

    axis_left.set_title("PETase21–PET dimer: PCA variance")

    figure.tight_layout()
    set_plot_area_size(figure, axis_left, square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_pcs_vs_time_plot(
    scores,
    explained_variance,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (5.8, 10.5) if square else (6.4, 5.2)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figure_size,
        sharex=True,
    )

    for axis in axes:
        style_graphpad_axis(axis)
        axis.set_xlim(0.0, TRAJECTORY_TOTAL_NS)
        axis.set_xticks(TIME_TICKS_NS)

    axes[0].plot(
        scores["time_ns"],
        scores["PC1"],
        color="#577E89",
        linewidth=1.2,
    )
    axes[0].set_ylabel(
        f"PC1 ({explained_variance[0]:.2f}%)"
    )

    axes[1].plot(
        scores["time_ns"],
        scores["PC2"],
        color="#E1A36F",
        linewidth=1.2,
    )
    axes[1].set_ylabel(
        f"PC2 ({explained_variance[1]:.2f}%)"
    )
    axes[1].set_xlabel("Time (ns; rescaled)")

    figure.suptitle(
        "PETase21–PET dimer: PCs versus time",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    set_plot_area_size(figure, axes[0], square=square)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_cluster_occupancy_plot(
    cluster_summary,
    output_path,
    square=False,
):
    set_plot_style()

    figure_size = (5.5, 5.5) if square else (5.6, 4.0)
    figure, axis = plt.subplots(figsize=figure_size)
    style_graphpad_axis(axis)

    ordered_clusters = cluster_summary["cluster"].astype(int).to_numpy()
    colors = cluster_colors(len(ordered_clusters))

    bars = axis.bar(
        cluster_summary["cluster"].astype(str),
        cluster_summary["occupancy_percent"],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            for value in cluster_summary["occupancy_percent"]
        ],
        padding=3,
        fontsize=9,
        fontweight="bold",
    )

    axis.set_xlabel("Cluster")
    axis.set_ylabel("Trajectory occupancy (%)")
    axis.set_ylim(
        0.0,
        max(105.0, cluster_summary["occupancy_percent"].max() * 1.15),
    )
    axis.set_title("PETase21–PET dimer: cluster occupancy")

    figure.tight_layout()
    set_plot_area_size(figure, axis, square=square)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# INPUT VALIDATION AND SYSTEM LOADING
# ============================================================

validate_file(TOPOLOGY, "Topology PSF")
validate_file(ENERGY_LOG, "NAMD energy log")
for trajectory_number, trajectory_path in enumerate(TRAJECTORIES, start=1):
    validate_file(
        trajectory_path,
        f"Trajectory DCD #{trajectory_number}",
    )

if ANALYSIS_MODE not in {
    "receptor_context",
    "internal_conformation",
}:
    raise ValueError(
        "ANALYSIS_MODE must be 'receptor_context' or "
        "'internal_conformation'."
    )

if not isinstance(STEP, int) or STEP < 1:
    raise ValueError("STEP must be an integer >= 1.")

print("=" * 78)
print("PETase21 + PET dimer - LIGAND PCA AND CLUSTERING")
print("=" * 78)
print(f"Analysis mode: {ANALYSIS_MODE}")
print(f"Topology: {TOPOLOGY}")
for trajectory_number, trajectory_path in enumerate(TRAJECTORIES, start=1):
    print(f"Trajectory {trajectory_number}: {trajectory_path}")
print(
    f"Displayed trajectory timeline (rescaled): 0-{TRAJECTORY_TOTAL_NS:g} ns"
)
print(f"Output folder: {OUTPUT_DIR}")

# Load the explicitly selected PETase21/PET-dimer DCD against ionized.psf.
# Global frame indices below refer to this trajectory.
universe = mda.Universe(TOPOLOGY, *TRAJECTORIES)

receptor = universe.select_atoms(RECEPTOR_SEL)
receptor_fit = universe.select_atoms(RECEPTOR_FIT_SEL)
ligand_all = universe.select_atoms(LIGAND_ALL_SEL)
ligand_pca = universe.select_atoms(LIGAND_PCA_SEL)
water = universe.select_atoms(WATER_SEL)
representative_complex = universe.select_atoms(
    REPRESENTATIVE_COMPLEX_SEL
)

if receptor.n_atoms == 0:
    raise ValueError(
        f"No PETase21 receptor atoms found with selection: {RECEPTOR_SEL}"
    )

if receptor_fit.n_atoms < 3:
    raise ValueError(
        "Fewer than three PETase21 backbone fitting atoms were found with "
        f"selection: {RECEPTOR_FIT_SEL}"
    )

if ligand_all.n_atoms == 0:
    raise ValueError(
        f"No ligand atoms found with selection: {LIGAND_ALL_SEL}"
    )

if receptor.n_residues != EXPECTED_RECEPTOR_RESIDUES:
    raise ValueError(
        f"Expected {EXPECTED_RECEPTOR_RESIDUES} PETase21 residues in segid AP1, "
        f"but found {receptor.n_residues}. Check that this is the intended ionized.psf."
    )

if ligand_all.n_residues != EXPECTED_LIGAND_RESIDUES:
    raise ValueError(
        f"Expected exactly one PET dimer LIG residue, but found "
        f"{ligand_all.n_residues} residues selected by {LIGAND_ALL_SEL}. "
        "Inspect ligand resname/segid in ionized.psf before analysis."
    )

if ligand_all.n_atoms != EXPECTED_LIGAND_ATOMS:
    raise ValueError(
        f"Expected {EXPECTED_LIGAND_ATOMS} atoms for the PET dimer LIG residue, "
        f"but found {ligand_all.n_atoms}. Check the topology/ligand selection."
    )

if ligand_pca.n_atoms != EXPECTED_LIGAND_PCA_HEAVY_ATOMS:
    raise ValueError(
        f"Expected {EXPECTED_LIGAND_PCA_HEAVY_ATOMS} PET-dimer heavy atoms for PCA, "
        f"but found {ligand_pca.n_atoms}. Check hydrogen/dummy-atom naming."
    )

if representative_complex.n_atoms != (
    receptor.n_atoms + ligand_all.n_atoms
):
    raise ValueError(
        "The representative PETase21-ligand selection contains an unexpected "
        "number of atoms. Check receptor/ligand selections for overlap."
    )

print(f"Total system atoms: {universe.atoms.n_atoms}")
print(
    f"PETase21 receptor: {receptor.n_atoms} atoms in "
    f"{receptor.n_residues} residues"
)
print(f"PETase21 backbone fitting atoms: {receptor_fit.n_atoms}")
print(f"All ligand atoms: {ligand_all.n_atoms}")
print(f"Ligand PCA heavy atoms: {ligand_pca.n_atoms}")
print(
    f"Water: {water.n_atoms} atoms in {water.n_residues} residues"
)
print(
    "Representative PETase21 + ligand atoms: "
    f"{representative_complex.n_atoms}"
)
print(f"Trajectory frames: {len(universe.trajectory)}")

total_frames = len(universe.trajectory)
if total_frames < 3:
    raise RuntimeError("At least three trajectory frames are required for PCA clustering.")

# STEP may not divide (N - 1). Include the final frame of the combined trajectory explicitly.
sampled_frame_indices = list(range(0, total_frames, STEP))
if sampled_frame_indices[-1] != total_frames - 1:
    sampled_frame_indices.append(total_frames - 1)


# Per-file audit for reproducibility; '500 ns' remains the display convention.
from MDAnalysis.coordinates.DCD import DCDReader

DCD_FRAME_COUNTS = []
for dcd_path in TRAJECTORIES:
    with DCDReader(dcd_path) as reader:
        DCD_FRAME_COUNTS.append(len(reader))
if any(count < 1 for count in DCD_FRAME_COUNTS):
    raise ValueError(f"An input DCD contains zero frames: {DCD_FRAME_COUNTS}")
DCD_OFFSETS = np.concatenate(([0], np.cumsum(DCD_FRAME_COUNTS, dtype=np.int64)))
if DCD_OFFSETS[-1] != total_frames:
    raise RuntimeError("DCD frame-count audit disagrees with chained trajectory.")


def source_for_frame(global_frame):
    dcd_zero_based = int(np.searchsorted(DCD_OFFSETS[1:], global_frame, side="right"))
    return (
        dcd_zero_based + 1,
        int(global_frame - DCD_OFFSETS[dcd_zero_based]),
        TRAJECTORIES[dcd_zero_based],
    )

# ============================================================
# EXTRACT RECEPTOR-REFERENCED LIGAND COORDINATES
# ============================================================

frame_numbers = []
times_ns = []
coordinates = []
ligand_com_relative = []
receptor_fit_rmsd_A = []
reference_receptor_coordinates = None
reference_ligand_coordinates = None

print(
    "\nExtracting PETase21-aligned, receptor-referenced ligand coordinates..."
)

for sampled_index, global_frame in enumerate(sampled_frame_indices):
    timestep = universe.trajectory[global_frame]
    reconstruct_receptor_ligand(
        receptor,
        ligand_all,
        timestep.dimensions,
    )

    if reference_receptor_coordinates is None:
        reference_receptor_coordinates = (
            receptor_fit.positions.astype(np.float64).copy()
        )

    current_receptor_fit_rmsd = align_system_to_receptor_reference(
        universe,
        receptor_fit,
        reference_receptor_coordinates,
    )

    receptor_com = receptor.center_of_mass()

    current_coordinates = (
        ligand_pca.positions.astype(np.float64).copy()
        - receptor_com
    )

    current_ligand_com = (
        ligand_all.center_of_mass()
        - receptor_com
    )

    if ANALYSIS_MODE == "internal_conformation":
        if reference_ligand_coordinates is None:
            reference_ligand_coordinates = (
                current_coordinates.copy()
            )

        current_coordinates = kabsch_superpose(
            current_coordinates,
            reference_ligand_coordinates,
        )

    frame_numbers.append(int(global_frame))

    # Deliberately remap the selected DCD trajectory to 0-500 display ns.
    plotted_time_ns = (
        TRAJECTORY_TOTAL_NS
        * float(global_frame)
        / float(total_frames - 1)
    )

    times_ns.append(plotted_time_ns)
    coordinates.append(current_coordinates.reshape(-1))
    ligand_com_relative.append(current_ligand_com)
    receptor_fit_rmsd_A.append(current_receptor_fit_rmsd)

    if (
        sampled_index % 100 == 0
        or global_frame == total_frames - 1
    ):
        print(
            f"Processed combined DCD frame {global_frame + 1}/"
            f"{total_frames} "
            f"({plotted_time_ns:.4f} displayed ns)"
        )

if not (
    np.isclose(times_ns[0], 0.0)
    and np.isclose(times_ns[-1], TRAJECTORY_TOTAL_NS)
    and frame_numbers[-1] == total_frames - 1
):
    raise RuntimeError("The sampled trajectory does not span the requested 0–500 ns axis.")

TIME_AXIS_METADATA = os.path.join(OUTPUT_DIR, "time_axis_method.txt")
with open(TIME_AXIS_METADATA, "w", encoding="utf-8") as metadata:
    metadata.write(
        "Analysis: PETase21 + PET dimer ligand PCA\n"
        "Trajectory: ionized.dcd (single explicitly selected DCD).\n"
        "Time convention: FORCED DISPLAY/EXPORT AXIS, not verified physical MD time.\n"
        "Formula: time_ns = global_frame / (total_frames - 1) * 500.\n"
        "First selected frame: 0 display ns; final selected frame: 500 display ns.\n"
        "Verify actual simulation time in the NAMD log, timestep, and DCD stride before reporting physical ns.\n"
        f"Frames: {total_frames}\n"
        f"Sampled frames: {len(frame_numbers)}\n"
        f"STEP: {STEP}\n"
        f"Topology: {TOPOLOGY}\n"
        f"Energy log (audit only; not PCA input): {ENERGY_LOG}\n"
        f"Analysis mode: {ANALYSIS_MODE}\n"
        f"Verified PETase21 residues expected: {EXPECTED_RECEPTOR_RESIDUES}\n"
        f"Verified PET dimer atoms expected: {EXPECTED_LIGAND_ATOMS}; PCA heavy atoms expected: {EXPECTED_LIGAND_PCA_HEAVY_ATOMS}\n"
    )
    for dcd_index, (dcd_path, frame_count) in enumerate(zip(TRAJECTORIES, DCD_FRAME_COUNTS), start=1):
        metadata.write(f"DCD {dcd_index}: {dcd_path} ({frame_count} frames)\n")
    metadata.write("PCA input is coordinate data from ionized.dcd; the energy log is retained only for audit/context.\n")

coordinate_matrix = np.asarray(coordinates, dtype=np.float64)
ligand_com_relative = np.asarray(
    ligand_com_relative,
    dtype=np.float64,
)
receptor_fit_rmsd_A = np.asarray(
    receptor_fit_rmsd_A,
    dtype=np.float64,
)

if coordinate_matrix.shape[0] < 3:
    raise RuntimeError(
        "Fewer than three frames were sampled; PCA clustering cannot proceed."
    )

if not np.all(np.isfinite(coordinate_matrix)):
    raise RuntimeError(
        "Non-finite ligand coordinates were found after preprocessing."
    )

print(
    f"Coordinate matrix: {coordinate_matrix.shape[0]} frames x "
    f"{coordinate_matrix.shape[1]} Cartesian features"
)
print(
    "PETase21 backbone alignment RMSD: "
    f"mean={receptor_fit_rmsd_A.mean():.4f} A, "
    f"max={receptor_fit_rmsd_A.max():.4f} A"
)

# ============================================================
# PCA
# ============================================================

n_components = min(
    MAX_PCS,
    coordinate_matrix.shape[0] - 1,
    coordinate_matrix.shape[1],
)

if n_components < 2:
    raise RuntimeError("At least two PCA components are required.")

pca = PCA(
    n_components=n_components,
    svd_solver="full",
)

pca_scores = pca.fit_transform(coordinate_matrix)
explained_percent = pca.explained_variance_ratio_ * 100.0

print("\nExplained variance:")
for component_index, percent in enumerate(
    explained_percent,
    start=1,
):
    print(f"PC{component_index}: {percent:.4f}%")


# ============================================================
# AUTOMATIC CLUSTERING IN PCA SPACE
# ============================================================

n_cluster_pcs = min(CLUSTER_PCS, n_components)
cluster_coordinates = pca_scores[:, :n_cluster_pcs]

print(
    f"\nSelecting K-means clusters using the first "
    f"{n_cluster_pcs} PCs..."
)

cluster_model, cluster_labels, silhouette_table = (
    choose_cluster_model(cluster_coordinates)
)

best_k = int(cluster_model.n_clusters)
best_silhouette = float(
    silhouette_table.loc[
        silhouette_table["n_clusters"] == best_k,
        "silhouette_score",
    ].iloc[0]
)

print(
    f"Selected cluster count: {best_k} "
    f"(silhouette = {best_silhouette:.6f})"
)


# ============================================================
# EXPLORATORY PERMANOVA AND PERMDISP
# ============================================================

warnings.warn(
    "PERMANOVA and PERMDISP are exploratory here: K-means clusters were "
    "defined from the same PC scores being tested, and consecutive MD "
    "frames are temporally autocorrelated. Do not interpret the permutation "
    "p-values as independent confirmation of cluster significance."
)

print(
    f"\nRunning exploratory one-way PERMANOVA and PERMDISP with "
    f"{MULTIVARIATE_PERMUTATIONS} permutations in the first "
    f"{n_cluster_pcs} PCs..."
)

(
    permanova_table,
    permdisp_table,
    permdisp_groups_table,
) = run_permanova_permdisp(
    cluster_coordinates,
    cluster_labels,
    permutations=MULTIVARIATE_PERMUTATIONS,
    random_state=RANDOM_STATE + 100,
)

permanova_result = permanova_table.iloc[0]
permdisp_result = permdisp_table.iloc[0]

print(
    "PERMANOVA: "
    f"pseudo-F = {permanova_result['pseudo_F']:.6f}, "
    f"R^2 = {permanova_result['R_squared']:.6f}, "
    f"p = {permanova_result['p_value']:.6f}"
)
print(
    "PERMDISP: "
    f"F = {permdisp_result['F']:.6f}, "
    f"eta^2 = {permdisp_result['eta_squared']:.6f}, "
    f"p = {permdisp_result['p_value']:.6f}"
)


# ============================================================
# PCA SCORE AND VARIANCE TABLES
# ============================================================

score_data = {
    "sample_index": np.arange(len(pca_scores), dtype=int),
    "trajectory_frame": frame_numbers,
    "DCD_number_1based": [source_for_frame(frame)[0] for frame in frame_numbers],
    "DCD_local_frame_0based": [source_for_frame(frame)[1] for frame in frame_numbers],
    "DCD_path": [source_for_frame(frame)[2] for frame in frame_numbers],
    "time_ns": times_ns,  # forced display axis, not measured physical time
    "cluster": cluster_labels,
    "ligand_COM_x_relative_A": ligand_com_relative[:, 0],
    "ligand_COM_y_relative_A": ligand_com_relative[:, 1],
    "ligand_COM_z_relative_A": ligand_com_relative[:, 2],
    "PETase21_backbone_fit_RMSD_A": receptor_fit_rmsd_A,
}

for component_index in range(n_components):
    score_data[f"PC{component_index + 1}"] = (
        pca_scores[:, component_index]
    )

scores_table = pd.DataFrame(score_data)

variance_table = pd.DataFrame(
    {
        "component": [
            f"PC{index}"
            for index in range(1, n_components + 1)
        ],
        "component_number": np.arange(1, n_components + 1),
        "eigenvalue": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": explained_percent,
        "cumulative_variance_percent": np.cumsum(
            explained_percent
        ),
    }
)


# ============================================================
# LOADINGS TABLE
# ============================================================

feature_labels = []

for atom in ligand_pca:
    atom_prefix = (
        f"index_{atom.index}_"
        f"{atom.segid}_{atom.resname}_{atom.resid}_{atom.name}"
    )

    for axis_name in ("x", "y", "z"):
        feature_labels.append(f"{atom_prefix}_{axis_name}")

loading_data = {
    "feature": feature_labels,
}

for component_index in range(n_components):
    loading_data[f"PC{component_index + 1}_loading"] = (
        pca.components_[component_index]
    )

loadings_table = pd.DataFrame(loading_data)


# ============================================================
# CLUSTER SUMMARIES AND REPRESENTATIVE FRAMES
# ============================================================

cluster_summary_rows = []
representative_rows = []

for cluster_number in range(1, best_k + 1):
    member_indices = np.where(
        cluster_labels == cluster_number
    )[0]

    member_coordinates = cluster_coordinates[member_indices]
    centroid = cluster_model.cluster_centers_[
        cluster_number - 1
    ]

    distances_to_centroid = np.linalg.norm(
        member_coordinates - centroid,
        axis=1,
    )

    representative_local_index = int(
        np.argmin(distances_to_centroid)
    )
    representative_index = int(
        member_indices[representative_local_index]
    )

    cluster_scores = scores_table.iloc[member_indices]

    cluster_summary_rows.append(
        {
            "cluster": cluster_number,
            "n_frames": len(member_indices),
            "occupancy_percent": (
                100.0 * len(member_indices) / len(scores_table)
            ),
            "first_time_ns": cluster_scores["time_ns"].min(),
            "last_time_ns": cluster_scores["time_ns"].max(),
            "mean_PC1": cluster_scores["PC1"].mean(),
            "mean_PC2": cluster_scores["PC2"].mean(),
            "mean_ligand_COM_x_relative_A": cluster_scores[
                "ligand_COM_x_relative_A"
            ].mean(),
            "mean_ligand_COM_y_relative_A": cluster_scores[
                "ligand_COM_y_relative_A"
            ].mean(),
            "mean_ligand_COM_z_relative_A": cluster_scores[
                "ligand_COM_z_relative_A"
            ].mean(),
            "mean_PETase21_backbone_fit_RMSD_A": cluster_scores[
                "PETase21_backbone_fit_RMSD_A"
            ].mean(),
        }
    )

    representative_outputs = write_representative_structures(
        universe=universe,
        receptor=receptor,
        receptor_fit=receptor_fit,
        reference_receptor_coordinates=reference_receptor_coordinates,
        ligand_all=ligand_all,
        ligand_pca=ligand_pca,
        representative_complex=representative_complex,
        representative_index=representative_index,
        trajectory_frame=int(frame_numbers[representative_index]),
        cluster_number=cluster_number,
        coordinate_matrix=coordinate_matrix,
    )

    representative_rows.append(
        {
            "cluster": cluster_number,
            "sample_index": representative_index,
            "trajectory_frame": frame_numbers[representative_index],
            "time_ns": times_ns[representative_index],
            "distance_to_cluster_centroid": distances_to_centroid[
                representative_local_index
            ],
            **representative_outputs,
        }
    )

cluster_summary_table = pd.DataFrame(cluster_summary_rows)
representatives_table = pd.DataFrame(representative_rows)


# ============================================================
# SAVE TABLES
# ============================================================

csv_settings = {
    "index": False,
    "encoding": "utf-8-sig",
}

scores_table.to_csv(SCORES_CSV, **csv_settings)
scores_table[["sample_index", "trajectory_frame", "DCD_number_1based",
              "DCD_local_frame_0based", "DCD_path", "time_ns"]].to_csv(
    os.path.join(OUTPUT_DIR, "PET_dimer_ligand_PCA_frame_source_audit.csv"),
    **csv_settings,
)
variance_table.to_csv(VARIANCE_CSV, **csv_settings)
loadings_table.to_csv(LOADINGS_CSV, **csv_settings)
silhouette_table.to_csv(SILHOUETTE_CSV, **csv_settings)
cluster_summary_table.to_csv(
    CLUSTER_SUMMARY_CSV,
    **csv_settings,
)
representatives_table.to_csv(
    REPRESENTATIVES_CSV,
    **csv_settings,
)
permanova_table.to_csv(PERMANOVA_CSV, **csv_settings)
permdisp_table.to_csv(PERMDISP_CSV, **csv_settings)
permdisp_groups_table.to_csv(
    PERMDISP_GROUPS_CSV,
    **csv_settings,
)

with open(
    MULTIVARIATE_TXT,
    "w",
    encoding="utf-8",
) as multivariate_file:
    multivariate_file.write(
        "PETase21 + PET dimer - exploratory PERMANOVA and PERMDISP\n"
    )
    multivariate_file.write("=" * 72 + "\n\n")
    multivariate_file.write(
        f"Coordinate space: PC1-PC{n_cluster_pcs}\n"
    )
    multivariate_file.write("Distance: Euclidean\n")
    multivariate_file.write("Grouping: K-means cluster\n")
    multivariate_file.write(
        f"Frames: {len(cluster_coordinates)}\n"
    )
    multivariate_file.write(f"Groups: {best_k}\n")
    multivariate_file.write(
        f"Permutations: {MULTIVARIATE_PERMUTATIONS}\n"
    )
    multivariate_file.write(
        "Permutation p-value calculation: "
        "(exceedances + 1) / (permutations + 1)\n\n"
    )

    multivariate_file.write("PERMANOVA\n")
    multivariate_file.write(
        f"  pseudo-F: {permanova_result['pseudo_F']:.10f}\n"
    )
    multivariate_file.write(
        f"  R-squared: {permanova_result['R_squared']:.10f}\n"
    )
    multivariate_file.write(
        f"  p-value: {permanova_result['p_value']:.10f}\n"
    )
    multivariate_file.write(
        f"  df: {int(permanova_result['df_between'])}, "
        f"{int(permanova_result['df_within'])}\n\n"
    )

    multivariate_file.write("PERMDISP\n")
    multivariate_file.write(
        f"  F: {permdisp_result['F']:.10f}\n"
    )
    multivariate_file.write(
        f"  eta-squared: {permdisp_result['eta_squared']:.10f}\n"
    )
    multivariate_file.write(
        f"  p-value: {permdisp_result['p_value']:.10f}\n"
    )
    multivariate_file.write(
        f"  df: {int(permdisp_result['df_between'])}, "
        f"{int(permdisp_result['df_within'])}\n\n"
    )

    multivariate_file.write("Dispersion by cluster\n")
    for row in permdisp_groups_table.itertuples(index=False):
        multivariate_file.write(
            f"  Cluster {row.cluster}: n={row.n_frames}, "
            f"mean={row.mean_distance_to_centroid:.10f}, "
            f"SD={row.sd_distance_to_centroid:.10f}, "
            f"median={row.median_distance_to_centroid:.10f}\n"
        )

    multivariate_file.write(
        "\nInterpretive caution\n"
        "The K-means groups were derived from the same PC coordinates used "
        "by these tests. Therefore, PERMANOVA separation is partly built "
        "into the grouping and is not independent cluster validation. "
        "Moreover, unrestricted permutations assume exchangeable "
        "observations, whereas adjacent MD frames are temporally "
        "autocorrelated. Use these results descriptively; formal inference "
        "should rely on independent replicate trajectories or an explicitly "
        "validated block-permutation strategy.\n"
    )


# ============================================================
# SAVE FIGURES
# ============================================================

save_cluster_plot(
    scores_table,
    explained_percent,
    cluster_model.cluster_centers_,
    PCA_CLUSTER_PNG,
    square=False,
)
save_cluster_plot(
    scores_table,
    explained_percent,
    cluster_model.cluster_centers_,
    PCA_CLUSTER_SQUARE_PNG,
    square=True,
)

save_time_colored_plot(
    scores_table,
    explained_percent,
    PCA_TIME_PNG,
    square=False,
)
save_time_colored_plot(
    scores_table,
    explained_percent,
    PCA_TIME_SQUARE_PNG,
    square=True,
)

save_scree_plot(
    variance_table,
    SCREE_PNG,
    square=False,
)
save_scree_plot(
    variance_table,
    SCREE_SQUARE_PNG,
    square=True,
)

save_pcs_vs_time_plot(
    scores_table,
    explained_percent,
    PCS_TIME_PNG,
    square=False,
)
save_pcs_vs_time_plot(
    scores_table,
    explained_percent,
    PCS_TIME_SQUARE_PNG,
    square=True,
)

save_cluster_occupancy_plot(
    cluster_summary_table,
    CLUSTER_OCCUPANCY_PNG,
    square=False,
)
save_cluster_occupancy_plot(
    cluster_summary_table,
    CLUSTER_OCCUPANCY_SQUARE_PNG,
    square=True,
)


# ============================================================
# TEXT SUMMARY
# ============================================================

with open(SUMMARY_TXT, "w", encoding="utf-8") as summary_file:
    summary_file.write(
        "PETase21 + PET dimer - ligand PCA and clustering\n"
    )
    summary_file.write("=" * 78 + "\n\n")
    summary_file.write(f"Analysis mode: {ANALYSIS_MODE}\n")
    summary_file.write("Temperature: not verified from NAMD production log\n")
    summary_file.write(
        "Displayed trajectory timeline (rescaled): "
        f"0-{TRAJECTORY_TOTAL_NS:g} ns\n"
    )
    summary_file.write(
        "Time convention: global frame index across the selected DCD "
        "was linearly remapped to the requested 0-500 ns range; DCD-internal "
        "timestamps were not used for plots/tables; this is NOT a verified physical duration.\n"
    )
    summary_file.write(f"Trajectory frames: {total_frames}\n")
    summary_file.write(f"Sampling step: {STEP}\n")
    summary_file.write(
        f"Sampled frames: {len(scores_table)}\n"
    )
    summary_file.write(
        f"PETase21 atoms: {receptor.n_atoms}\n"
    )
    summary_file.write(
        f"PETase21 residues: {receptor.n_residues}\n"
    )
    summary_file.write(
        f"PETase21 backbone fitting atoms: {receptor_fit.n_atoms}\n"
    )
    summary_file.write(
        f"Ligand PCA heavy atoms: {ligand_pca.n_atoms}\n"
    )
    summary_file.write(
        f"Cartesian features: {coordinate_matrix.shape[1]}\n"
    )
    summary_file.write(
        "PETase21 backbone fit RMSD (A): "
        f"mean={receptor_fit_rmsd_A.mean():.6f}, "
        f"max={receptor_fit_rmsd_A.max():.6f}\n"
    )
    summary_file.write(
        f"PCs retained: {n_components}\n\n"
    )

    summary_file.write("Explained variance:\n")
    for component_index, percent in enumerate(
        explained_percent,
        start=1,
    ):
        summary_file.write(
            f"  PC{component_index}: {percent:.6f}%\n"
        )

    summary_file.write(
        f"\nClustering PCs used: {n_cluster_pcs}\n"
    )
    summary_file.write(f"Selected clusters: {best_k}\n")
    summary_file.write(
        f"Best silhouette score: {best_silhouette:.6f}\n\n"
    )

    summary_file.write("Exploratory multivariate tests:\n")
    summary_file.write(
        "  Space and distance: "
        f"PC1-PC{n_cluster_pcs}, Euclidean\n"
    )
    summary_file.write(
        f"  Permutations: {MULTIVARIATE_PERMUTATIONS}\n"
    )
    summary_file.write(
        "  PERMANOVA: "
        f"pseudo-F={permanova_result['pseudo_F']:.6f}, "
        f"R-squared={permanova_result['R_squared']:.6f}, "
        f"p={permanova_result['p_value']:.6f}\n"
    )
    summary_file.write(
        "  PERMDISP: "
        f"F={permdisp_result['F']:.6f}, "
        f"eta-squared={permdisp_result['eta_squared']:.6f}, "
        f"p={permdisp_result['p_value']:.6f}\n"
    )
    summary_file.write(
        "  Caution: clusters were derived from the tested PC space, and "
        "trajectory frames are temporally autocorrelated. Treat these tests "
        "as descriptive rather than independent validation.\n\n"
    )

    summary_file.write("Cluster occupancy:\n")
    for row in cluster_summary_table.itertuples(index=False):
        summary_file.write(
            f"  Cluster {row.cluster}: {row.n_frames} frames "
            f"({row.occupancy_percent:.4f}%)\n"
        )

    summary_file.write("\nInput files:\n")
    summary_file.write(f"  PSF: {TOPOLOGY}\n")
    summary_file.write(f"  Energy log (audit only): {ENERGY_LOG}\n")
    for trajectory_number, trajectory_path in enumerate(
        TRAJECTORIES,
        start=1,
    ):
        summary_file.write(
            f"  DCD {trajectory_number}: {trajectory_path}\n"
        )

    summary_file.write("\nSelections:\n")
    summary_file.write(f"  PETase21 receptor: {RECEPTOR_SEL}\n")
    summary_file.write(f"  PETase21 fit: {RECEPTOR_FIT_SEL}\n")
    summary_file.write(f"  Ligand all atoms: {LIGAND_ALL_SEL}\n")
    summary_file.write(f"  Ligand PCA atoms: {LIGAND_PCA_SEL}\n")
    summary_file.write(
        "  Representative PETase21 + ligand: "
        f"{REPRESENTATIVE_COMPLEX_SEL}\n"
    )

    summary_file.write("\nRepresentative structures:\n")
    summary_file.write(
        "  Each cluster has a ligand-only PCA-coordinate PDB and a "
        "PETase21 + PET dimer PDB from the same physical trajectory frame.\n"
    )
    summary_file.write(
        "  The representative frame is reconstructed across PBC, aligned on "
        "the PETase21 backbone, and translated so the PETase21 COM is at "
        "(0, 0, 0); receptor-ligand geometry is preserved.\n"
    )
    summary_file.write(
        "  The one-frame full-system DCD retains original PSF atom order "
        "and can be analyzed together with the input PSF.\n"
    )
    for row in representatives_table.itertuples(index=False):
        summary_file.write(
            f"  Cluster {row.cluster}: frame {row.trajectory_frame}, "
            f"time {row.time_ns:.6f} ns\n"
        )
        summary_file.write(
            f"    PETase21 + ligand PDB: "
            f"{row.representative_PETase21_ligand_pdb}\n"
        )
        if row.representative_full_system_dcd:
            summary_file.write(
                f"    Full-system DCD: "
                f"{row.representative_full_system_dcd}\n"
            )

# ============================================================
# FINAL REPORT
# ============================================================

print("\nAnalysis completed successfully.")
print(f"Selected clusters: {best_k}")
print(f"Best silhouette score: {best_silhouette:.6f}")
print(
    "PERMANOVA: "
    f"pseudo-F = {permanova_result['pseudo_F']:.6f}, "
    f"R^2 = {permanova_result['R_squared']:.6f}, "
    f"p = {permanova_result['p_value']:.6f}"
)
print(
    "PERMDISP: "
    f"F = {permdisp_result['F']:.6f}, "
    f"eta^2 = {permdisp_result['eta_squared']:.6f}, "
    f"p = {permdisp_result['p_value']:.6f}"
)
print(f"\nResults were saved in:\n{OUTPUT_DIR}")
print("\nMain outputs:")
print(f"  {SCORES_CSV}")
print(f"  {VARIANCE_CSV}")
print(f"  {CLUSTER_SUMMARY_CSV}")
print(f"  {REPRESENTATIVES_CSV}")
print(
    "  Per-cluster representative PETase21 + ligand PDB files and "
    "one-frame full-system DCD files"
)
print(f"  {PERMANOVA_CSV}")
print(f"  {PERMDISP_CSV}")
print(f"  {PERMDISP_GROUPS_CSV}")
print(f"  {MULTIVARIATE_TXT}")
print(f"  {PCA_CLUSTER_PNG}")
print(f"  {PCA_CLUSTER_SQUARE_PNG}")
print(f"  {PCA_TIME_PNG}")
print(f"  {PCA_TIME_SQUARE_PNG}")
print(f"  {SCREE_PNG}")
print(f"  {SCREE_SQUARE_PNG}")
print(f"  {PCS_TIME_PNG}")
print(f"  {PCS_TIME_SQUARE_PNG}")
print(f"  {CLUSTER_OCCUPANCY_PNG}")
print(f"  {CLUSTER_OCCUPANCY_SQUARE_PNG}")
print(f"  {SUMMARY_TXT}")
print(f"  {TIME_AXIS_METADATA}")
