#!/usr/bin/env python3
import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import queue as queue_module
import signal
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

RDLogger.DisableLog("rdApp.warning")

DRUG_3D_ATOM_COUNT = 64
DRUG_3D_FEATURE_DIM = 128
DRUG_3D_CONFORMER_COUNT = 8
DRUG_3D_MAX_HEAVY_ATOMS = 160
DRUG_1D_TOKEN_COUNT = 354
DRUG_1D_FEATURE_DIM = 128
PROTEIN_MAX_LEN = 1200
PROTEIN_FEATURE_LEN = 128
PROTEIN_REPLACE_AA = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})
FEATURE_SCHEMA_VERSION = 3

# ---- D1: geometry-aware atom features (only used when --drug3d_feature_variant != "vanilla") ----
# There are two geometry variants:
#
#   geo_v1 (D1 minimal fix): whole-molecule SE(3)-invariant descriptors
#     broadcast to every atom of the conformer. Makes feature[k] vary across K
#     but leaves feature[k, i] atom-invariant on the geo block. 10 dims / atom.
#
#   geo_v2 (per-atom local geometry, this PR): SE(3)-invariant descriptors
#     computed independently for each atom of each conformer, so
#     feature[k, i] varies across BOTH k and i on the geo block. This gives
#     the EGNN / score MLP a per-(atom, conformer) K-varying signal, which
#     is what the original vanilla cache was missing. 12 dims / atom.
#
# DRUG_3D_GEO_RBF_CENTERS spans a typical bond-to-mid-range neighborhood (Å).
# Choosing fixed centers (rather than learnable) keeps the feature reproducible across cache builds.
DRUG_3D_GEO_RBF_CENTERS = (1.0, 1.5, 2.0, 2.5, 3.5, 5.0)  # 6 centers
DRUG_3D_GEO_RBF_GAMMA = 1.0                                # exp(-gamma * (d - c)^2)
DRUG_3D_GEO_FEATURE_DIM = 4 + len(DRUG_3D_GEO_RBF_CENTERS)  # min/mean/std/max + RBF sums -> 10

# geo_v2 per-atom local geometry:
DRUG_3D_GEO_V2_FEATURE_DIM = 12          # see per_atom_geometry_descriptor layout below
DRUG_3D_GEO_V2_DENSITY_RADII = (2.0, 4.0, 6.0)   # three scales for local density
DRUG_3D_GEO_V2_RBF_CENTERS = (1.5, 2.5)          # nn1 RBFs (single bond, 1-3 distance)
DRUG_3D_GEO_V2_RBF_SIGMA = 0.5                   # Å
DRUG_3D_GEO_V2_ANGLE_RADIUS = 4.0                # Å, defines "local neighbors" for triplet angles
DRUG_3D_GEO_V2_PLANE_RADIUS = 3.0                # Å, defines the atom-local plane-fit neighborhood

# variants -> (cache_filename_suffix, geometry_mode) where geometry_mode in {"none", "v1", "v2"}
DRUG_3D_FEATURE_VARIANTS = {
    "vanilla": ("", "none"),
    "geo_v1": ("_geo_v1", "v1"),
    "geo_v2": ("_geo_v2", "v2"),
}


def default_drug3d_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2))


class Drug3DTimeout(RuntimeError):
    pass


def _drug3d_timeout_handler(signum, frame):
    raise Drug3DTimeout("drug3d conformer generation timed out")


def format_protein_sequence(sequence):
    sequence = "".join(str(sequence).upper().split()).translate(PROTEIN_REPLACE_AA)
    return " ".join(sequence)


def pool_token_features(token_features, output_len, output_dim=None):
    if token_features.numel() == 0:
        output_dim = output_dim or int(token_features.size(-1))
        return torch.zeros(output_len, output_dim, dtype=torch.float32)

    token_features = token_features.to(torch.float32)
    if output_dim is None:
        token_features = token_features.transpose(0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool1d(token_features, output_len)
        return pooled.squeeze(0).transpose(0, 1)

    token_features = token_features.unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(token_features, (output_len, output_dim))
    return pooled.squeeze(0).squeeze(0)


def valid_token_features(sequence_output, attention_mask, remove_special_tokens=True):
    valid = sequence_output[attention_mask.bool()]
    if remove_special_tokens and valid.size(0) > 2:
        valid = valid[1:-1]
    return valid


def cache_tensor(entry):
    if isinstance(entry, dict):
        return entry.get("feature")
    return entry


def valid_smiles_entry(entry):
    feature = cache_tensor(entry)
    return torch.is_tensor(feature) and tuple(feature.shape) == (DRUG_1D_TOKEN_COUNT, DRUG_1D_FEATURE_DIM)


def valid_protein_entry(entry):
    if not isinstance(entry, dict):
        return False

    feature = entry.get("feature")
    mask = entry.get("mask")
    return (
        torch.is_tensor(feature)
        and tuple(feature.shape) == (PROTEIN_FEATURE_LEN, 1024)
        and torch.is_tensor(mask)
        and tuple(mask.shape) == (PROTEIN_FEATURE_LEN,)
    )


def valid_drug3d_entry(entry):
    if not isinstance(entry, dict):
        return False

    feature = entry.get("feature")
    coor = entry.get("coor")
    conf_mask = entry.get("conf_mask")
    energy = entry.get("energy")
    return (
        entry.get("schema_version") == FEATURE_SCHEMA_VERSION
        and torch.is_tensor(feature)
        and tuple(feature.shape) == (DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM)
        and torch.is_tensor(coor)
        and tuple(coor.shape) == (DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, 3)
        and torch.is_tensor(conf_mask)
        and tuple(conf_mask.shape) == (DRUG_3D_CONFORMER_COUNT,)
        and torch.is_tensor(energy)
        and tuple(energy.shape) == (DRUG_3D_CONFORMER_COUNT,)
    )


def prune_invalid_cache(cache, validator, cache_name):
    valid_cache = {}
    dropped = 0
    for key, value in cache.items():
        if validator(value):
            valid_cache[key] = value
        else:
            dropped += 1

    if dropped:
        print(f"[cache] drop {dropped} invalid {cache_name} entries; they will be rebuilt")
    return valid_cache


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = "Unknown"
    return [x == s for s in allowable_set]


def one_of_k_encoding(x, allowable_set):
    return [1 if x == s else 0 for s in allowable_set]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As",
                "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti",
                "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt",
                "Hg", "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def geometry_descriptor(coordinates):
    """Compute per-atom SE(3)-invariant geometric descriptors from absolute coords.

    Returns (N, DRUG_3D_GEO_FEATURE_DIM) float64 array. The result depends on the
    conformer geometry (and so varies across K), but is invariant to global
    rotation / translation because it is derived from pairwise distances only.

    Layout per atom i:
        [d_min_i, d_mean_i, d_std_i, d_max_i,
         Σ_{j != i} exp(-γ (d_ij - c_b)^2)   for b = 1..len(RBF_CENTERS)]
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    n = coordinates.shape[0]
    geo = np.zeros((n, DRUG_3D_GEO_FEATURE_DIM), dtype=np.float64)
    if n <= 1:
        return geo

    # pairwise distance matrix; diagonal set to a large value so it doesn't
    # affect min / RBF sums.
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    mask_self = np.eye(n, dtype=bool)
    safe_dist = np.where(mask_self, np.inf, dist)

    d_min = safe_dist.min(axis=1)
    finite = np.where(mask_self, 0.0, dist)
    sum_d = finite.sum(axis=1)
    denom = float(n - 1)
    d_mean = sum_d / denom
    var_d = ((finite - d_mean[:, None]) ** 2).sum(axis=1) / denom - (d_mean ** 2) * 0.0
    var_d = np.maximum(var_d, 0.0)
    d_std = np.sqrt(var_d)
    masked = np.where(mask_self, -np.inf, dist)
    d_max = masked.max(axis=1)

    geo[:, 0] = d_min
    geo[:, 1] = d_mean
    geo[:, 2] = d_std
    geo[:, 3] = d_max

    # RBF sums: Σ_{j != i} exp(-γ (d_ij - c)^2). The (j == i) term contributes
    # exp(-γ c^2), which is constant across atoms (and small), so subtracting
    # the self diagonal post-hoc is fine.
    gamma = float(DRUG_3D_GEO_RBF_GAMMA)
    for b, center in enumerate(DRUG_3D_GEO_RBF_CENTERS):
        rbf_full = np.exp(-gamma * (dist - center) ** 2)
        rbf_self = np.exp(-gamma * center ** 2)  # scalar; from j == i
        geo[:, 4 + b] = rbf_full.sum(axis=1) - rbf_self

    return geo


def per_atom_geometry_descriptor(coordinates):
    """Compute per-atom SE(3)-invariant *local* geometric descriptors.

    Returns (N, DRUG_3D_GEO_V2_FEATURE_DIM) float64 array. Unlike
    :func:`geometry_descriptor` whose output was later broadcast, this one
    gives each atom of each conformer its own descriptor vector, so
    feature[k, i] on the geo block varies along BOTH the conformer axis k
    AND the atom axis i.

    All entries are derived from pairwise distances or angles, so the
    result is invariant to global rotation / translation (SE(3)-inv).

    Layout per atom i (12 dims total):
        0  dist_to_COM               center-of-mass distance
        1  dist_to_COM_norm          (0) / radius_of_gyration
        2  local_density_2A          #(j != i) with d_ij <= 2A / (N - 1)
        3  local_density_4A          same at 4A
        4  local_density_6A          same at 6A
        5  nn1_dist                  distance to nearest other atom
        6  nn1_rbf_1.5A              exp(-((5) - 1.5)^2 / 2*sigma^2)
        7  nn1_rbf_2.5A              exp(-((5) - 2.5)^2 / 2*sigma^2)
        8  mean_angle                mean of triplet angles (i as apex,
                                     two neighbors within angle-radius)
        9  std_angle                 std of the same triplet angles
        10 planarity                 |x_i - plane| for plane fit through
                                     atoms within plane-radius of i
        11 gyration_contrib          |x_i - x_COM|^2 / sum_j |x_j - x_COM|^2

    Physical rationale is documented at length in the collapse-fix design
    doc; every dim is a single physically meaningful geometric quantity.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    n = coordinates.shape[0]
    out = np.zeros((n, DRUG_3D_GEO_V2_FEATURE_DIM), dtype=np.float64)
    if n <= 1:
        return out

    # ---- pairwise distance matrix (upper bound on many ops below) ----
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    self_mask = np.eye(n, dtype=bool)
    safe_dist = np.where(self_mask, np.inf, dist)   # ignore self on min / kNN

    # ---- 1..2: distance to COM & normalized radius ----
    com = coordinates.mean(axis=0)                  # (3,)
    d_com = np.linalg.norm(coordinates - com, axis=1)   # (n,)
    r_gyration = float(np.sqrt(np.mean(d_com ** 2))) or 1.0
    out[:, 0] = d_com
    out[:, 1] = d_com / r_gyration

    # ---- 3..5: local density at three scales ----
    denom = float(n - 1) if n > 1 else 1.0
    for j, radius in enumerate(DRUG_3D_GEO_V2_DENSITY_RADII):
        # Count neighbors strictly inside radius (self excluded via safe_dist=inf)
        within = (safe_dist <= radius).sum(axis=1).astype(np.float64)
        out[:, 2 + j] = within / denom

    # ---- 6..8: nearest-neighbor distance + two RBFs ----
    nn1 = safe_dist.min(axis=1)                     # (n,)
    out[:, 5] = nn1
    sigma = float(DRUG_3D_GEO_V2_RBF_SIGMA)
    denom_rbf = 2.0 * sigma * sigma
    for j, center in enumerate(DRUG_3D_GEO_V2_RBF_CENTERS):
        out[:, 6 + j] = np.exp(-((nn1 - center) ** 2) / denom_rbf)

    # ---- 9..10: triplet angle stats (i as apex, neighbors within radius) ----
    # For each atom i, take neighbor set N_i = {j : d_ij <= angle_radius, j != i}.
    # Compute the pairwise angles (v_ij, v_ik) for j < k in N_i and take
    # mean / std over the resulting set. This captures sp2 / sp3 / bond-angle
    # deformation without depending on the RDKit adjacency (which is fixed
    # across conformers and would not distinguish conformer geometries).
    r_ang = float(DRUG_3D_GEO_V2_ANGLE_RADIUS)
    for i in range(n):
        nbr_idx = np.where((safe_dist[i] <= r_ang))[0]
        if nbr_idx.size < 2:
            continue
        v = coordinates[nbr_idx] - coordinates[i]           # (m, 3)
        norms = np.linalg.norm(v, axis=1)                   # (m,)
        norms_safe = np.where(norms > 1e-8, norms, 1.0)
        u = v / norms_safe[:, None]                         # unit vectors
        cos = np.clip(u @ u.T, -1.0, 1.0)
        iu = np.triu_indices(nbr_idx.size, k=1)
        cos_pairs = cos[iu]
        valid = (norms[iu[0]] > 1e-8) & (norms[iu[1]] > 1e-8)
        cos_pairs = cos_pairs[valid]
        if cos_pairs.size == 0:
            continue
        angles = np.arccos(cos_pairs)
        out[i, 8] = float(angles.mean())
        out[i, 9] = float(angles.std())

    # ---- 11: planarity (distance from atom to plane fitted through nbrs) ----
    r_plane = float(DRUG_3D_GEO_V2_PLANE_RADIUS)
    for i in range(n):
        nbr_idx = np.where(safe_dist[i] <= r_plane)[0]
        if nbr_idx.size < 3:
            continue
        pts = coordinates[nbr_idx]                  # (m, 3)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        try:
            _, sv, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if sv.size < 3 or (sv[-1] < 1e-8 and sv[1] < 1e-8):
            continue
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        out[i, 10] = float(abs(np.dot(coordinates[i] - centroid, normal)))

    # ---- 12: gyration contribution ----
    total_r2 = float(np.sum(d_com ** 2))
    if total_r2 > 1e-12:
        out[:, 11] = (d_com ** 2) / total_r2
    else:
        out[:, 11] = 0.0

    return out


def pool_drug_3d_sample(feature, coor):
    if feature.size == 0 or coor.size == 0:
        return (
            torch.zeros(DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM, dtype=torch.float32),
            torch.zeros(DRUG_3D_ATOM_COUNT, 3, dtype=torch.float32),
        )

    feature = torch.as_tensor(feature, dtype=torch.float32)
    coor = torch.as_tensor(coor, dtype=torch.float32)

    if feature.ndim != 2 or coor.ndim != 2 or coor.size(-1) != 3 or feature.size(0) != coor.size(0):
        return (
            torch.zeros(DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM, dtype=torch.float32),
            torch.zeros(DRUG_3D_ATOM_COUNT, 3, dtype=torch.float32),
        )

    feature = feature.transpose(0, 1).unsqueeze(0)                      # (1, D_feat, N)
    feature = F.adaptive_avg_pool1d(feature, DRUG_3D_ATOM_COUNT).squeeze(0).transpose(0, 1)
    if feature.size(1) < DRUG_3D_FEATURE_DIM:
        feature = F.pad(feature, (0, DRUG_3D_FEATURE_DIM - feature.size(1)))
    elif feature.size(1) > DRUG_3D_FEATURE_DIM:
        feature = feature[:, :DRUG_3D_FEATURE_DIM]

    # Use the SAME adaptive_avg_pool1d bin structure for coor so that
    # feature[i] and coor[i] represent the same bin (bin-mean of the same
    # subset of original atoms). Historically coor used
    # F.interpolate(mode="nearest") which picked one original atom per bin
    # while feature averaged, so feature[i] and coor[i] pointed to
    # semantically different summaries of the atoms in bin i. Matching the
    # pool operator makes the geo_v2 dim-0 (bin-avg of per-atom dist_to_COM)
    # numerically consistent with dist(bin-avg-coor, molecule COM) up to
    # Jensen slack.
    coor = coor.transpose(0, 1).unsqueeze(0)                            # (1, 3, N)
    coor = F.adaptive_avg_pool1d(coor, DRUG_3D_ATOM_COUNT).squeeze(0).transpose(0, 1)  # (64, 3)

    return feature, coor
def build_raw_3d_conformers(smile, num_conformers, geometry_mode="none"):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smile}")

        heavy_atoms = mol.GetNumAtoms()
        if DRUG_3D_MAX_HEAVY_ATOMS > 0 and heavy_atoms > DRUG_3D_MAX_HEAVY_ATOMS:
            raise ValueError(
                f"skip ETKDG for molecule with {heavy_atoms} heavy atoms "
                f"(limit {DRUG_3D_MAX_HEAVY_ATOMS})"
            )

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.5
        params.numThreads = 1
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))

        if len(ids) == 0:
            raise ValueError(f"Embedding failed for SMILES: {smile}")

        conformers = []
        has_uff_params = AllChem.UFFHasAllMoleculeParams(mol)
        for conf_id in ids:
            energy = 0.0
            try:
                if has_uff_params:
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=100)
                    force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    energy = float(force_field.CalcEnergy())
            except Exception:
                energy = 0.0

            base_features = []
            coordinates = []

            conf = mol.GetConformer(conf_id)
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                feature = atom_features(atom)
                base_features.append(feature / max(sum(feature), 1.0))

                pos = conf.GetAtomPosition(atom_idx)
                coordinates.append(np.array([pos.x, pos.y, pos.z]))

            base_features = np.array(base_features, dtype=np.float64)
            coordinates = np.array(coordinates, dtype=np.float64)
            if geometry_mode == "v1":
                # Whole-molecule descriptors broadcast to every atom.
                geo = geometry_descriptor(coordinates)
                combined = np.concatenate([base_features, geo], axis=1)
            elif geometry_mode == "v2":
                # Per-atom local descriptors: feature[k, i] varies over both k and i.
                geo = per_atom_geometry_descriptor(coordinates)
                combined = np.concatenate([base_features, geo], axis=1)
            else:
                combined = base_features

            conformers.append(
                {
                    "feature": combined,
                    "coor": coordinates,
                    "energy": energy,
                }
            )

        conformers.sort(key=lambda item: item["energy"])
        return conformers[:num_conformers], True

    except Exception as e:
        logging.warning("Failed to build 3D conformers for SMILES %s: %s", smile, e)
        return [], False


def build_drug3d_entry(smile, num_conformers, timeout_seconds=0, geometry_mode="none"):
    previous_handler = None
    if timeout_seconds and timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _drug3d_timeout_handler)
        signal.alarm(int(timeout_seconds))

    try:
        raw_conformers, ok = build_raw_3d_conformers(smile, num_conformers, geometry_mode=geometry_mode)
    except Drug3DTimeout as exc:
        logging.warning("Timeout building 3D conformers for SMILES %s: %s", smile, exc)
        raw_conformers, ok = [], False
    finally:
        if timeout_seconds and timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    features = torch.zeros(
        num_conformers, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM, dtype=torch.float32
    )
    coors = torch.zeros(num_conformers, DRUG_3D_ATOM_COUNT, 3, dtype=torch.float32)
    conf_mask = torch.zeros(num_conformers, dtype=torch.float32)
    energies = torch.zeros(num_conformers, dtype=torch.float32)

    for idx, conformer in enumerate(raw_conformers[:num_conformers]):
        feature, coor = pool_drug_3d_sample(conformer["feature"], conformer["coor"])
        features[idx] = feature
        coors[idx] = coor
        conf_mask[idx] = 1.0
        energies[idx] = float(conformer["energy"])

    if conf_mask.sum().item() == 0:
        conf_mask[0] = 1.0

    valid_energy = energies[conf_mask.bool()]
    if valid_energy.numel() > 1:
        mean = valid_energy.mean()
        std = valid_energy.std(unbiased=False).clamp_min(1e-6)
        energies[conf_mask.bool()] = (valid_energy - mean) / std

    return {
        "feature": features.cpu(),
        "coor": coors.cpu(),
        "conf_mask": conf_mask.cpu(),
        "energy": energies.cpu(),
        "ok": ok,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }


def build_empty_drug3d_entry(num_conformers):
    features = torch.zeros(
        num_conformers, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM, dtype=torch.float32
    )
    coors = torch.zeros(num_conformers, DRUG_3D_ATOM_COUNT, 3, dtype=torch.float32)
    conf_mask = torch.zeros(num_conformers, dtype=torch.float32)
    conf_mask[0] = 1.0
    energies = torch.zeros(num_conformers, dtype=torch.float32)
    return {
        "feature": features.cpu(),
        "coor": coors.cpu(),
        "conf_mask": conf_mask.cpu(),
        "energy": energies.cpu(),
        "ok": False,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }


def build_drug3d_entry_worker(smile, num_conformers, timeout_seconds, geometry_mode):
    return smile, build_drug3d_entry(smile, num_conformers, timeout_seconds, geometry_mode=geometry_mode)


def serialize_drug3d_entry(entry):
    return {
        "feature": entry["feature"].numpy(),
        "coor": entry["coor"].numpy(),
        "conf_mask": entry["conf_mask"].numpy(),
        "energy": entry["energy"].numpy(),
        "ok": bool(entry.get("ok", False)),
        "schema_version": int(entry.get("schema_version", FEATURE_SCHEMA_VERSION)),
    }


def deserialize_drug3d_entry(entry):
    return {
        "feature": torch.as_tensor(entry["feature"], dtype=torch.float32),
        "coor": torch.as_tensor(entry["coor"], dtype=torch.float32),
        "conf_mask": torch.as_tensor(entry["conf_mask"], dtype=torch.float32),
        "energy": torch.as_tensor(entry["energy"], dtype=torch.float32),
        "ok": bool(entry.get("ok", False)),
        "schema_version": int(entry.get("schema_version", FEATURE_SCHEMA_VERSION)),
    }


def build_drug3d_process_target(smile, num_conformers, queue, geometry_mode):
    try:
        queue.put(
            serialize_drug3d_entry(
                build_drug3d_entry(smile, num_conformers, timeout_seconds=0, geometry_mode=geometry_mode)
            )
        )
    except Exception as exc:
        logging.warning("Failed isolated drug3d process for SMILES %s: %s", smile, exc)
        queue.put(serialize_drug3d_entry(build_empty_drug3d_entry(num_conformers)))


def build_drug3d_entry_isolated(smile, num_conformers, timeout_seconds, geometry_mode="none"):
    if not timeout_seconds or timeout_seconds <= 0:
        return build_drug3d_entry(smile, num_conformers, timeout_seconds=0, geometry_mode=geometry_mode)

    context_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(context_name)
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=build_drug3d_process_target,
        args=(smile, num_conformers, queue, geometry_mode),
    )
    process.start()

    try:
        entry = queue.get(timeout=timeout_seconds)
    except queue_module.Empty:
        process.terminate()
        process.join(5)
        logging.warning("Timeout building 3D conformers for SMILES %s after %ss", smile, timeout_seconds)
        return build_empty_drug3d_entry(num_conformers)
    except (EOFError, OSError) as exc:
        process.terminate()
        process.join(5)
        logging.warning("Failed receiving isolated drug3d result for SMILES %s: %s", smile, exc)
        return build_empty_drug3d_entry(num_conformers)

    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        logging.warning("Timed out waiting for isolated drug3d process cleanup for SMILES %s", smile)
        return build_empty_drug3d_entry(num_conformers)

    if process.exitcode != 0:
        logging.warning("Failed building 3D conformers for SMILES %s with exit code %s", smile, process.exitcode)
        return build_empty_drug3d_entry(num_conformers)

    return deserialize_drug3d_entry(entry)


def save_drug3d_cache(cache, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, output_path)


def encode_drug3d(
    smiles_list,
    output_path: Path,
    overwrite: bool,
    num_conformers: int,
    workers: int,
    save_every: int,
    timeout_seconds: int,
    geometry_mode: str = "none",
):
    cache = prune_invalid_cache(maybe_load_cache(output_path, overwrite), valid_drug3d_entry, "drug 3D")
    missing = [x for x in smiles_list if x not in cache]

    print(
        f"[drug3d] total={len(smiles_list)} existing={len(cache)} missing={len(missing)} "
        f"num_conformers={num_conformers} workers={workers} timeout={timeout_seconds}s "
        f"geometry_mode={geometry_mode}"
    )
    if not missing:
        return cache

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_sequential(smiles_to_build, completed=0):
        total = len(missing)
        for offset, smile in enumerate(smiles_to_build, start=1):
            idx = completed + offset
            cache[smile] = build_drug3d_entry(
                smile, num_conformers, timeout_seconds, geometry_mode=geometry_mode
            )

            if idx == 1 or idx == total or idx % 100 == 0:
                print(f"[drug3d] {idx}/{total}")
            if save_every > 0 and idx % save_every == 0:
                save_drug3d_cache(cache, output_path)
                print(f"[drug3d] checkpoint saved: {output_path}")

    try:
        if workers <= 1:
            run_sequential(missing)
        else:
            completed_smiles = set()
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
            )
            try:
                pending_iter = iter(missing)
                future_to_smile = {}
                max_in_flight = max(workers * 4, workers)

                def submit_next():
                    try:
                        next_smile = next(pending_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        build_drug3d_entry_worker,
                        next_smile,
                        num_conformers,
                        timeout_seconds,
                        geometry_mode,
                    )
                    future_to_smile[future] = next_smile
                    return True

                for _ in range(max_in_flight):
                    if not submit_next():
                        break

                idx = 0
                while future_to_smile:
                    for future in as_completed(tuple(future_to_smile)):
                        smile = future_to_smile.pop(future)
                        idx += 1
                        break

                    try:
                        smile, entry = future.result()
                    except Exception as exc:
                        print(
                            f"[drug3d] parallel worker failed ({exc}); "
                            "fallback to sequential generation for remaining molecules"
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        remaining = [item for item in missing if item not in completed_smiles]
                        run_sequential(remaining, completed=len(completed_smiles))
                        break

                    cache[smile] = entry
                    completed_smiles.add(smile)

                    if idx == 1 or idx == len(missing) or idx % 100 == 0:
                        print(f"[drug3d] {idx}/{len(missing)}")
                    if save_every > 0 and idx % save_every == 0:
                        save_drug3d_cache(cache, output_path)
                        print(f"[drug3d] checkpoint saved: {output_path}")

                    submit_next()

                executor.shutdown()
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        print("[drug3d] interrupted; saving partial cache before exit")
        save_drug3d_cache(cache, output_path)
        raise

    save_drug3d_cache(cache, output_path)
    print(f"[drug3d] saved: {output_path}")
    return cache
def masked_mean_pool(sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
    masked_output = sequence_output * mask
    return masked_output.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def load_unique_entities(data_dir: Path):
    smiles = []
    proteins = []

    for split in ("train", "val", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        df = pd.read_csv(csv_path)
        if "SMILES" not in df.columns or "Protein" not in df.columns:
            raise ValueError(f"{csv_path} must contain 'SMILES' and 'Protein' columns")

        smiles.extend(df["SMILES"].astype(str).tolist())
        proteins.extend(df["Protein"].astype(str).tolist())

    unique_smiles = list(dict.fromkeys(smiles))
    unique_proteins = list(dict.fromkeys(proteins))
    return unique_smiles, unique_proteins


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def maybe_load_cache(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        print(f"[cache] load existing: {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            print(f"[cache] failed to load {path}: {exc}")
            print("[cache] ignore broken cache and rebuild it")
    return {}


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def encode_smiles(
    smiles_list,
    model_dir: Path,
    device: str,
    batch_size: int,
    output_path: Path,
    overwrite: bool,
):
    cache = prune_invalid_cache(maybe_load_cache(output_path, overwrite), valid_smiles_entry, "SMILES")
    missing = [x for x in smiles_list if x not in cache]

    print(f"[smiles] total={len(smiles_list)} existing={len(cache)} missing={len(missing)}")
    if not missing:
        return cache

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    total_batches = math.ceil(len(missing) / batch_size)

    with torch.inference_mode():
        for idx, batch_smiles in enumerate(batched(missing, batch_size), start=1):
            tokenized = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = model(**tokenized).last_hidden_state

            for item_idx, smile in enumerate(batch_smiles):
                token_features = valid_token_features(
                    outputs[item_idx],
                    tokenized["attention_mask"][item_idx],
                    remove_special_tokens=True,
                )
                feature = pool_token_features(
                    token_features,
                    output_len=DRUG_1D_TOKEN_COUNT,
                    output_dim=DRUG_1D_FEATURE_DIM,
                )
                cache[smile] = feature.to(torch.float16).cpu()

            if idx == 1 or idx == total_batches or idx % 10 == 0:
                print(f"[smiles] batch {idx}/{total_batches}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, output_path)
    print(f"[smiles] saved: {output_path}")
    return cache


def encode_proteins(
    protein_list,
    model_dir: Path,
    device: str,
    batch_size: int,
    output_path: Path,
    overwrite: bool,
):
    cache = prune_invalid_cache(maybe_load_cache(output_path, overwrite), valid_protein_entry, "protein")
    missing = [x for x in protein_list if x not in cache]

    print(f"[protein] total={len(protein_list)} existing={len(cache)} missing={len(missing)}")
    if not missing:
        return cache

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = BertModel.from_pretrained(model_dir)
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    total_batches = math.ceil(len(missing) / batch_size)

    with torch.inference_mode():
        for idx, batch_proteins in enumerate(batched(missing, batch_size), start=1):
            formatted_proteins = [format_protein_sequence(protein) for protein in batch_proteins]
            tokenized = tokenizer(
                formatted_proteins,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=PROTEIN_MAX_LEN,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = model(**tokenized).last_hidden_state

            for item_idx, protein in enumerate(batch_proteins):
                token_features = valid_token_features(
                    outputs[item_idx],
                    tokenized["attention_mask"][item_idx],
                    remove_special_tokens=True,
                )
                feature = pool_token_features(token_features, output_len=PROTEIN_FEATURE_LEN)
                cache[protein] = {
                    "feature": feature.to(torch.float16).cpu(),
                    "mask": torch.ones(PROTEIN_FEATURE_LEN, dtype=torch.float32),
                }

            if idx == 1 or idx == total_batches or idx % 10 == 0:
                print(f"[protein] batch {idx}/{total_batches}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, output_path)
    print(f"[protein] saved: {output_path}")
    return cache


def main():
    global DRUG_1D_TOKEN_COUNT, DRUG_1D_FEATURE_DIM, PROTEIN_FEATURE_LEN
    global DRUG_3D_CONFORMER_COUNT, DRUG_3D_MAX_HEAVY_ATOMS

    
    '''
    USAGE：
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
    python scripts/pre_extract.py \
        --data sample \
        --split random \
        --device cuda:0
    '''
    parser = argparse.ArgumentParser(description="Build offline feature cache for LDM-DTI")
    parser.add_argument("--data", required=True, help="dataset path under data/, e.g. sample or datasets/bindingdb")
    parser.add_argument("--split", default="random", help="dataset split name")
    parser.add_argument("--device", default="cuda:0", help="encoding device, e.g. cuda:0 or cpu")
    parser.add_argument("--smiles_batch_size", type=int, default=512)
    parser.add_argument("--protein_batch_size", type=int, default=8)
    parser.add_argument("--drug_1d_token_count", type=int, default=DRUG_1D_TOKEN_COUNT)
    parser.add_argument("--drug_1d_feature_dim", type=int, default=DRUG_1D_FEATURE_DIM)
    parser.add_argument("--protein_feature_len", type=int, default=PROTEIN_FEATURE_LEN)
    parser.add_argument("--num_conformers", type=int, default=DRUG_3D_CONFORMER_COUNT)
    parser.add_argument("--drug3d_workers", type=int, default=default_drug3d_workers())
    parser.add_argument("--drug3d_save_every", type=int, default=500)
    parser.add_argument(
        "--drug3d_max_heavy_atoms",
        type=int,
        default=DRUG_3D_MAX_HEAVY_ATOMS,
        help="skip RDKit ETKDG for very large molecules; 0 disables this guard",
    )
    parser.add_argument(
        "--drug3d_timeout",
        type=int,
        default=60,
        help="maximum seconds per SMILES for RDKit 3D conformer generation; 0 disables timeout",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_drug3d", action="store_true", help="only rebuild drug 3D feature cache")
    parser.add_argument("--output_dir", default="cache/features")
    parser.add_argument("--chemberta_dir", default="model/ChemBERTa-77M-MTR")
    parser.add_argument("--protbert_dir", default="model/prot_bert")
    parser.add_argument(
        "--drug3d_feature_variant",
        default="vanilla",
        choices=sorted(DRUG_3D_FEATURE_VARIANTS.keys()),
        help=(
            "atom feature variant for the drug 3D cache. "
            "'vanilla' preserves the original K-invariant atom features (74d -> 128d pad); "
            "'geo_v1' concatenates whole-molecule SE(3)-invariant descriptors (10d) broadcast to every atom, "
            "so feature[k] varies across conformers K but is atom-invariant on the geo block; "
            "'geo_v2' concatenates per-atom SE(3)-invariant local geometry descriptors (12d), "
            "so feature[k, i] varies across BOTH conformer K and atom i on the geo block. "
            "Non-vanilla variants are stored under drug3d_features_<variant>.pt to avoid clobbering the main cache."
        ),
    )
    args = parser.parse_args()

    DRUG_1D_TOKEN_COUNT = args.drug_1d_token_count
    DRUG_1D_FEATURE_DIM = args.drug_1d_feature_dim
    PROTEIN_FEATURE_LEN = args.protein_feature_len
    DRUG_3D_CONFORMER_COUNT = args.num_conformers
    DRUG_3D_MAX_HEAVY_ATOMS = args.drug3d_max_heavy_atoms

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / args.data / args.split
    output_dir = repo_root / args.output_dir / args.data / args.split

    unique_smiles, unique_proteins = load_unique_entities(data_dir)
    print(f"[data] dir={data_dir}")
    print(f"[data] unique_smiles={len(unique_smiles)} unique_proteins={len(unique_proteins)}")

    smiles_cache_path = output_dir / "smiles_features.pt"
    protein_cache_path = output_dir / "protein_features.pt"
    variant_suffix, geometry_mode = DRUG_3D_FEATURE_VARIANTS[args.drug3d_feature_variant]
    drug3d_cache_path = output_dir / f"drug3d_features{variant_suffix}.pt"
    meta_path = output_dir / "meta.json" if args.drug3d_feature_variant == "vanilla" else output_dir / f"meta_drug3d{variant_suffix}.json"

    encode_smiles(
        unique_smiles,
        model_dir=repo_root / args.chemberta_dir,
        device=args.device,
        batch_size=args.smiles_batch_size,
        output_path=smiles_cache_path,
        overwrite=args.overwrite,
    )

    encode_proteins(
        unique_proteins,
        model_dir=repo_root / args.protbert_dir,
        device=args.device,
        batch_size=args.protein_batch_size,
        output_path=protein_cache_path,
        overwrite=args.overwrite,
    )

    encode_drug3d(
        unique_smiles,
        output_path=drug3d_cache_path,
        overwrite=args.overwrite or args.overwrite_drug3d,
        num_conformers=args.num_conformers,
        workers=args.drug3d_workers,
        save_every=args.drug3d_save_every,
        timeout_seconds=args.drug3d_timeout,
        geometry_mode=geometry_mode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "data": args.data,
        "split": args.split,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "unique_smiles": len(unique_smiles),
        "unique_proteins": len(unique_proteins),
        "smiles_cache": str(smiles_cache_path),
        "protein_cache": str(protein_cache_path),
        "drug3d_cache": str(drug3d_cache_path),
        "chemberta_dir": str(repo_root / args.chemberta_dir),
        "protbert_dir": str(repo_root / args.protbert_dir),
        "smiles_feature": {
            "type": "token_hidden_adaptive_pool",
            "shape": [DRUG_1D_TOKEN_COUNT, DRUG_1D_FEATURE_DIM],
            "dtype": "float16",
        },
        "protein_feature": {
            "type": "protbert_token_hidden_adaptive_pool",
            "shape": [PROTEIN_FEATURE_LEN, 1024],
            "dtype": "float16",
            "sequence_format": "space_separated_amino_acids_with_UZOB_to_X",
        },
        "drug3d_feature": {
            "type": "multi_conformer_atom_features_adaptive_pool_pad128_with_coordinates",
            "variant": args.drug3d_feature_variant,
            "geometry_mode": geometry_mode,
            "geo_feature_dim": (
                DRUG_3D_GEO_FEATURE_DIM if geometry_mode == "v1"
                else DRUG_3D_GEO_V2_FEATURE_DIM if geometry_mode == "v2"
                else 0
            ),
            "num_conformers": DRUG_3D_CONFORMER_COUNT,
            "max_heavy_atoms_for_etkdg": DRUG_3D_MAX_HEAVY_ATOMS,
            "shape": [DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM],
            "coor_shape": [DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, 3],
            "conf_mask_shape": [DRUG_3D_CONFORMER_COUNT],
            "energy_shape": [DRUG_3D_CONFORMER_COUNT],
            "dtype": "float32",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[done] meta saved: {meta_path}")


if __name__ == "__main__":
    main()
