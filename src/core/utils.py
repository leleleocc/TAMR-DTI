import os
import random
import numpy as np
import torch
import dgl
import logging
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DRUG_3D_ATOM_COUNT = 64
DRUG_3D_FEATURE_DIM = 128
DRUG_3D_CONFORMER_COUNT = 8


def _pool_drug_3d_sample(feature, coor):
    if getattr(feature, "size", 0) == 0 or getattr(coor, "size", 0) == 0:
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

    feature = feature.transpose(0, 1).unsqueeze(0)
    feature = F.adaptive_avg_pool1d(feature, DRUG_3D_ATOM_COUNT).squeeze(0).transpose(0, 1)
    if feature.size(1) < DRUG_3D_FEATURE_DIM:
        feature = F.pad(feature, (0, DRUG_3D_FEATURE_DIM - feature.size(1)))
    elif feature.size(1) > DRUG_3D_FEATURE_DIM:
        feature = feature[:, :DRUG_3D_FEATURE_DIM]

    coor = coor.unsqueeze(0).unsqueeze(1)
    coor = F.interpolate(coor, size=(DRUG_3D_ATOM_COUNT, 3), mode="nearest")
    coor = coor.squeeze(0).squeeze(0)

    return feature, coor


def _normalize_drug_3d_sample(feature, coor, conf_mask=None, energy=None):
    norm_feature = torch.zeros(
        DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM, dtype=torch.float32
    )
    norm_coor = torch.zeros(DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, 3, dtype=torch.float32)
    norm_conf_mask = torch.zeros(DRUG_3D_CONFORMER_COUNT, dtype=torch.float32)
    norm_energy = torch.zeros(DRUG_3D_CONFORMER_COUNT, dtype=torch.float32)

    if torch.is_tensor(feature) and torch.is_tensor(coor):
        if tuple(feature.shape) == (DRUG_3D_CONFORMER_COUNT, DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM):
            norm_feature = feature.float()
            norm_coor = coor.float()
            if torch.is_tensor(conf_mask) and tuple(conf_mask.shape) == (DRUG_3D_CONFORMER_COUNT,):
                norm_conf_mask = conf_mask.float()
            else:
                norm_conf_mask[:] = 1.0
            if torch.is_tensor(energy) and tuple(energy.shape) == (DRUG_3D_CONFORMER_COUNT,):
                norm_energy = energy.float()
        elif tuple(feature.shape) == (DRUG_3D_ATOM_COUNT, DRUG_3D_FEATURE_DIM) and tuple(coor.shape) == (DRUG_3D_ATOM_COUNT, 3):
            pooled_feature, pooled_coor = _pool_drug_3d_sample(feature, coor)
            norm_feature[0] = pooled_feature
            norm_coor[0] = pooled_coor
            norm_conf_mask[0] = 1.0

    if norm_conf_mask.sum().item() == 0:
        norm_conf_mask[0] = 1.0

    return norm_feature, norm_coor, norm_conf_mask, norm_energy


#def graph_collate_func(x):
    #d, p, y = zip(*x)
   # d = dgl.batch(d)
    #return d, torch.tensor(np.array(p)), torch.tensor(y)
def graph_collate_func(x):
    feature_vectors, feature, coor, conf_mask, energy, d, v_p, protein_mask, y = zip(*x)
    drug_3d_features = [
        _normalize_drug_3d_sample(f, c, m, e) for f, c, m, e in zip(feature, coor, conf_mask, energy)
    ]
    feature = torch.stack([item[0] for item in drug_3d_features], dim=0)
    coor = torch.stack([item[1] for item in drug_3d_features], dim=0)
    conf_mask = torch.stack([item[2] for item in drug_3d_features], dim=0)
    energy = torch.stack([item[3] for item in drug_3d_features], dim=0)
    d = dgl.batch(d)
    v_p = torch.stack(v_p, dim=0)
    protein_mask = torch.tensor(np.array(protein_mask), dtype=torch.float32)
    y = torch.tensor(np.array(y))
    if feature_vectors[0].ndim == 2:
        pooled_items = []
        for item in feature_vectors:
            if item.shape != (354, 128):
                item = F.adaptive_avg_pool2d(item.unsqueeze(0).unsqueeze(0), (354, 128)).squeeze(0).squeeze(0)
            pooled_items.append(item)
        feature_vectors = torch.stack(pooled_items, dim=0)
    else:
        feature_vectors = torch.stack(
            [item.squeeze(0) if item.ndim == 2 and item.size(0) == 1 else item for item in feature_vectors],
            dim=0,
        )
        feature_vectors = feature_vectors.unsqueeze(2)
        feature_vectors = feature_vectors.repeat(1, 1, 128)
        pooled_feature_vectors = F.adaptive_avg_pool2d(feature_vectors.permute(0, 2, 1).unsqueeze(3), (354, 128))
        reshaped_feature_vectors = pooled_feature_vectors.permute(0, 2, 1, 3)
        feature_vectors = reshaped_feature_vectors[:, :, :, 0]
    return feature_vectors, feature, coor, conf_mask, energy, d, v_p, protein_mask, y

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
