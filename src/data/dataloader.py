from pathlib import Path
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

FEATURE_CACHE_BY_DIR = {}
DRUG_1D_SHAPE = (354, 128)
PROTEIN_SHAPE = (128, 1024)
PROTEIN_MASK_SHAPE = (128,)
DRUG3D_NUM_CONFORMERS = 8
DRUG3D_ATOM_COUNT = 64
DRUG3D_FEATURE_DIM = 128
DRUG3D_FEATURE_SHAPE = (DRUG3D_NUM_CONFORMERS, DRUG3D_ATOM_COUNT, DRUG3D_FEATURE_DIM)
DRUG3D_COOR_SHAPE = (DRUG3D_NUM_CONFORMERS, DRUG3D_ATOM_COUNT, 3)
DRUG3D_SINGLE_FEATURE_SHAPE = (DRUG3D_ATOM_COUNT, DRUG3D_FEATURE_DIM)
DRUG3D_SINGLE_COOR_SHAPE = (DRUG3D_ATOM_COUNT, 3)


def _entry_feature(entry):
    if isinstance(entry, dict):
        return entry.get("feature")
    return entry


def _shape_of(entry):
    if torch.is_tensor(entry):
        return tuple(entry.shape)
    return None


def _cache_error(cache_name, key, actual, expected):
    return (
        f"Invalid {cache_name} cache shape for key {str(key)[:80]!r}: "
        f"got {actual}, expected {expected}. "
        "Rebuild feature cache with: python scripts/pre_extract.py --data <dataset> --split <split> --overwrite"
    )


def _normalize_drug3d_entry(entry):
    if not isinstance(entry, dict):
        return None

    feature = entry.get("feature")
    coor = entry.get("coor")
    if not torch.is_tensor(feature) or not torch.is_tensor(coor):
        return None

    norm_feature = torch.zeros(DRUG3D_FEATURE_SHAPE, dtype=torch.float32)
    norm_coor = torch.zeros(DRUG3D_COOR_SHAPE, dtype=torch.float32)
    conf_mask = torch.zeros(DRUG3D_NUM_CONFORMERS, dtype=torch.float32)
    energy = torch.zeros(DRUG3D_NUM_CONFORMERS, dtype=torch.float32)

    if tuple(feature.shape) == DRUG3D_SINGLE_FEATURE_SHAPE and tuple(coor.shape) == DRUG3D_SINGLE_COOR_SHAPE:
        norm_feature[0] = feature.float()
        norm_coor[0] = coor.float()
        conf_mask[0] = 1.0
        return norm_feature, norm_coor, conf_mask, energy

    if feature.ndim != 3 or coor.ndim != 3:
        return None
    if tuple(feature.shape[1:]) != DRUG3D_SINGLE_FEATURE_SHAPE:
        return None
    if tuple(coor.shape[1:]) != DRUG3D_SINGLE_COOR_SHAPE:
        return None

    valid_count = min(int(feature.shape[0]), DRUG3D_NUM_CONFORMERS)
    if valid_count <= 0:
        conf_mask[0] = 1.0
        return norm_feature, norm_coor, conf_mask, energy

    norm_feature[:valid_count] = feature[:valid_count].float()
    norm_coor[:valid_count] = coor[:valid_count].float()

    raw_mask = entry.get("conf_mask")
    if torch.is_tensor(raw_mask):
        conf_mask[:valid_count] = raw_mask[:valid_count].float()
    else:
        conf_mask[:valid_count] = 1.0
    if conf_mask.sum().item() == 0:
        conf_mask[0] = 1.0

    raw_energy = entry.get("energy")
    if torch.is_tensor(raw_energy):
        energy[:valid_count] = raw_energy[:valid_count].float()

    return norm_feature, norm_coor, conf_mask, energy


def load_feature_cache(cache_dir):
    cache_dir = Path(cache_dir)
    cache_key = str(cache_dir.resolve())
    if cache_key in FEATURE_CACHE_BY_DIR:
        return FEATURE_CACHE_BY_DIR[cache_key]

    smiles_path = cache_dir / "smiles_features.pt"
    protein_path = cache_dir / "protein_features.pt"
    drug3d_path = cache_dir / "drug3d_features.pt"

    if not smiles_path.exists():
        raise FileNotFoundError(
            f"Missing SMILES feature cache: {smiles_path}. "
            "Run scripts/pre_extract.py before training."
        )
    if not protein_path.exists():
        raise FileNotFoundError(
            f"Missing protein feature cache: {protein_path}. "
            "Run scripts/pre_extract.py before training."
        )
    if not drug3d_path.exists():
        raise FileNotFoundError(
            f"Missing drug 3D feature cache: {drug3d_path}. "
            "Run scripts/pre_extract.py before training."
        )

    smiles_cache = torch.load(smiles_path, map_location="cpu", weights_only=True)
    protein_cache = torch.load(protein_path, map_location="cpu", weights_only=True)
    drug3d_cache = torch.load(drug3d_path, map_location="cpu", weights_only=True)
    FEATURE_CACHE_BY_DIR[cache_key] = (smiles_cache, protein_cache, drug3d_cache)
    return smiles_cache, protein_cache, drug3d_cache


class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df, cache_dir, max_drug_nodes=290, graph_cache_size=2048):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.graph_cache_size = graph_cache_size
        self.graph_cache = OrderedDict()
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.smiles_cache, self.protein_cache, self.drug3d_cache = load_feature_cache(cache_dir)
        self.all_smiles = self.df["SMILES"].astype(str).tolist()
        self.all_proteins = self.df["Protein"].astype(str).tolist()

        missing_smiles = sorted(set(smile for smile in self.all_smiles if smile not in self.smiles_cache))
        missing_proteins = sorted(set(protein for protein in self.all_proteins if protein not in self.protein_cache))
        missing_drug3d = sorted(set(smile for smile in self.all_smiles if smile not in self.drug3d_cache))
        if missing_smiles:
            raise KeyError(
                f"Missing {len(missing_smiles)} SMILES features in cache. "
                f"Example: {missing_smiles[0]}"
            )
        if missing_proteins:
            raise KeyError(
                f"Missing {len(missing_proteins)} protein features in cache. "
                f"Example: {missing_proteins[0][:30]}"
            )
        if missing_drug3d:
            raise KeyError(
                f"Missing {len(missing_drug3d)} drug 3D features in cache. "
                f"Example: {missing_drug3d[0]}"
            )
        self.validate_feature_schema()

    def validate_feature_schema(self):
        if not self.all_smiles or not self.all_proteins:
            return

        smile = self.all_smiles[0]
        smiles_feature = _entry_feature(self.smiles_cache[smile])
        if not torch.is_tensor(smiles_feature) or tuple(smiles_feature.shape) != DRUG_1D_SHAPE:
            raise ValueError(_cache_error("SMILES", smile, _shape_of(smiles_feature), DRUG_1D_SHAPE))

        protein = self.all_proteins[0]
        protein_entry = self.protein_cache[protein]
        protein_feature = _entry_feature(protein_entry)
        protein_mask = protein_entry.get("mask") if isinstance(protein_entry, dict) else None
        if not torch.is_tensor(protein_feature) or tuple(protein_feature.shape) != PROTEIN_SHAPE:
            raise ValueError(_cache_error("protein", protein, _shape_of(protein_feature), PROTEIN_SHAPE))
        if not torch.is_tensor(protein_mask) or tuple(protein_mask.shape) != PROTEIN_MASK_SHAPE:
            raise ValueError(_cache_error("protein mask", protein, _shape_of(protein_mask), PROTEIN_MASK_SHAPE))

        drug3d_entry = self.drug3d_cache[smile]
        drug3d = _normalize_drug3d_entry(drug3d_entry)
        if drug3d is None:
            drug3d_feature = drug3d_entry.get("feature") if isinstance(drug3d_entry, dict) else None
            drug3d_coor = drug3d_entry.get("coor") if isinstance(drug3d_entry, dict) else None
            raise ValueError(
                _cache_error(
                    "drug 3D feature/coordinate",
                    smile,
                    (_shape_of(drug3d_feature), _shape_of(drug3d_coor)),
                    (DRUG3D_FEATURE_SHAPE, DRUG3D_COOR_SHAPE),
                )
            )

    def get_base_graph(self, smile):
        if self.graph_cache_size <= 0:
            return self.fc(
                smiles=smile,
                node_featurizer=self.atom_featurizer,
                edge_featurizer=self.bond_featurizer,
            )

        graph = self.graph_cache.get(smile)
        if graph is not None:
            self.graph_cache.move_to_end(smile)
            return graph.clone()

        graph = self.fc(
            smiles=smile,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )
        self.graph_cache[smile] = graph
        if len(self.graph_cache) > self.graph_cache_size:
            self.graph_cache.popitem(last=False)
        return graph.clone()

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smile = str(self.df.iloc[index]["SMILES"])
        protein = str(self.df.iloc[index]["Protein"])
        feature_vectors = self.smiles_cache[smile]
        feature_vectors = _entry_feature(feature_vectors)
        if not torch.is_tensor(feature_vectors) or tuple(feature_vectors.shape) != DRUG_1D_SHAPE:
            raise ValueError(_cache_error("SMILES", smile, _shape_of(feature_vectors), DRUG_1D_SHAPE))
        feature_vectors = feature_vectors.float()
        v_d = smile
        drug3d = self.drug3d_cache[smile]
        normalized_drug3d = _normalize_drug3d_entry(drug3d)
        if normalized_drug3d is None:
            feature = drug3d.get("feature") if isinstance(drug3d, dict) else None
            coor = drug3d.get("coor") if isinstance(drug3d, dict) else None
            raise ValueError(
                _cache_error(
                    "drug 3D feature/coordinate",
                    smile,
                    (_shape_of(feature), _shape_of(coor)),
                    (DRUG3D_FEATURE_SHAPE, DRUG3D_COOR_SHAPE),
                )
            )
        feature, coor, conf_mask, energy = normalized_drug3d
        v_d = self.get_base_graph(v_d)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        if num_virtual_nodes < 0:
            raise ValueError(
                f"SMILES has {num_actual_nodes} graph nodes, exceeding max_drug_nodes={self.max_drug_nodes}: {smile}"
            )
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        y = self.df.iloc[index]["Y"]
        protein_feature = self.protein_cache[protein]
        if not isinstance(protein_feature, dict):
            raise ValueError(_cache_error("protein", protein, _shape_of(protein_feature), PROTEIN_SHAPE))

        v_p = protein_feature["feature"]
        protein_mask = protein_feature.get("mask")
        if not torch.is_tensor(v_p) or tuple(v_p.shape) != PROTEIN_SHAPE:
            raise ValueError(_cache_error("protein", protein, _shape_of(v_p), PROTEIN_SHAPE))
        if not torch.is_tensor(protein_mask) or tuple(protein_mask.shape) != PROTEIN_MASK_SHAPE:
            raise ValueError(_cache_error("protein mask", protein, _shape_of(protein_mask), PROTEIN_MASK_SHAPE))
        v_p = v_p.float()
        protein_mask = protein_mask.float().numpy()
        '''
        feature_vectors：[354, 128]
        ChemBERTa hidden states 去掉特殊 token 后自适应池化得到的 token 矩阵

        v_d：DGLGraph（不是普通 Tensor）
        图里节点特征 v_d.ndata['h'] 的形状是 [290, 75]
        其中 290 = max_drug_nodes，75 = 74 + 1(virtual node bit)，74维表示原子特征

        feature：[8, 64, 128]（Tensor）
        来自 drug3d_features.pt 的离线多构象 3D 原子特征，已固定长度池化

        coor：[8, 64, 3]（Tensor）
        来自 drug3d_features.pt 的离线多构象 3D 坐标，已固定长度池化

        conf_mask：[8]（Tensor）
        有效构象 mask

        energy：[8]（Tensor）
        归一化后的构象能量

        v_p：[128, 1024]
        ProtBERT hidden states 去掉特殊 token 后自适应池化得到的残基 token 矩阵

        protein_mask：[128]（numpy）
        池化后的蛋白 token 有效位

        y：标量（0/1）
        '''
        return feature_vectors, feature, coor, conf_mask, energy, v_d, v_p, protein_mask, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
