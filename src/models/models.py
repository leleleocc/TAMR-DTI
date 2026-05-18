from os.path import exists
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from torch.nn import SiLU
from src.models.Proteinencoder import CNNTransBlock
from src.models.DWFusion import (
    BiIntention,
    FieldEnhancedBiIntention,
    InteractionFieldFusion,
    MambaEnhancedBiIntention,
    MambaFusion,
    ProteinMambaBiIntention,
    ProteinMambaDirectGateBiIntention,
)
from torch import Tensor
from typing import Tuple
from src.models.Proteinencoder import FeedForwardModule
from src.models.egnn_pytorch import EGNN
import time

def binary_cross_entropy(pred_output, labels, pos_weight=None):
    logits = torch.squeeze(pred_output, 1)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    pred = torch.sigmoid(logits)
    loss = loss_fct(logits, labels)
    return pred, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def masked_mean(inputs, mask=None, dim=1):
    if mask is None:
        return inputs.mean(dim=dim)

    mask = mask.to(dtype=inputs.dtype, device=inputs.device).unsqueeze(-1)
    return (inputs * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1.0)


class TargetAwareConformerEncoder(nn.Module):
    def __init__(self, dim=128, use_target_aware=True):
        super(TargetAwareConformerEncoder, self).__init__()
        self.use_target_aware = use_target_aware
        self.egnn = EGNN(dim=dim)
        self.energy_proj = nn.Linear(1, dim)
        self.score = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, feature, coor, conf_mask, energy, protein_tokens, protein_mask=None):
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)
            coor = coor.unsqueeze(1)
            conf_mask = torch.ones(feature.size(0), 1, device=feature.device, dtype=feature.dtype)
            energy = torch.zeros(feature.size(0), 1, device=feature.device, dtype=feature.dtype)

        batch_size, num_conformers, atom_count, dim = feature.shape
        flat_feature = feature.reshape(batch_size * num_conformers, atom_count, dim)
        flat_coor = coor.reshape(batch_size * num_conformers, atom_count, 3)

        conf_tokens, _ = self.egnn(flat_feature, flat_coor)
        conf_tokens = conf_tokens.reshape(batch_size, num_conformers, atom_count, dim)
        conf_global = conf_tokens.mean(dim=2)

        if self.use_target_aware:
            protein_global = masked_mean(protein_tokens, protein_mask, dim=1)
            protein_global = protein_global.unsqueeze(1).expand(-1, num_conformers, -1)
            energy_emb = self.energy_proj(energy.unsqueeze(-1).to(dtype=conf_global.dtype))
            score_input = torch.cat([conf_global, protein_global, energy_emb], dim=-1)
            logits = self.score(score_input).squeeze(-1)
        else:
            logits = torch.zeros(
                batch_size,
                num_conformers,
                device=feature.device,
                dtype=feature.dtype,
            )

        conf_mask = conf_mask.to(device=feature.device, dtype=torch.bool)
        logits = logits.masked_fill(~conf_mask, -1e9)
        empty_mask = ~conf_mask.any(dim=1)
        if empty_mask.any():
            logits[empty_mask, 0] = 0.0

        conf_weight = torch.softmax(logits, dim=1)
        selected_tokens = (conf_weight[:, :, None, None] * conf_tokens).sum(dim=1)
        selected_global = (conf_weight[:, :, None] * conf_global).sum(dim=1)
        return selected_tokens, selected_global, conf_weight


class LDMDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(LDMDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_embedding = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_max_len = config["PROTEIN"]["MAX_LEN"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']
        use_target_aware_conf = config["DRUG"]["USE_TARGET_AWARE_CONF"]
        fusion_gate_bias = config["DRUG"].get("FUSION_GATE_BIAS", -0.85)
        use_ligand_film = config["PROTEIN"]["USE_LIGAND_FILM"]
        film_scale = config["PROTEIN"]["FILM_SCALE"]
        fusion_mode = config["CROSSINTENTION"].get("FUSION_MODE", None)
        if fusion_mode is None:
            fusion_mode = "interaction_field" if config["CROSSINTENTION"]["USE_INTERACTION_FIELD"] else "biintention"
        self.fusion_mode = str(fusion_mode)
        self.use_interaction_field = self.fusion_mode in ("interaction_field", "field_enhanced_bi")
        field_enhance_gate_bias = config["CROSSINTENTION"].get("FIELD_ENHANCE_GATE_BIAS", -2.0)
        mamba_d_state = config["CROSSINTENTION"].get("MAMBA_D_STATE", 16)
        mamba_d_conv = config["CROSSINTENTION"].get("MAMBA_D_CONV", 4)
        mamba_expand = config["CROSSINTENTION"].get("MAMBA_EXPAND", 2)
        mamba_bidirectional = config["CROSSINTENTION"].get("MAMBA_BIDIRECTIONAL", True)
        mamba_dropout = config["CROSSINTENTION"].get("MAMBA_DROPOUT", 0.1)
        mamba_use_fast_path = config["CROSSINTENTION"].get("MAMBA_USE_FAST_PATH", False)
        mamba_backend = config["CROSSINTENTION"].get("MAMBA_BACKEND", "mamba2")
        mamba_enhance_gate_bias = config["CROSSINTENTION"].get("MAMBA_ENHANCE_GATE_BIAS", -2.0)
        mamba_gate_hidden_dim = config["CROSSINTENTION"].get("MAMBA_GATE_HIDDEN_DIM", cross_emb_dim)
        mamba_gate_delta_scale = config["CROSSINTENTION"].get("MAMBA_GATE_DELTA_SCALE", 1.0)
        mamba_gate_detach_context = config["CROSSINTENTION"].get("MAMBA_GATE_DETACH_CONTEXT", False)
        mamba_gate_bounded = config["CROSSINTENTION"].get("MAMBA_GATE_BOUNDED", False)
        self.cf_top_ratio = config["CROSSINTENTION"]["CF_TOP_RATIO"]
        self.cf_margin = config["CROSSINTENTION"]["CF_MARGIN"]
        self.cf_key_weight = config["CROSSINTENTION"]["CF_KEY_WEIGHT"]
        self.cf_stable_weight = config["CROSSINTENTION"]["CF_STABLE_WEIGHT"]
        self.field_entropy_weight = config["CROSSINTENTION"]["FIELD_ENTROPY_WEIGHT"]


        self.drug_encoding = TargetAwareConformerEncoder(
            dim=cross_emb_dim,
            use_target_aware=use_target_aware_conf,
        )
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.drug_1d_norm = nn.LayerNorm(cross_emb_dim)
        self.drug_geo_norm = nn.LayerNorm(cross_emb_dim)
        self.drug_fusion_gate = nn.Linear(cross_emb_dim * 2, cross_emb_dim)
        nn.init.constant_(self.drug_fusion_gate.bias, fusion_gate_bias)
        self.protein_input_norm = nn.LayerNorm(1024)
        self.protein_proj = nn.Linear(1024, protein_embedding)
        self.protein_extractor = ProteinACmix(max_len=protein_max_len,encoder_dim=protein_embedding,num_layers=3,num_attention_heads=8,feed_forward_expansion_factor=4,feed_forward_dropout_p=0.1,attention_dropout_p=0.1,conv_dropout_p=0.1,conv_kernel_size=3,use_ligand_film=use_ligand_film,film_scale=film_scale)

        if self.fusion_mode == "interaction_field":
            self.cross_intention = InteractionFieldFusion(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)
        elif self.fusion_mode == "field_enhanced_bi":
            self.cross_intention = FieldEnhancedBiIntention(
                embed_dim=cross_emb_dim,
                num_head=cross_num_head,
                layer=cross_layer,
                field_gate_bias=field_enhance_gate_bias,
                device=device,
            )
        elif self.fusion_mode == "biintention":
            self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)
        elif self.fusion_mode == "mamba":
            self.cross_intention = MambaFusion(
                embed_dim=cross_emb_dim,
                num_head=cross_num_head,
                layer=cross_layer,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bidirectional=mamba_bidirectional,
                dropout=mamba_dropout,
                use_fast_path=mamba_use_fast_path,
                backend=mamba_backend,
                device=device,
            )
        elif self.fusion_mode == "mamba_enhanced_bi":
            self.cross_intention = MambaEnhancedBiIntention(
                embed_dim=cross_emb_dim,
                num_head=cross_num_head,
                layer=cross_layer,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bidirectional=mamba_bidirectional,
                dropout=mamba_dropout,
                use_fast_path=mamba_use_fast_path,
                backend=mamba_backend,
                mamba_gate_bias=mamba_enhance_gate_bias,
                device=device,
            )
        elif self.fusion_mode == "protein_mamba_biintention":
            self.cross_intention = ProteinMambaBiIntention(
                embed_dim=cross_emb_dim,
                num_head=cross_num_head,
                layer=cross_layer,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bidirectional=mamba_bidirectional,
                dropout=mamba_dropout,
                use_fast_path=mamba_use_fast_path,
                backend=mamba_backend,
                mamba_gate_bias=mamba_enhance_gate_bias,
                device=device,
            )
        elif self.fusion_mode == "protein_mamba_direct_gate_biintention":
            self.cross_intention = ProteinMambaDirectGateBiIntention(
                embed_dim=cross_emb_dim,
                num_head=cross_num_head,
                layer=cross_layer,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bidirectional=mamba_bidirectional,
                dropout=mamba_dropout,
                use_fast_path=mamba_use_fast_path,
                backend=mamba_backend,
                mamba_gate_bias=mamba_enhance_gate_bias,
                gate_hidden_dim=mamba_gate_hidden_dim,
                gate_delta_scale=mamba_gate_delta_scale,
                gate_detach_context=mamba_gate_detach_context,
                gate_bounded=mamba_gate_bounded,
                device=device,
            )
        else:
            raise ValueError(
                "Unknown CROSSINTENTION.FUSION_MODE "
                f"'{self.fusion_mode}'. Expected biintention, interaction_field, "
                "field_enhanced_bi, mamba, mamba_enhanced_bi, "
                "protein_mamba_biintention, or protein_mamba_direct_gate_biintention."
            )

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, feature_vectors,feature,coor,conf_mask,energy,bg_d,v_p, protein_mask,mode="train",return_aux=False):
        '''
        1、3D特征本来第二维度是N_atom_3d，重采样统一为64，即每个分子重采样到64个原子
        2、feature_vectors由ChemBERTa token hidden states池化得到，shape为[bs, 354, 128]
        '''
        # bg_d (药物 2D 图结构): 由290个原子节点和它们之间的化学键边组成，[bs, 290, 75]
        # feature (药物原子 3D 节点特征): 每个原子的3D坐标，[bs, 64, Dim]
        # coor (药物原子 3D 坐标): 每个原子的3D坐标，[bs, 64, 3]
        # 上面特征第二维度290是药物分子中实际存在的原子数，64是药物分子中虚拟原子的数量
        # 2D特征
        v_d = self.drug_extractor(bg_d)#v_d.shape(bs, 290, 128)
        protein_seed = self.protein_proj(self.protein_input_norm(v_p))
        # 多构象 3D 特征: feature/coor 为 [bs, K, 64, *]，由蛋白初始表示动态加权
        drug_feature_tensor, drug_geo_ctx, conf_weight = self.drug_encoding(
            feature,
            coor,
            conf_mask,
            energy,
            protein_seed,
            protein_mask=protein_mask,
        )
        # 将 2D (v_d) 和 target-aware 3D (drug_feature_tensor) 拼接
        v_d = torch.cat([v_d, drug_feature_tensor], dim=1)  #(bs, 354, 128)
        drug_1d = self.drug_1d_norm(feature_vectors)
        drug_geo = self.drug_geo_norm(v_d)
        fusion_gate = torch.sigmoid(self.drug_fusion_gate(torch.cat([drug_1d, drug_geo], dim=-1)))
        v_d = fusion_gate * drug_1d + (1.0 - fusion_gate) * drug_geo
        drug_ctx = v_d.mean(dim=1)
        v_p = self.protein_extractor(protein_seed, protein_mask, ligand_ctx=drug_ctx)#v_p.shape:(bs, 1200, 128)
        drug_tokens = v_d
        protein_tokens = v_p
        if self.fusion_mode == "biintention":
            f, v_d, v_p, att = self.cross_intention(drug=drug_tokens, protein=protein_tokens)#f:[bs, 256]
        else:
            f, v_d, v_p, att = self.cross_intention(drug=drug_tokens, protein=protein_tokens, protein_mask=protein_mask)#f:[bs, 256]
        score = self.mlp_classifier(f)
        aux = {
            "field": att,
            "drug_tokens": drug_tokens,
            "protein_tokens": protein_tokens,
            "protein_mask": protein_mask,
            "conformer_weight": conf_weight,
            "drug_geo_ctx": drug_geo_ctx,
        }
        if return_aux:
            return score, aux
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

    def score_from_tokens(self, drug_tokens, protein_tokens, protein_mask=None):
        if self.fusion_mode == "biintention":
            f, _, _, field = self.cross_intention(drug=drug_tokens, protein=protein_tokens)
        else:
            f, _, _, field = self.cross_intention(
                drug=drug_tokens,
                protein=protein_tokens,
                protein_mask=protein_mask,
            )
        score = self.mlp_classifier(f)
        return score, field

    def counterfactual_loss(self, score, aux, labels, return_parts=False):
        field = aux.get("field")
        if field is None:
            loss = score.new_tensor(0.0)
            if return_parts:
                return loss, {
                    "cf_key_raw": loss.detach(),
                    "cf_key_weighted": loss.detach(),
                    "cf_stable_raw": loss.detach(),
                    "cf_stable_weighted": loss.detach(),
                    "cf_entropy_raw": loss.detach(),
                    "cf_entropy_weighted": loss.detach(),
                }
            return loss

        drug_tokens = aux["drug_tokens"]
        protein_tokens = aux["protein_tokens"]
        protein_mask = aux.get("protein_mask")
        field_for_rank = field.detach()
        protein_importance = field_for_rank.max(dim=1).values

        if protein_mask is not None:
            protein_mask_bool = protein_mask.to(dtype=torch.bool, device=protein_importance.device)
            protein_importance_top = protein_importance.masked_fill(~protein_mask_bool, -1e9)
            protein_importance_low = protein_importance.masked_fill(~protein_mask_bool, 1e9)
            valid_count = protein_mask_bool.sum(dim=1).clamp_min(1)
            top_k = max(1, int(protein_importance.size(1) * self.cf_top_ratio))
            top_k = min(top_k, int(valid_count.max().item()))
        else:
            protein_importance_top = protein_importance
            protein_importance_low = protein_importance
            top_k = max(1, int(protein_importance.size(1) * self.cf_top_ratio))

        top_idx = protein_importance_top.topk(top_k, dim=1).indices
        low_idx = protein_importance_low.topk(top_k, dim=1, largest=False).indices

        key_mask = torch.ones_like(protein_tokens[..., :1])
        nonkey_mask = torch.ones_like(protein_tokens[..., :1])
        key_mask.scatter_(1, top_idx.unsqueeze(-1), 0.0)
        nonkey_mask.scatter_(1, low_idx.unsqueeze(-1), 0.0)

        key_score, _ = self.score_from_tokens(drug_tokens, protein_tokens * key_mask, protein_mask)
        nonkey_score, _ = self.score_from_tokens(drug_tokens, protein_tokens * nonkey_mask, protein_mask)

        full_logit = score.squeeze(-1).detach()
        key_logit = key_score.squeeze(-1)
        nonkey_logit = nonkey_score.squeeze(-1)
        labels = labels.to(dtype=score.dtype, device=score.device)

        key_loss = labels * F.relu(key_logit - full_logit + self.cf_margin)
        stable_loss = F.mse_loss(nonkey_logit, full_logit, reduction="none")

        key_raw = key_loss.mean()
        stable_raw = stable_loss.mean()
        key_weighted = self.cf_key_weight * key_raw
        stable_weighted = self.cf_stable_weight * stable_raw
        entropy_raw = score.new_tensor(0.0)
        entropy_weighted = score.new_tensor(0.0)

        loss = key_weighted + stable_weighted

        if self.field_entropy_weight > 0:
            field_prob = F.softmax(field, dim=-1)
            entropy_raw = -(field_prob * torch.log(field_prob.clamp_min(1e-8))).sum(dim=-1).mean()
            entropy_weighted = self.field_entropy_weight * entropy_raw
            loss = loss + entropy_weighted

        if return_parts:
            return loss, {
                "cf_key_raw": key_raw.detach(),
                "cf_key_weighted": key_weighted.detach(),
                "cf_stable_raw": stable_raw.detach(),
                "cf_stable_weighted": stable_weighted.detach(),
                "cf_entropy_raw": entropy_raw.detach(),
                "cf_entropy_weighted": entropy_weighted.detach(),
            }
        return loss


class MolecularGCN(nn.Module):
    '''
    输入: batch_graph（多个分子图拼成一个大图）

    [总节点数 × 74]         ← 原始原子特征
        ↓ init_transform
    [总节点数 × 128]        ← 对齐到 embedding 空间
        ↓ GCN
    [总节点数 × output_dim] ← 聚合了邻居信息的节点表示
        ↓ view reshape
    [batch × 节点数 × output_dim]  ← 按分子拆开

    输出: 每个分子的节点特征矩阵
    '''
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        # hidden_feats = [128, 64, 32] 表示3层 GCN，每层的输出维度依次是 128、64、32，对于小分子而言2-3层就足够了
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[:, -1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]


    '''
    假设 batch=4个分子，每个分子有290个原子，hidden_feats = [128, 64, 32]
    batch_graph:         4个分子图拼成的大图
                     总节点数 = 4×290 = 1160

    pop('h'):            [1160 × 74]    原始原子特征
    init_transform:      [1160 × 128]   线性对齐
    gnn:                 [1160 × 32]    图卷积后
    view reshape:        [4 × 290 × 32] 按分子拆开

    输出: [4 × 290 × 32]
    '''
    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinACmix(nn.Module):
    def __init__(
            self,
            max_len: int = 1200,
            encoder_dim: int = 512,
            num_layers: int = 3,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            top_k: int = 5,
            k1: int = 3,
            use_ligand_film: bool = False,
            film_scale: float = 0.1,
    ):
        super(ProteinACmix, self).__init__()
        self.use_ligand_film = use_ligand_film
        self.film_scale = film_scale
        self.CNNTranslayers = nn.ModuleList([CNNTransBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            top_k=top_k,
            k1=k1,
            max_len=max_len
        ) for _ in range(num_layers)])

        self.FFlayers = nn.ModuleList([FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        ) for _ in range(num_layers)])

        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim * 2),
                nn.SiLU(),
                nn.Linear(encoder_dim * 2, encoder_dim * 2),
            )
            for _ in range(num_layers)
        ])

    def forward(self, inputs: Tensor, mask: Tensor, ligand_ctx: Tensor = None) -> Tuple[Tensor,Tensor]:
        #mask = (inputs.sum(dim=-1) != 0).float()
        CNNTransOutputs = inputs.to(dtype=torch.float32)
        for num in range(len(self.CNNTranslayers)):
            FF_output = 0.5 * self.FFlayers[num](CNNTransOutputs) + 0.5 * CNNTransOutputs
            if self.use_ligand_film and ligand_ctx is not None:
                gamma, beta = self.film_layers[num](ligand_ctx).chunk(2, dim=-1)
                gamma = self.film_scale * torch.tanh(gamma)
                beta = self.film_scale * beta
                FF_output = FF_output * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            CNNTransOutputs = self.CNNTranslayers[num](FF_output,mask)
        return CNNTransOutputs

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.LayerNorm(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


BINDTI = LDMDTI
