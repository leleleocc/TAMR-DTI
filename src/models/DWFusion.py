import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

try:
    import mamba_ssm.modules.mamba_simple as mamba_simple
    import mamba_ssm.modules.mamba2_simple as mamba2_simple
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    Mamba = mamba_simple.Mamba
    Mamba2Simple = mamba2_simple.Mamba2Simple
except ImportError:
    Mamba = None
    Mamba2Simple = None
    mamba_chunk_scan_combined = None


class Mamba2Compat(Mamba2Simple if Mamba2Simple is not None else nn.Module):
    def forward(self, u, seq_idx=None):
        if self.use_mem_eff_path:
            return super(Mamba2Compat, self).forward(u, seq_idx=seq_idx)
        if mamba_chunk_scan_combined is None:
            raise ImportError("mamba_ssm Triton scan is required for Mamba2Compat.")

        batch, seqlen, dim = u.shape
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)
        xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :seqlen].transpose(1, 2))

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        return self.out_proj(y)


class Intention(nn.Module):

    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.alpha = nn.Parameter(torch.rand(1))
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)

        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)
        # 新增可学习的权重参数，用于动态调整各头的权重
        self.head_weights = nn.Parameter(torch.rand(num_heads))
    def forward(self, x, query=None):
        if query is None:
            query = x
        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.clone().permute(0, 1, 3, 2)
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        kk = key_t @ key
        eye = torch.eye(kk.shape[-1], device=kk.device, dtype=kk.dtype)
        kk = self.alpha * eye.view(1, 1, kk.shape[-1], kk.shape[-1]) + kk
        kk_inv = torch.linalg.pinv(kk)
        attn_map = (kk_inv @ key_t) @ value
        attn_map = self.softmax(attn_map)
        # 应用动态权重调整各头的注意力权重
        attn_map = attn_map * self.head_weights.view(1, -1, 1, 1)
        out = (query @ attn_map)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out=self.out(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(p=dropout)
        )
        self.wk = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(p=dropout)
        )
        self.wv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(p=dropout)
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out


class IntentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        att = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        return out


class BiIntention(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiIntention, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        # self attention
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug)
        protein = self.attn_protein(protein)

        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug, protein = v_d, v_p

        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')

        f = torch.cat((v_d, v_p), dim=1)

        return f, v_d, v_p, None


class InteractionFieldFusion(nn.Module):
    def __init__(self, embed_dim, num_head=8, layer=1, device='cuda'):
        super(InteractionFieldFusion, self).__init__()
        self.drug_self = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.protein_self = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.drug_proj = nn.Linear(embed_dim, embed_dim)
        self.protein_proj = nn.Linear(embed_dim, embed_dim)
        self.drug_update = nn.Linear(embed_dim, embed_dim)
        self.protein_update = nn.Linear(embed_dim, embed_dim)
        self.drug_norm = nn.LayerNorm(embed_dim)
        self.protein_norm = nn.LayerNorm(embed_dim)

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        drug = self.drug_self(drug)
        protein = self.protein_self(protein)

        q_d = self.drug_proj(self.drug_norm(drug))
        k_p = self.protein_proj(self.protein_norm(protein))
        field = torch.matmul(q_d, k_p.transpose(-1, -2)) / (drug.size(-1) ** 0.5)

        if protein_mask is not None:
            field = field.masked_fill(protein_mask[:, None, :] == 0, -1e9)

        d2p = F.softmax(field, dim=-1)
        p2d = F.softmax(field.transpose(-1, -2), dim=-1)

        drug_ctx = torch.matmul(d2p, protein)
        protein_ctx = torch.matmul(p2d, drug)

        drug_fused = drug + self.drug_update(drug_ctx)
        protein_fused = protein + self.protein_update(protein_ctx)

        drug_pool = reduce(drug_fused, 'B H W -> B W', 'max')
        if protein_mask is not None:
            min_value = torch.finfo(protein_fused.dtype).min
            protein_for_pool = protein_fused.masked_fill(protein_mask[:, :, None] == 0, min_value)
            protein_pool = reduce(protein_for_pool, 'B H W -> B W', 'max')
        else:
            protein_pool = reduce(protein_fused, 'B H W -> B W', 'max')

        f = torch.cat((drug_pool, protein_pool), dim=1)
        return f, drug_fused, protein_fused, field


class FieldEnhancedBiIntention(BiIntention):
    def __init__(self, embed_dim, layer=1, num_head=8, field_gate_bias=-2.0, device='cuda'):
        super(FieldEnhancedBiIntention, self).__init__(
            embed_dim=embed_dim,
            layer=layer,
            num_head=num_head,
            device=device,
        )
        self.field_fusion = InteractionFieldFusion(
            embed_dim=embed_dim,
            num_head=num_head,
            layer=layer,
            device=device,
        )
        self.field_delta = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.field_gate_logit = nn.Parameter(torch.tensor(float(field_gate_bias)))

        # Start from the stable BiIntention decision surface, then let the field branch learn a residual.
        nn.init.zeros_(self.field_delta[-1].weight)
        nn.init.zeros_(self.field_delta[-1].bias)

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        f_bi, v_d, v_p, _ = super(FieldEnhancedBiIntention, self).forward(drug, protein)
        f_field, _, _, field = self.field_fusion(
            drug=drug,
            protein=protein,
            protein_mask=protein_mask,
            drug_mask=drug_mask,
        )
        field_gate = torch.sigmoid(self.field_gate_logit).to(dtype=f_bi.dtype, device=f_bi.device)
        f = f_bi + field_gate * self.field_delta(f_field)
        return f, v_d, v_p, field


class MambaEnhancedBiIntention(BiIntention):
    def __init__(
        self,
        embed_dim,
        layer=1,
        num_head=8,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        dropout=0.1,
        use_fast_path=False,
        backend="mamba2",
        mamba_gate_bias=-2.0,
        device='cuda',
    ):
        super(MambaEnhancedBiIntention, self).__init__(
            embed_dim=embed_dim,
            layer=layer,
            num_head=num_head,
            device=device,
        )
        self.mamba_fusion = MambaFusion(
            embed_dim=embed_dim,
            num_head=num_head,
            layer=layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
            dropout=dropout,
            use_fast_path=use_fast_path,
            backend=backend,
            device=device,
        )
        self.mamba_delta = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.mamba_gate_logit = nn.Parameter(torch.tensor(float(mamba_gate_bias)))

        # Start from the stable BiIntention decision surface, then let Mamba learn a residual.
        nn.init.zeros_(self.mamba_delta[-1].weight)
        nn.init.zeros_(self.mamba_delta[-1].bias)

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        f_bi, v_d, v_p, _ = super(MambaEnhancedBiIntention, self).forward(drug, protein)
        f_mamba, _, _, _ = self.mamba_fusion(
            drug=drug,
            protein=protein,
            protein_mask=protein_mask,
            drug_mask=drug_mask,
        )
        mamba_gate = torch.sigmoid(self.mamba_gate_logit).to(dtype=f_bi.dtype, device=f_bi.device)
        f = f_bi + mamba_gate * self.mamba_delta(f_mamba)
        return f, v_d, v_p, None


def masked_max_pool(x, mask=None):
    if mask is None:
        return reduce(x, 'B H W -> B W', 'max')

    mask = mask.to(device=x.device, dtype=torch.bool)
    min_value = torch.finfo(x.dtype).min
    x = x.masked_fill(mask[:, :, None] == 0, min_value)
    pooled = reduce(x, 'B H W -> B W', 'max')
    empty = ~mask.any(dim=1)
    if empty.any():
        pooled = pooled.masked_fill(empty[:, None], 0.0)
    return pooled


class MambaResidualBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        dropout=0.1,
        use_fast_path=False,
        backend="mamba2",
        device='cuda',
    ):
        super(MambaResidualBlock, self).__init__()
        self.backend = backend
        if backend == "mamba2":
            if Mamba2Simple is None:
                raise ImportError("mamba_ssm Mamba2Simple is required when MAMBA_BACKEND='mamba2'.")
            mixer_cls = Mamba2Compat
            mixer_kwargs = {
                "d_model": embed_dim,
                "d_state": max(d_state, 16),
                "d_conv": d_conv,
                "expand": expand,
                "headdim": embed_dim,
                "use_mem_eff_path": False,
                "device": device,
            }
        elif backend == "mamba1":
            if Mamba is None:
                raise ImportError("mamba_ssm is required when CROSSINTENTION.FUSION_MODE='mamba'.")
            mixer_cls = Mamba
            mixer_kwargs = {
                "d_model": embed_dim,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "use_fast_path": use_fast_path,
                "device": device,
            }
        else:
            raise ValueError(f"Unknown Mamba backend '{backend}'. Expected 'mamba2' or 'mamba1'.")

        if mixer_cls is None:
            raise ImportError("mamba_ssm is required when CROSSINTENTION.FUSION_MODE='mamba'.")

        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(embed_dim)
        self.mamba_fwd = mixer_cls(**mixer_kwargs)
        if bidirectional:
            self.mamba_bwd = mixer_cls(**mixer_kwargs)
            self.mix_proj = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.mamba_bwd = None
            self.mix_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * mask

        h = self.norm(x)
        fwd = self.mamba_fwd(h)
        if self.bidirectional:
            bwd = torch.flip(self.mamba_bwd(torch.flip(h, dims=[1])), dims=[1])
            h = self.mix_proj(torch.cat([fwd, bwd], dim=-1))
        else:
            h = self.mix_proj(fwd)

        x = x + self.dropout(h)
        x = x + self.ffn(x)

        if mask is not None:
            x = x * mask
        return x


class MambaFusion(nn.Module):
    def __init__(
        self,
        embed_dim,
        layer=1,
        num_head=8,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        dropout=0.1,
        use_fast_path=False,
        backend="mamba2",
        device='cuda',
    ):
        super(MambaFusion, self).__init__()
        self.embed_dim = embed_dim
        self.type_embedding = nn.Parameter(torch.zeros(2, embed_dim))
        nn.init.normal_(self.type_embedding, std=0.02)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(
                embed_dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bidirectional=bidirectional,
                dropout=dropout,
                use_fast_path=use_fast_path,
                backend=backend,
                device=device,
            )
            for _ in range(layer)
        ])
        self.drug_norm = nn.LayerNorm(embed_dim)
        self.protein_norm = nn.LayerNorm(embed_dim)

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        if drug_mask is None:
            drug_mask = torch.ones(
                drug.size(0),
                drug.size(1),
                device=drug.device,
                dtype=torch.bool,
            )
        else:
            drug_mask = drug_mask.to(device=drug.device, dtype=torch.bool)

        if protein_mask is None:
            protein_mask = torch.ones(
                protein.size(0),
                protein.size(1),
                device=protein.device,
                dtype=torch.bool,
            )
        else:
            protein_mask = protein_mask.to(device=protein.device, dtype=torch.bool)

        drug = drug + self.type_embedding[0].view(1, 1, -1).to(dtype=drug.dtype, device=drug.device)
        protein = protein + self.type_embedding[1].view(1, 1, -1).to(dtype=protein.dtype, device=protein.device)
        x = torch.cat([drug, protein], dim=1)
        seq_mask = torch.cat([drug_mask, protein_mask], dim=1)

        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x, seq_mask)

        drug_fused, protein_fused = x[:, :drug.size(1)], x[:, drug.size(1):]
        drug_fused = self.drug_norm(drug_fused)
        protein_fused = self.protein_norm(protein_fused)

        drug_pool = masked_max_pool(drug_fused, drug_mask)
        protein_pool = masked_max_pool(protein_fused, protein_mask)
        f = torch.cat((drug_pool, protein_pool), dim=1)
        return f, drug_fused, protein_fused, None


class ProteinMambaBiIntention(BiIntention):
    def __init__(
        self,
        embed_dim,
        layer=1,
        num_head=8,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        dropout=0.1,
        use_fast_path=False,
        backend="mamba2",
        mamba_gate_bias=-4.0,
        device='cuda',
    ):
        super(ProteinMambaBiIntention, self).__init__(
            embed_dim=embed_dim,
            layer=layer,
            num_head=num_head,
            device=device,
        )
        self.protein_mamba = MambaResidualBlock(
            embed_dim=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
            dropout=dropout,
            use_fast_path=use_fast_path,
            backend=backend,
            device=device,
        )
        self.protein_mamba_gate_logit = nn.Parameter(torch.tensor(float(mamba_gate_bias)))

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        protein_mamba = self.protein_mamba(protein, protein_mask)
        gate = torch.sigmoid(self.protein_mamba_gate_logit).to(
            dtype=protein.dtype,
            device=protein.device,
        )
        protein = protein + gate * (protein_mamba - protein)
        return super(ProteinMambaBiIntention, self).forward(drug, protein)


class ProteinMambaDirectGateBiIntention(BiIntention):
    def __init__(
        self,
        embed_dim,
        layer=1,
        num_head=8,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        dropout=0.1,
        use_fast_path=False,
        backend="mamba2",
        mamba_gate_bias=-2.0,
        gate_hidden_dim=None,
        gate_delta_scale=1.0,
        gate_detach_context=False,
        gate_bounded=False,
        device='cuda',
    ):
        super(ProteinMambaDirectGateBiIntention, self).__init__(
            embed_dim=embed_dim,
            layer=layer,
            num_head=num_head,
            device=device,
        )
        self.protein_mamba = MambaResidualBlock(
            embed_dim=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
            dropout=dropout,
            use_fast_path=use_fast_path,
            backend=backend,
            device=device,
        )
        gate_hidden_dim = gate_hidden_dim or embed_dim
        self.gate_delta_scale = float(gate_delta_scale)
        self.gate_detach_context = bool(gate_detach_context)
        self.gate_bounded = bool(gate_bounded)
        self.protein_mamba_gate_logit = nn.Parameter(torch.tensor(float(mamba_gate_bias)))
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, gate_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, drug, protein, protein_mask=None, drug_mask=None):
        protein_mamba = self.protein_mamba(protein, protein_mask)
        drug_ctx = masked_max_pool(drug, drug_mask)
        protein_ctx = masked_max_pool(protein, protein_mask)
        gate_input = torch.cat([drug_ctx, protein_ctx, drug_ctx * protein_ctx], dim=-1)
        if self.gate_detach_context:
            gate_input = gate_input.detach()
        gate_delta = self.gate_mlp(gate_input)
        if self.gate_bounded:
            gate_delta = self.gate_delta_scale * torch.tanh(gate_delta)
        else:
            gate_delta = self.gate_delta_scale * gate_delta
        gate_logit = self.protein_mamba_gate_logit + gate_delta
        gate = torch.sigmoid(gate_logit).to(dtype=protein.dtype, device=protein.device).view(-1, 1, 1)
        protein = protein + gate * (protein_mamba - protein)
        return super(ProteinMambaDirectGateBiIntention, self).forward(drug, protein)
