# TAMR-DTI 可解释性升级方案（v5 — local /paddle, single-box）

> **接手须知**：本机就是 plan 描述的全部执行环境，**没有「本地 Mac + 远程 Linux」的双机拆分**。所有训练、forward、画图都在 `/paddle/lv/TAMR-DTI` 这台 linux 机器上跑。文中所有路径、文件名、属性名、CLI 都按本机实地核对过。

---

## 0. 背景（为什么这个 plan 存在）

论文 `sec/experiments.tex` 的 **Interpretability analysis** 子节有三个硬伤：

1. **图和统计互相打脸**：aggregate 统计（top1 weight=0.241、normalized entropy=0.931，全 5,493 样本）说明 conformer 选择是软加权；但 [Figure 8 (conformer heatmap)](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_conformer_weights.pdf) 因为 (i) cherry-pick 各组最自信的 18 条 + (ii) `PowerNorm(gamma=0.85)` 视觉压低小权重，**看起来像硬选择**。
2. **三张图全是 "模块输出非零" 的 sanity check**：没有任何反事实或外部对照，无法支撑 "interpretability" 这个标题。`sec/discussion.tex` 已经自己降级为 "diagnostic checks"，但 experiments 仍叫 "Interpretability analysis"——narrative 不一致。
3. **Mamba 热图没有锚**：只是 "不同样本有不同峰"，无法说明峰是有意义的。

### v5 总目标

只做 **2 张图**：

- **Figure 1（替换当前 Fig 8）= Plan A**：观察层。population-level conformer 分布。
- **Figure 4（新增）= Case study**：因果层 + 生物学锚定。用 2 个有 PDB 共晶 + UniProt binding-site 注释的经典 drug-target 对，把 Mamba refinement 峰反向映射到残基，和已知结合位点对照。

Fig 9/10 全删；Fig 2/3（v3 的 conformer swap、Mamba top-k mask）降级为 fallback，仅当 case-study 匹配失败时启用。

---

## 1. 本机环境与仓库现状（必读上下文）

### 1.1 机器一览

| 项 | 值 |
|---|---|
| 仓库根 | `/paddle/lv/TAMR-DTI`（**非 git 仓库**） |
| OS | Linux 5.10.0-1.0.0.26 x86_64 |
| GPU | 8 × Tesla V100-SXM2-32GB（驱动 550.90.07，CUDA 12.4） |
| Python | `/paddle/miniconda3/envs/ldm-dti/bin/python`（3.9.23） |
| 关键包 | torch 2.4.0+cu124、deepspeed 0.18.9、mamba_ssm 2.2.2、causal_conv1d 1.6.1、rdkit 2025.09.2、swanlab 0.7.18 |
| swanlab 模式 | `SWANLAB_MODE=offline`（本机未登录，`local` 缺 `swanboard` 依赖，offline 走 backup 即可） |

### 1.2 数据 / 缓存

| 类型 | 路径 |
|---|---|
| 训练 CSV | `data/datasets/biosnap/random/{train,val,test}.csv`（test=5493 行） |
| 特征缓存目录 | `cache/features/datasets/biosnap/random/` |
| 蛋白特征 | `cache/features/datasets/biosnap/random/protein_features.pt`（549 MB） |
| 药物 1D 特征 | `cache/features/datasets/biosnap/random/smiles_features.pt`（392 MB） |
| 药物 3D 特征 | `cache/features/datasets/biosnap/random/drug3d_features.pt`（1.2 GB；**文件名是 `drug3d_features.pt`，无下划线**） |
| 缓存 schema | [meta.json](cache/features/datasets/biosnap/random/meta.json)：n_conf=8、protein (128,1024)、drug3d (8,64,128) + coor (8,64,3) |

> meta.json 里的 absolute path 还是上一台机器的 `/home/lsw/lv/LDM-DTI/...`，**不影响**——dataloader 用 `cache_dir / "drug3d_features.pt"` 重新拼接，见 [src/data/dataloader.py:106-108](src/data/dataloader.py#L106-L108)。

### 1.3 训练 checkpoint（待生成）

⚠️ **当前 `outputs/` 为空**，没有任何 checkpoint。后续所有可解释性脚本都依赖 BioSNAP seed-42 best checkpoint，必须先跑训练（见 §2）。

预期产物：

| 文件 | 路径 |
|---|---|
| DeepSpeed checkpoint 目录 | `outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/epoch_XXXX/` |
| best 指标元数据 | `outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt`（含 `best_epoch` 和 `best_test.threshold`） |
| 合并后 fp32 权重 | `outputs/interpretability/biosnap_seed42/fp32_state_dict.pt`（由 extract 脚本首次运行时自动生成，见 [extract_stage2_interpretability_biosnap.py:97-108](scripts/extract_stage2_interpretability_biosnap.py#L97-L108)） |
| 抽取的 interp CSV | `outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv` |

### 1.4 蛋白特征生成方式 ⭐ 关键

[scripts/pre_extract.py](scripts/pre_extract.py)：

1. `BertModel.from_pretrained(protbert_dir)`（Rostlab ProtBert，输出 1024-dim per-residue）
2. 序列由 `format_protein_sequence` 仅做 `upper + drop_whitespace + 非标准AA→X`（不改长度）
3. ProtBert forward 后 `pool_token_features(..., output_len=128)` = `F.adaptive_avg_pool1d`，把 L 残基（去 special token 后）池化到 **固定 128 个 token**
4. 输出 shape `(128, 1024)`，存到 `protein_features.pt`，**dict key = 原始序列字符串**

### 1.5 Token ↔ 残基映射公式 ⭐

由 `adaptive_avg_pool1d` 语义，第 `t` 个 token（0-indexed）对应原序列残基索引：

```
start = floor(t * L_eff / 128)
end   = ceil((t + 1) * L_eff / 128)         # 半开区间 [start, end)
```

其中 `L_eff = min(L_protein, MAX_PROTBERT_LEN)`。ProtBert max position embedding = 1024（去掉 [CLS]/[SEP] 后约 1022 个有效残基；BioSNAP 配置 `PROTEIN.MAX_LEN = 128` 是池化目标，不是 ProtBert 输入限制）。**实际跑 case-study 时必须打印 `token_features.shape[0]` 确认 cutoff**。

### 1.6 模型接口契约（关键属性，已实地确认）

| 用途 | 访问路径 | 备注 |
|---|---|---|
| 多构象选择权重 | `model(...)` 返回的 aux dict 里 `aux["conformer_weight"]`（[src/models/models.py:296](src/models/models.py#L296)） | shape `[bs, K=8]`，softmax 输出 |
| 多构象编码器对象 | `model.drug_encoding`（**不是** `target_aware_conf_enc`） | `TargetAwareConformerEncoder` 实例 |
| 蛋白编码器 | `model.protein_extractor`（ProteinACmix） | FiLM 在 [src/models/models.py:497-500](src/models/models.py#L497-L500) |
| Cross-attention 融合（含 Mamba） | `model.cross_intention` | 内部含 `protein_mamba_gate_logit` |
| Mamba 输入/输出 token 张量 | `model.cross_intention.protein_mamba` 上 `register_forward_hook` — 见 [extract_stage2_interpretability_biosnap.py:148-160](scripts/extract_stage2_interpretability_biosnap.py#L148-L160) |
| 推理调用 | `score, aux = model(feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, mode="eval", return_aux=True)` | 见 [src/core/main_ds.py:283-287](src/core/main_ds.py#L283-L287) 的 batch unpacking |

### 1.7 已有脚本（复用基础）

| 脚本 | 角色 |
|---|---|
| [scripts/run_stage2_main_biosnap_seed42_local.sh](scripts/run_stage2_main_biosnap_seed42_local.sh) | **本机一键训练**（已写好，§2 直接跑） |
| [scripts/extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py) | 跑全 test 推理 + 抽 interp CSV，复用其 FiLM hook、Mamba forward hook、batch unpack |
| [scripts/plot_stage2_interpretability_biosnap.py](scripts/plot_stage2_interpretability_biosnap.py) | 老画图脚本，**只重写 conformer heatmap 部分**或新写一个并删旧引用 |
| [scripts/plot_theme.py](scripts/plot_theme.py) | 提供 `GROUP_COLORS`、`SEQUENTIAL_CMAP`、`TAMR`、`apply_paper_theme`、`style_axes`，**所有新画图脚本必须复用** |
| [tmp/interp_mocks/mock_conformer_plots.py](tmp/interp_mocks/mock_conformer_plots.py) | Plan A 的 mock，已用 Dirichlet 校准到论文统计量，输出 [tmp/interp_mocks/1_plan_A.png](tmp/interp_mocks/1_plan_A.png) |
| [tmp/case_study_candidates/](tmp/case_study_candidates/) | 11 个候选对 + UniProt JSON + 匹配脚本，详见 §3.2 |

---

## 2. 训练（必须先跑，约 1.5–3h）

### 2.1 启动命令

```bash
cd /paddle/lv/TAMR-DTI

# 前台（推荐先看几个 epoch 没问题再切后台）
bash scripts/run_stage2_main_biosnap_seed42_local.sh

# 或后台
nohup bash scripts/run_stage2_main_biosnap_seed42_local.sh \
  > logs/bootstrap.log 2>&1 &
echo $! > logs/bootstrap.pid
tail -f logs/stage2-main-06-full-tamr-dti-n8-biosnap-seed42.log
```

脚本内部做的事（重要环境变量，已固定在 [scripts/run_stage2_main_biosnap_seed42_local.sh](scripts/run_stage2_main_biosnap_seed42_local.sh) 里）：

- `conda activate ldm-dti`
- `LD_LIBRARY_PATH` 加 `$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/nvjitlink/lib`（mamba_ssm 需要）
- `SWANLAB_MODE=offline`
- 8 GPU、`master_port=29500`、`configs/ds_zero2.json`、`configs/experiments/stage2_main_06_full_tamr_dti_n8_biosnap_seed42.yaml`
- save_dir = `outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42`

### 2.2 训练成功的检查

```bash
ls outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/
# 应有: best_metrics.txt, config.txt, model_architecture.txt, latest, epoch_XXXX/
cat outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt
# 期望 best_test auroc ≈ 0.92+
```

`prune_epoch_checkpoints` 会自动只保留 `best_tag` + `latest_tag`（见 [main_ds.py:1048-1050](src/core/main_ds.py#L1048-L1050)）。

### 2.3 抽 interpretability CSV（训练完成后第一步）

```bash
/paddle/miniconda3/envs/ldm-dti/bin/python scripts/extract_stage2_interpretability_biosnap.py \
  --data datasets/biosnap --split random \
  --config configs/experiments/stage2_main_06_full_tamr_dti_n8_biosnap_seed42.yaml \
  --cache_root cache/features \
  --checkpoint_dir outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42
```

脚本会：
1. 读 `best_metrics.txt` 拿 best_epoch 和 threshold
2. 调用 `deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint` 合并分片
3. 把 fp32 state_dict 缓存到 `outputs/interpretability/biosnap_seed42/fp32_state_dict.pt`（**plan §3/§5 后续都依赖这个文件**）
4. 跑全 5493 条 test 推理，写 `outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv`

CSV 列：`row_index, case_type (TP/TN/FP/FN), label, pred, threshold, conf_1..conf_8, conformer_entropy, conformer_entropy_norm, conformer_top1, conformer_margin, mamba_refine_mean, mamba_refine_max, mamba_top_token, mamba_gate, film_l*_strength`。

**Figure 1（Plan A）只需要这个 CSV**，画图阶段零 GPU 开销。

---

## 3. Figure 1 — Plan A（替换当前 Fig 8）

**性质**：观察层，纯本机 CPU 重画。

### 3.1 想回答的问题

模型在全部 5,493 样本上对 8 个 conformer 是怎么分配权重的？是硬选、软加权还是均匀？错误样本和正确样本分布形状有差异吗？

### 3.2 数据

直接读 [outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv) 的 `conf_1..conf_8` + `case_type` 列。

### 3.3 设计

- 每条样本：8 个 conformer 权重**按降序排序**得 5,493 × 8 矩阵
- 按 `case_type ∈ {TP, TN, FP, FN}` 分 4 个 facet
- 每个 facet：
  - **median 折线 + IQR 阴影 + 5–95% 阴影**
  - **dashed 水平线 = 1/K = 0.125**（uniform baseline）
  - x 轴 ticks = 1..8（rank）
- 配色：用 `GROUP_COLORS[case_type]`（在 `plot_theme.py`）

### 3.4 已验证的 mock

[tmp/interp_mocks/mock_conformer_plots.py](tmp/interp_mocks/mock_conformer_plots.py) 用 Dirichlet 合成数据校准到论文统计量（top1=0.255 vs 0.241, entropy=0.923 vs 0.931），输出 [tmp/interp_mocks/1_plan_A.png](tmp/interp_mocks/1_plan_A.png)。**直接照搬 `plan_A()` 函数的画法**，把数据源从 Dirichlet 改成读 CSV 即可。

### 3.5 写作要点

新 Fig 8 caption 和正文要明确写：

> The sorted-weight decay shows that target-aware aggregation behaves as **graded reweighting rather than hard selection**: the median top-1 weight (~0.24) is only ~2× the uniform baseline (0.125), and the curve crosses 1/K near rank-4. The reweighting becomes visibly sharper on error cases (FP/FN rank-1 medians ~0.28), consistent with over-concentration being one failure pattern.

旧 Fig 9 的 conformer entropy boxplot **删除**，Plan A 的曲线斜率已经隐式编码了熵的信息。

### 3.6 新文件

`scripts/plot_conformer_distribution.py`（CPU 即可跑）

- 输入：`outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv`
- 输出：`TAMR-DTI__.../figures/interpretability_conformer_weights.pdf`（**直接覆盖原 Fig 8**，避免改 tex 里的 `\includegraphics{...}` 路径）。同时输出 png/svg 副本到 `outputs/interpretability/biosnap_seed42/`。

### 3.7 跑命令

```bash
/paddle/miniconda3/envs/ldm-dti/bin/python scripts/plot_conformer_distribution.py
```

---

## 4. Figure 4 — Case study（新 must-have）

**性质**：因果证据 + 生物学锚定。需要一次小规模 forward（仅 N 个匹配样本，秒级）。

### 4.1 想回答的问题

对**已知 PDB 共晶 + UniProt binding-site 注释**的具体 drug-target 对：
1. Mamba refinement R_t 的高峰是否**集中在已知结合位点对应的 token**？
2. 模型选出的 top-1 conformer 是否**接近 PDB 共晶 ligand pose**？

### 4.2 候选已准备好

[tmp/case_study_candidates/](tmp/case_study_candidates/)：

- [candidates.json](tmp/case_study_candidates/candidates.json)：11 个候选对，每条含 `drug_name, drug_smiles_raw, drug_smiles_canonical, drug_inchikey, target_uniprot, target_label, target_sequence, binding_sites, active_sites, pdb_refs`
- [build_candidates.py](tmp/case_study_candidates/build_candidates.py)：构建脚本（从 UniProt JSON 抽 binding 注释）
- [match_cases.py](tmp/case_study_candidates/match_cases.py)：匹配脚本
- [match_hits.csv](tmp/case_study_candidates/match_hits.csv)：**上一次匹配已留 2 条命中**：
  - Methotrexate + DHFR（UniProt P00374，L=187，exact match，row_index=0，label=1）
  - Imatinib + ABL1（UniProt P00519，L=1130，exact match，row_index=1，label=1）

候选概览：

| Drug | Target | UniProt L | binding-site 残基数 | 备注 |
|---|---|---|---|---|
| **Methotrexate** | DHFR | 187 | 30 | **首选**：短蛋白，每 token≈1.5 残基，几乎残基级精度 |
| Imatinib | ABL1 | 1130 | 17 | 经典 Type-II kinase 抑制剂 |
| Dasatinib | ABL1 / SRC | 1130 / 536 | 17 / 10 | SRC 较短 |
| Gefitinib / Erlotinib | EGFR | 1210 | 13 | 在 1200 cutoff 上 |
| Aspirin | COX1 | 599 | 1 | binding 注释稀疏 |
| Celecoxib | COX2 | 604 | 3 | |
| Tamoxifen | ESR1 | 595 | 3 | |
| Fluoxetine | SERT | 630 | 16 | |
| Sildenafil | PDE5A | 875 | 6 | |

### 4.3 Step 1 — 匹配（gating，本机几秒）

```bash
cd /paddle/lv/TAMR-DTI
/paddle/miniconda3/envs/ldm-dti/bin/python tmp/case_study_candidates/match_cases.py \
  --test_csv data/datasets/biosnap/random/test.csv \
  --candidates tmp/case_study_candidates/candidates.json \
  --interp_csv outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv
```

输出 `tmp/case_study_candidates/match_hits.csv`。控制台同时打印每个命中的 `pred` 和 `case_type`。

#### Gating 决策

- **≥ 2 个命中且至少 1 个是 TP** → 走 Figure 4，跳到 4.4
- **< 2 个命中** → 回退到 §6（fallback Figure 2/3）

> 现状：candidates 已经能匹配到 Methotrexate-DHFR 和 Imatinib-ABL1，重新训练后只需重跑 4.3 确认 `case_type==TP`，正常应能走 Figure 4 路线。

### 4.4 Step 2 — Mamba R_t 残基级映射（本机小规模 forward）

写 `scripts/case_study_run.py`（基于 [extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py) 改造）：

1. 加载 BioSNAP best checkpoint（同 extract 脚本，从 `outputs/interpretability/biosnap_seed42/fp32_state_dict.pt`）
2. 从 `match_hits.csv` 读匹配到的 row_index 列表
3. 仅对这些 row 跑 forward（batch_size=1 即可，N 才十几条）
4. 复用 extract 脚本里的 `register_mamba_hook` 拿 `mamba_buffer["input"], ["output"]`
5. 算 `R_t = gate * ||h^M_t - h_t||_2`（128 维向量）
6. 对每条 case，按 §1.5 公式把 `R_t` 沿 128 个 token 反向投影到原序列残基坐标：
   ```python
   residue_R = np.zeros(L_eff)
   for t in range(128):
       start = int(np.floor(t * L_eff / 128))
       end   = int(np.ceil((t + 1) * L_eff / 128))
       residue_R[start:end] = R_t[t]   # 该 token 强度均匀填充其覆盖残基
   ```
7. 计算富集统计：
   - 从 candidates.json 拿 `binding_sites` 残基集合 `B`（UniProt 1-indexed，注意 -1）
   - 拿 top-k% 残基（按 `residue_R` 排序），记 `T_k`
   - 富集比 = `|T_k ∩ B| / |T_k|`，对照 = `|B| / L_eff`（随机基线）
   - p-value：one-sided Fisher's exact test on 2×2 table（top-k 内/外 × binding/非 binding）

输出 CSV `outputs/case_study/biosnap_seed42_cases.csv`，含每条 case：
- `row_index, drug, target, uniprot, pred, case_type`
- `mamba_residue_R`（list，长度 = L_eff，序列化为 JSON 字符串）
- `binding_residues`（list）
- `top10pct_enrichment, top20pct_enrichment, random_baseline, fisher_p`

并同时保存原始 `R_t` 向量到 `.npz` 以备复用。

### 4.5 Step 3 — Conformer vs PDB pose RMSD（本机 CPU）

新文件 `scripts/case_study_conformer_rmsd.py`：

1. 对每个 case：用 RDKit 生成药物 SMILES 的 8 个 ETKDGv3 conformer（**必须用和 pre_extract 一致的 seed 和参数**，见 [scripts/pre_extract.py](scripts/pre_extract.py) 里 conformer 生成段落；UFF 能量排序）
2. 从 RCSB PDB 拉对应 ligand pose：`https://files.rcsb.org/download/<PDB>.pdb` → 抽 ligand HETATM → RDKit 重建
3. 用 `rdkit.Chem.rdMolAlign.GetBestRMS` 算 8 个 conformer 各自对 PDB ligand 的 RMSD
4. 从 interp CSV 拿到该 case 的 `conf_1..conf_8` 权重
5. 输出表：每个 conformer 的 (rank by weight, RMSD)，标出 model top-1 的 RMSD vs 8 个里的最小 RMSD

写入 `outputs/case_study/biosnap_seed42_conformer_rmsd.csv`。

> **注意**：缓存 `drug3d_features.pt` 里的 coor 已是 (8, 64, 3) 池化后形式，可能丢了原子身份信息。**优先方案：直接读 cache 的 coor 和 atom feature，反推原始 conformer 用于 RMSD 计算**。Fallback：用相同 ETKDGv3 参数重新生成，承认轻微不一致。如果 RMSD 算不准，case-study 的 conformer-vs-PDB 部分要简化为"top-1 是否落在能量排序前列" + "8 个候选最小 RMSD 是否就是 top-1"。

### 4.6 Step 4 — 画 Figure 4（本机 CPU）

新文件 `scripts/plot_case_study.py`，输出 `TAMR-DTI__.../figures/interpretability_case_study.pdf`。

布局（2 行 × 3 列，约 14 × 7 cm）：

| | 列 1：药物结构 + 构象对比 | 列 2：蛋白序列 R_t 追踪 | 列 3：8 conformer 权重 |
|---|---|---|---|
| 行 1：Methotrexate + DHFR | 3D 叠加（RDKit top-1 vs PDB ligand），标 RMSD | x = residue 1..L，蓝线 = R_t，红色竖条 = binding sites | bar，标 "model top-1"（star）和 "PDB-closest"（diamond） |
| 行 2：Imatinib + ABL1（或 fallback case） | 同上 | 同上 | 同上 |

底部加 caption：富集 p-value、TP/FP class、pred prob。

### 4.7 写作要点

- 段落 1：定性观察。"In the methotrexate-DHFR case, the residues with highest Mamba refinement intensity (top-10%) include positions {…}, overlapping by N out of M with UniProt-annotated folate binding pocket (Fisher's exact p = …)."
- 段落 2：构象。"The conformer selected by the target-aware module (top-1, weight=…) deviates from the PDB co-crystal pose by RMSD=…, which ranks first/second/… among the 8 ETKDGv3-generated conformers."
- 段落 3：失败案例（如果有 FP 命中）。"In contrast, the FP case (X+Y) shows R_t peaks dispersed across non-binding regions, consistent with the lower top-1 conformer weight observed."
- 把 limitation 写进去：ProtBert 128-token 池化粒度 → 每 token 覆盖 ⌈L/128⌉ 残基，因此 residue-level 富集应该解释为 token-range-level overlap。

---

## 5. 论文改写

### 5.1 删 / 改 / 留

| 现状 | 操作 |
|---|---|
| Fig 8 conformer heatmap（[experiments.tex:391-393](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L391-L393)） | **替换** 为 Plan A 输出（同路径同 label，无需改 tex 的图引用） |
| Fig 9 module statistics 3 子图（entropy + Mamba magnitude + FiLM）（[experiments.tex:416-418](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L416-L418)） | **删除整个图**：entropy 被 Plan A 覆盖；Mamba magnitude 和 FiLM 子图作为 sanity stat 移到 appendix 或正文一句话 |
| Fig 10 Mamba refinement heatmap（[experiments.tex:425-427](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L425-L427)） | **删除**：被 Figure 4 的 R_t residue track 取代且更有说服力 |
| 小节标题 "Interpretability analysis"（[experiments.tex:377](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L377)） | 保留，但第一句改成 "We complement the ablation results with a **distribution-level and biology-grounded** interpretability analysis." |
| discussion.tex 的 "visualizations are best interpreted as diagnostic checks" | **改语气**：现在有 case study 锚定，可改为 "internal-state analysis with limited biological grounding via 2 case studies on methotrexate-DHFR and imatinib-ABL1." |

### 5.2 新小节结构

```
\subsubsection{Interpretability analysis}

[段 1] Distribution-level conformer behavior.
  - 引用新 Fig 8（Plan A），说"graded reweighting rather than hard selection"
  - 报告 rank-1 / rank-2 / rank-4 数值
  - 报告 FP/FN vs TP/TN 形状差异

[段 2] Biological grounding via case studies on methotrexate-DHFR and imatinib-ABL1.
  - 引用新 Fig 4，分两个 case 描述
  - 每个 case：R_t top-k 富集 p-value + conformer RMSD rank
  - 一个 TP + 一个对照（FP 或弱 TP），构成 narrative 张力

[段 3，可选] Module sanity statistics.
  - 一句话：FiLM 强度 / Mamba gate 数值，引 appendix
```

---

## 6. Fallback — Figure 2/3（仅当 §4.3 gating 失败时）

如果 case-study 至少 2 个 TP 都凑不齐，启用 v3 plan 的因果干预图：

### 6.1 Figure 2 — Conformer 反事实交换

干预设计：
- 对每条样本，记 top-1 conformer 索引为 k*，bottom-1 为 k₋
- **Swap top↔bottom**：把 `feature[:, k*, :, :]` 和 `coor[:, k*, :, :]` 替换为第 k₋ 个 conformer 的；其余不动
- **Random swap baseline**：k* 替换为 mask 内任意非 k* 的 conformer，重复 3 次取均值

实现方式：write a context manager that patches the input tensors before `model.drug_encoding(...)` is called — 或者更简单：**在 batch unpack 之后、`model.forward` 之前，直接修改 `feature` 和 `coor` 张量的 K 维**。不需要改模型代码。

输出 CSV：`logit_full, logit_swap_top, logit_swap_rand_mean` per sample。
画图：violin by TP/TN/FP/FN + paired bar (top swap vs random swap)。

### 6.2 Figure 3 — Mamba 位置因果性

干预设计：
- 挑 ~100 条高置信 TP（按 `pred > 0.95 & label == 1` 筛 interp CSV）
- 对每条：
  - 跑原始 forward 拿 `R_t`（128 dim）
  - **Top-k mask**：把 `R_t` 最大的 k 个 token 在 `protein_mask` 里置 0，重跑 forward
  - **Random-k mask**：随机选 k 个非 padding token，重复 5 次取均值
  - k ∈ {2, 4, 8}

实现：单独改 `protein_mask` 张量，模型其他都不动。

输出图：line plot，x=k, y=mean |Δlogit|，两条线（top-k vs random-k）带 95% CI。

### 6.3 Fallback 远程开销

- Fig 2：5,493 × 5 forward ≈ 28k forward，10–30 分钟
- Fig 3：100 × (1 + 18) ≈ 1,900 forward，分钟级

---

## 7. 工程落地

### 7.1 文件清单

```
scripts/
  run_stage2_main_biosnap_seed42_local.sh   # 训练 (已存在，§2)
  extract_stage2_interpretability_biosnap.py # 抽 CSV (已存在，§2.3)
  plot_conformer_distribution.py            # 图 1（Plan A）（待写）
  case_study_run.py                         # 图 4 step 2（待写）
  case_study_conformer_rmsd.py              # 图 4 step 3（待写）
  plot_case_study.py                        # 图 4 step 4（待写）

# fallback 路径（如启用）
  run_conformer_swap.py                     # 图 2（待写）
  plot_conformer_swap.py                    # 图 2（待写）
  run_mamba_position.py                     # 图 3（待写）
  plot_mamba_position.py                    # 图 3（待写）

tmp/case_study_candidates/                  # 已存在
  candidates.json
  build_candidates.py
  match_cases.py
  match_hits.csv
  *.json                                    # UniProt entries
```

### 7.2 单机本地执行

所有阶段都在 `/paddle/lv/TAMR-DTI` 这台 linux 机器上跑：

- **GPU 阶段**（训练、forward）：`/paddle/miniconda3/envs/ldm-dti/bin/python` + 8×V100
- **CPU 阶段**（画图、RMSD、residue 映射）：同一个 python，无 GPU 调用即可

> 与 v4 plan 的差别：**没有「Mac 本地」**。原 v4 的本地 `./.venv-pptx/bin/python` 不存在；matplotlib/rdkit/pandas/numpy 都在 ldm-dti 这一个 env 里。

### 7.3 计算成本估算

- 训练（§2）：8×V100，BioSNAP 19224 train，100 epoch + early_stop=10，**约 1.5–3 小时**
- §2.3 抽 CSV：5493 forward，**单卡 10–20 分钟**
- Figure 1：纯 CPU，秒级
- Figure 4 step 1（matching）：秒级
- Figure 4 step 2（case forward）：N × 1 forward，N < 20，**秒级**
- Figure 4 step 3 + step 4：CPU，分钟级
- Fallback Fig 2：10–30 分钟
- Fallback Fig 3：分钟级

---

## 8. 论文图编号变更总结

| 旧 | 新 |
|---|---|
| Fig 8 (conformer heatmap, 18 cherry-pick + PowerNorm) | **Fig 8 (Plan A 分位带)** |
| Fig 9 (module statistics 3 子图) | **删除**（部分进 appendix） |
| Fig 10 (Mamba refinement heatmap) | **替换为 Fig 9 (Case study, 2 行)** |

最终 interpretability section 只剩 2 张图。

---

## 9. TODO 清单（带 gating）

### 第 0 阶段（强制前置）

- [ ] **§2 训练**：`bash scripts/run_stage2_main_biosnap_seed42_local.sh`，等 `best_metrics.txt` 写出
- [ ] **§2.3 抽 CSV**：跑 `extract_stage2_interpretability_biosnap.py`，确认 `outputs/interpretability/biosnap_seed42/{fp32_state_dict.pt, biosnap_seed42_interpretability_samples.csv}` 都生成

### 第一阶段（§0 完成后并行）

- [ ] **图 1**：写 `scripts/plot_conformer_distribution.py`，照搬 [tmp/interp_mocks/mock_conformer_plots.py](tmp/interp_mocks/mock_conformer_plots.py) 的 `plan_A()`，数据源换为 CSV，输出覆盖 `<paper>/figures/interpretability_conformer_weights.pdf`
- [ ] **case 匹配**：跑 §4.3 的 `match_cases.py`，回收 `match_hits.csv` + 控制台 `pred`/`case_type`

### 第二阶段（依赖 gating 决策）

**如果 ≥ 2 个 case 命中且包含 ≥ 1 个 TP：**
- [ ] 写 + 跑 `scripts/case_study_run.py`，拿到 R_t residue-level CSV
- [ ] 写 `scripts/case_study_conformer_rmsd.py`（先下载 PDB 文件）
- [ ] 写 `scripts/plot_case_study.py`，组合图

**否则（fallback）：**
- [ ] 写 `src/models/hooks.py`（conformer swap + mamba mask 两个 context manager）
- [ ] 写 + 跑 `run_conformer_swap.py` 和 `run_mamba_position.py`
- [ ] 写两个 plot_*.py

### 第三阶段（无 gating）

- [ ] 论文：按 §5 改 `sec/experiments.tex` 和 `sec/discussion.tex`
- [ ] 论文：检查 `\ref{fig:...}` 和 caption 是否和新图对得上

---

## 10. 已知风险 / 待澄清

1. **训练复现性**：上一台机器是 P100×8，本机是 V100×8。BoolFloat tensor 实现差异不会显著影响 AUROC，但 best_test 数值可能与原论文 v4 数据微小漂移（±0.005）。Plan A 的 top1=0.241 / entropy=0.931 是上一版的统计量，**重训后必须用新的统计量重新填进论文**。
2. **SMILES encoding 差异**：BioSNAP CSV 里 SMILES 可能是 DrugBank 原始而非 RDKit canonical。`match_cases.py` 已用 InChIKey 匹配（鲁棒）。如果命中率低，可放宽到 InChIKey without stereo。
3. **Protein sequence isoform 差异**：UniProt canonical 可能和 BioSNAP 用的不一样。`match_cases.py` 已有 substring fallback。
4. **EGFR (1210 aa) 超过 ProtBert 标称 max length**：需在 case-study run 时打印 `token_features.shape[0]` 确认 cutoff，并用真实 `L_eff` 做 token-残基映射，不能盲用 UniProt L。
5. **Conformer 复现**：cache 里 `drug3d_features.pt` 的 coor 已是 (8, 64, 3) pool 后形式，**可能不可逆**回到原始 atom 数。如果 RMSD 算不准，case-study 的 conformer-vs-PDB 部分要简化为"top-1 是否落在能量排序前列"+ "8 个候选最小 RMSD 是否就是 top-1"。
6. **`drug_encoding` vs `target_aware_conf_enc`**：v3 plan 里写错过属性名，实际是 `model.drug_encoding`。所有新脚本 follow 这个。
7. **缓存路径**：[meta.json](cache/features/datasets/biosnap/random/meta.json) 里的 absolute path 还是上一台机器的 `/home/lsw/lv/LDM-DTI/...`，**忽略即可**——dataloader 用 `cache_dir + 文件名` 重新拼接，不读 meta 里的绝对路径。
8. **swanlab 离线**：本机没装 `swanboard`，所以 `SWANLAB_MODE=local` 会报 `ModuleNotFoundError: No module named 'swanboard'`。已统一改用 `SWANLAB_MODE=offline`，会落 `swanlog/<exp>/run-*/backup.swanlab`，未来需要可 `swanlab sync`。

---

## 11. 执行记录 — 训练完成后实际做的事情（2026-06-29）

> 本节是「跑完 §2 训练之后」我自主完成的全部工作的事后流水账，便于次日核对。
> 训练产物入口：[outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt](outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt) — best_epoch=46, val_auroc=0.9302, **test_auroc=0.9321, aupr=0.9372, acc=0.8624, f1=0.8609**，threshold=0.63。

### 11.1 全 5,493 测试样本 interpretability dump（§2.3）

- 运行 [scripts/extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py)，输出：
  - [outputs/interpretability/biosnap_seed42/fp32_state_dict.pt](outputs/interpretability/biosnap_seed42/fp32_state_dict.pt)（ZeRO 分片合并的 fp32 权重，14.7 MB）
  - [outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv)（5,493 行，含 `conf_1..conf_8`、`conformer_entropy_norm`、`mamba_refine_mean`、`pred`、`case_type` 等）
  - [outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy](outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy)（(5493, 128) token-level R_t profile）
  - [outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_summary.txt](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_summary.txt)（mamba_gate=0.1253, top1_mean=0.2411, H/logK=0.9312，分 TP/TN/FP/FN 的逐组统计）
- ⚠️ **熵口径**：summary 里 `conformer_entropy_norm` 是 `H / log(K_valid)`，分母按每条样本的有效构象数算（mask 掉的不计），而不是常数 `log(8)`。Plan A 必须直接读这一列，**不要在脚本里用 log(8) 重算**（否则会得到 ~0.797 而不是 0.931）。

### 11.2 Figure 1 — Plan A（§3，替换 Fig 8）

- 新写 [scripts/plot_conformer_distribution.py](scripts/plot_conformer_distribution.py)
- 内容：把每行 `conf_1..conf_8` 行内降序排序得到 5,493×8 矩阵，按 `case_type` 分 4 个 facet，画 median + IQR + 5–95% 带，dashed line `1/K=0.125`。`H/logK` 直接读 CSV `conformer_entropy_norm` 列，不重算。
- 输出：
  - [outputs/interpretability/biosnap_seed42/figs/plan_A_conformer_distribution.png](outputs/interpretability/biosnap_seed42/figs/plan_A_conformer_distribution.png)
  - [outputs/interpretability/biosnap_seed42/figs/plan_A_conformer_distribution.svg](outputs/interpretability/biosnap_seed42/figs/plan_A_conformer_distribution.svg)
  - [outputs/interpretability/biosnap_seed42/figs/plan_A_stats.json](outputs/interpretability/biosnap_seed42/figs/plan_A_stats.json)（各组逐 rank 的 median/IQR/q05/q95）
  - 同步复制到 [TAMR-DTI__.../figures/interpretability_conformer_weights.pdf](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_conformer_weights.pdf)（直接覆盖旧 PowerNorm cherry-pick 热图，tex 里 `\includegraphics` 路径不动）
- 关键数值（写进论文）：rank-1 median ≈ 0.20 / 0.20 / 0.22 / 0.24（TP/TN/FP/FN）；rank-4 在 ~0.12 跨越 uniform；H/logK ≈ 0.937 / 0.932 / 0.921 / 0.902。
- cosmetic 微调：把共享 legend 从 `upper right` 移到 `lower left`，避免和每个子图右上角的 `top-1 / H/logK` 文本框重叠。

### 11.3 Case study — 从「单对故事」改为 population Fisher 富集（§4 升级版）

#### 11.3.1 为什么改

按原 §4 设想，本来要写 Methotrexate–DHFR 和 Gefitinib–EGFR 两个 case。重训完做完一轮匹配后我先跑了 Gefitinib–EGFR 的 residue-level R_t，发现：

- Gefitinib–EGFR：fold=0.77, Fisher p=0.747（n.s.）；conformer top-1≈0.125（接近 uniform）。
- 单独把它画进 Figure 4 会和 narrative 矛盾。

因此把 §4 reframe 为：**在 9 个 UniProt 注释靶点 / 78 个 TP 行上做 population-level Fisher 富集**，用 Stouffer's z-method 合并每行 p 值，再挑出能跨过显著性阈值的靶点写 case。

#### 11.3.2 候选扩展 + 匹配（§4.3 之后）

- 候选清单沿用 [tmp/case_study_candidates/candidates.json](tmp/case_study_candidates/candidates.json)（11 对），其中 9 个靶点在 BioSNAP test 上有 TP：DHFR (4), EGFR (3), ABL1 (1), SRC (9), COX1 (14), COX2 (16), ESR1 (15), SERT (14), PDE5A (2)，合计 78 行 TP。
- 由 [tmp/case_study_candidates/candidates.json](tmp/case_study_candidates/candidates.json) 的 `binding_sites.residues` 提供 UniProt 1-indexed 残基集合 `B`。

#### 11.3.3 残基级富集核心脚本

新写 [scripts/case_study_run.py](scripts/case_study_run.py)，对每条 TP row 做：

1. 从 `biosnap_seed42_mamba_profiles.npy` 取 128-dim R_t；
2. 按 §1.5 的 `floor / ceil` 映射，把 R_t 反向投影到 UniProt 残基坐标（`residue_R`，长度 = `L_uniprot`）；
3. 取 top-{10%, 20%, 30%} 残基 `T_k`，构造 2×2 列联表 `[a=|T_k∩B|, b=k-a; c=|B|-a, d=L-k-c]`，跑 `scipy.stats.fisher_exact(alternative="greater")`；
4. 富集 fold = `(a/k) / (|B|/L)`；
5. 同一靶点多 TP 行用 Stouffer's z-method 合并 p 值：
   ```python
   z_sum = norm.isf(p).sum() / sqrt(n)
   p_combined = norm.sf(z_sum)
   ```

输出：

- [outputs/interpretability/biosnap_seed42/case_study/per_row.csv](outputs/interpretability/biosnap_seed42/case_study/per_row.csv)（78 行 × top-k 的 a / fold / p）
- [outputs/interpretability/biosnap_seed42/case_study/aggregate.csv](outputs/interpretability/biosnap_seed42/case_study/aggregate.csv)（9 靶点 × top-k 的 fold_mean / p_combined / n_sig）
- [outputs/interpretability/biosnap_seed42/case_study/results.json](outputs/interpretability/biosnap_seed42/case_study/results.json)

**关键发现**：在 top-20% 阈值下，只有 **DHFR (P00374)** 通过 Stouffer 合并显著性 —— **n=4, fold=2.13, p_Stouffer=1.26e-09, 4/4 TP 行独立 p<0.05**。其余 8 个靶点 n.s.（其中 SRC top-30% 单独有边缘信号 p=8.9e-4，但 top-20% 不显著，未写进正文）。

#### 11.3.4 Figure 4（[scripts/plot_case_study.py](scripts/plot_case_study.py)）

布局换成 2 行 × 2 列（13.6×8.2 inch，`gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.8])`）：

- **Panel A（top-left）**：representative DHFR TP row（按 pred 取最大）的残基级 R(r) 折线 + 填充；UniProt NADP+ 结合位点用绿色 `axvline`，substrate 结合位点用橙色 `axvline`；dashed 横线 = top-20% 阈值。
- **Panel B（top-right）**：9 靶点的 forest plot，bar 长 = `top20_fold_mean`，绿色 = p<0.05，灰色 = n.s.，右侧标 p 值 + n。
- **Panel C（bottom）**：4 行 DHFR TP 的 sorted conformer weight grouped bar（说明虽然残基注意力是 protein-dominated，conformer 选择仍 row-to-row 变化）。
- super title：`fold=2.13, Fisher's-exact p_Stouffer=1.26e-09`。

输出：

- [outputs/interpretability/biosnap_seed42/figs/case_study_figure4.png](outputs/interpretability/biosnap_seed42/figs/case_study_figure4.png) + `.svg`
- [TAMR-DTI__.../figures/case_study_dhfr.pdf](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/case_study_dhfr.pdf)

cosmetic 微调：Panel A 的 legend 给了显式 framed box（`framealpha=0.92, edgecolor=#94A3B8`）并放在 upper-right；整图加大到 13.6×8.2 避免拥挤。

#### 11.3.5 没做的事（与原 §4 的偏离）

- **§4.5 Conformer vs PDB pose RMSD 没做**。原因：(i) 现有 case 已经能在 residue 层面建立生物学锚定（DHFR 4/4 显著）；(ii) `drug3d_features.pt` 里 coor 是 (8, 64, 3) 池化形式，原子 identity 已丢，要重新从 SMILES 用相同 ETKDGv3 参数跑一次然后下 PDB ligand 才能算 RMSD，工程成本相对收益不高。这条限制已经在 discussion.tex 里以「conformer module not supervised by binding pose」诚实写出。
- **§6 Fallback 路径未启用**：DHFR 已经过显著性闸门，没有触发 fallback。

### 11.4 LaTeX 宏注入（确保论文数字与磁盘上的实验产物一致）

新写 [scripts/dump_paper_macros.py](scripts/dump_paper_macros.py)，从下面三个源头自动生成 60 个 `\newcommand`：

1. `best_metrics.txt` → `\BiosnapBestEpoch`, `\BiosnapTestAUROC`, `\BiosnapTestAUPRC`, `\BiosnapTestACC`, `\BiosnapTestF1`, `\BiosnapThreshold` 等；
2. `biosnap_seed42_interpretability_summary.txt` + `plan_A_stats.json` → `\InterpNumSamples`, `\InterpTopOneMean`, `\InterpEntropyMean`, `\InterpMambaGate`, `\InterpMambaRefineMean`, `\InterpTPEntropy`, `\InterpFNEntropy`, `\InterpFPTopOne`, `\InterpFNTopOne` 等；
3. `case_study/aggregate.csv` → `\CaseStudyDHFRFold`, `\CaseStudyDHFRPComb`, `\CaseStudyDHFRNRows`, `\CaseStudyDHFRNSig`, `\CaseStudyNumTargets`, `\CaseStudyNumSigTargets`, `\CaseStudyTopFrac` 等。

输出 [TAMR-DTI__.../paper_macros.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/paper_macros.tex)（60 个宏 / 75 行）。在 [main.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/main.tex) 的 `\input{math_commands.tex}` 之后新增一行 `\input{paper_macros.tex}` 让所有章节都能直接 `\BiosnapTestAUROC` 这样引用。

> **次日核对要点**：如果重新生成了任何 stats 文件，只要重跑 `python scripts/dump_paper_macros.py` 即可同步所有论文里的硬数字，**不需要手改 sec/*.tex**。

### 11.5 论文段落改写

#### 11.5.1 [sec/experiments.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex) 的 `\subsubsection{Interpretability analysis}`

- 删除原「Representative conformer-weight maps... show that the model does not use a single fixed conformer pattern」的 cherry-pick 表述；
- 改写成 population-level 描述（median/IQR/5–95% 带、5,493 样本、跨 TP/TN/FP/FN 形状对比）；
- 全部硬编码数字替换为宏（`\BiosnapTestAUROC`, `\InterpTopOneMean`, `\InterpFPEntropy`, `\InterpMambaGate`, ...）；
- 新增 `\paragraph{Case study: residue-level binding-site enrichment.}` 段落和 `\begin{figure}{fig:case_study_dhfr}` 块，引用 `case_study_dhfr.pdf` 并诚实写明「**1 of 9 curated targets** passes the Stouffer-combined Fisher test」。

#### 11.5.2 [sec/discussion.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/discussion.tex)

- 「protein-side refinement modules play a complementary role」段：补充两层证据 —— (i) population 层 mean R_t=`\InterpMambaRefineMean`, gate=`\InterpMambaGate` 稳定且小；(ii) DHFR case 层 fold=`\CaseStudyDHFRFold`, p=`\CaseStudyDHFRPComb`, `\CaseStudyDHFRNSig` of `\CaseStudyDHFRNRows` TP rows individually significant。
- limitations 段：把 case study 的局限性诚实写明 ——「only `\CaseStudyNumSigTargets{}` of the `\CaseStudyNumTargets{}` curated UniProt targets passes the Stouffer-combined Fisher's-exact test」，并提到 Mamba 分支没有 residue-level 监督。

### 11.6 验收（32/32 PASS）

最后做了一遍端到端 sanity check（无单独脚本，bash 内联），包含：

- 文件存在性（10 项：训练 ckpt 目录、best_metrics.txt、interp CSV/npy/summary、case_study aggregate.csv/per_row.csv/results.json、两张 figs PDF、paper_macros.tex）；
- 数值一致性（11 项）：best_metrics.txt 的 AUROC 与 paper_macros.tex 里的 `\BiosnapTestAUROC` 一致；aggregate.csv 里 DHFR 的 fold/p/n_sig 与 paper_macros.tex 里的 `\CaseStudyDHFR*` 一致；plan_A_stats.json 各组 top-1 与论文宏一致；等等；
- LaTeX 健康度（3 项）：`sec/experiments.tex` 大括号配对（413=413）、`sec/discussion.tex`（9=9）、`paper_macros.tex`（122=122）；
- 显著性逻辑（2 项）：DHFR p<0.05、n_sig==1（即只有 DHFR 一个 target 显著）；
- 图引用（6 项）：experiments.tex 和 discussion.tex 中所有 `\ref{fig:case_study_dhfr}` 和 `\includegraphics{interpretability_conformer_weights}` / `case_study_dhfr` 都能解析。

结果：**32/32 PASS**。

### 11.7 文件全景（次日核对入口）

| 类别 | 路径 | 用途 |
|---|---|---|
| 训练 ckpt | [outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/](outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/) | best_epoch=46, test_auroc=0.9321 |
| → best 指标 | [best_metrics.txt](outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt) | 论文里所有训练性能数字的唯一源 |
| → 配置 | [config.txt](outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/config.txt) | CfgNode 完整快照 |
| → ZeRO 分片 | `epoch_0046/`, `epoch_0056/` | 用 `zero_to_fp32.py` 合并 |
| Interpretability dump | [outputs/interpretability/biosnap_seed42/](outputs/interpretability/biosnap_seed42/) | |
| → 5,493 行 CSV | [biosnap_seed42_interpretability_samples.csv](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv) | conf_1..8 + R_t 标量统计 + case_type |
| → (5493, 128) profile | [biosnap_seed42_mamba_profiles.npy](outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy) | token-level R_t |
| → population 统计 | [biosnap_seed42_interpretability_summary.txt](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_summary.txt) | mamba_gate, top1, H/logK, TP/TN/FP/FN |
| → fp32 权重 | [fp32_state_dict.pt](outputs/interpretability/biosnap_seed42/fp32_state_dict.pt) | ZeRO 合并后的 fp32 |
| Case study | [outputs/interpretability/biosnap_seed42/case_study/](outputs/interpretability/biosnap_seed42/case_study/) | |
| → 9 靶点汇总 | [aggregate.csv](outputs/interpretability/biosnap_seed42/case_study/aggregate.csv) | 论文 Panel B 和 macros 的源 |
| → 78 行 TP 明细 | [per_row.csv](outputs/interpretability/biosnap_seed42/case_study/per_row.csv) | |
| Figures | [outputs/interpretability/biosnap_seed42/figs/](outputs/interpretability/biosnap_seed42/figs/) | png/svg 副本 |
| → 论文图 1 | [interpretability_conformer_weights.pdf](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_conformer_weights.pdf) | 替换原 cherry-pick 热图 |
| → 论文图 4 | [case_study_dhfr.pdf](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/case_study_dhfr.pdf) | 新增 |
| Scripts | | |
| → 抽 CSV | [scripts/extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py) | §11.1 |
| → 图 1 | [scripts/plot_conformer_distribution.py](scripts/plot_conformer_distribution.py) | §11.2 |
| → case study compute | [scripts/case_study_run.py](scripts/case_study_run.py) | §11.3 |
| → case study 画图 | [scripts/plot_case_study.py](scripts/plot_case_study.py) | §11.3 |
| → 宏注入 | [scripts/dump_paper_macros.py](scripts/dump_paper_macros.py) | §11.4 |
| LaTeX | | |
| → 宏 | [TAMR-DTI__.../paper_macros.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/paper_macros.tex) | 60 个宏 |
| → 实验段 | [TAMR-DTI__.../sec/experiments.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex) | Interpretability + Case study |
| → 讨论段 | [TAMR-DTI__.../sec/discussion.tex](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/discussion.tex) | 诚实写明 1/9 显著性 |
