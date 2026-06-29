# TAMR-DTI 可解释性升级方案（v4 — self-contained）

> **本文档面向远程机器上接手的另一个 Claude / 工程师。** 假设你只有这份文档 + 当前仓库，不需要追问背景就能执行。所有路径、文件名、属性名、CLI 都是实地验证过的。

---

## 0. 背景（为什么这个 plan 存在）

论文 `sec/experiments.tex` 的 **Interpretability analysis** 子节有三个硬伤：

1. **图和统计互相打脸**：aggregate 统计（top1 weight=0.241、normalized entropy=0.931，全 5,493 样本）说明 conformer 选择是软加权；但 [Figure 8 (conformer heatmap)](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_conformer_weights.pdf) 因为 (i) cherry-pick 各组最自信的 18 条 + (ii) `PowerNorm(gamma=0.85)` 视觉压低小权重，**看起来像硬选择**。审稿人对得起来就会扣分。
2. **三张图全是 "模块输出非零" 的 sanity check**：没有任何反事实或外部对照，无法支撑 "interpretability" 这个标题。`sec/discussion.tex` 已经自己降级为 "diagnostic checks"，但 experiments 仍叫 "Interpretability analysis"——narrative 不一致。
3. **Mamba 热图没有锚**：只是 "不同样本有不同峰"，无法说明峰是有意义的。

### v4 总目标

只做 **2 张图**（其中 1 张完全本地，1 张需要远程小规模 forward）：

- **Figure 1（替换当前 Fig 8）= Plan A**：观察层。population-level conformer 分布。
- **Figure 4（新增）= Case study**：因果层 + 生物学锚定。用 2 个有 PDB 共晶 + UniProt binding-site 注释的经典 drug-target 对，把 Mamba refinement 峰反向映射到残基，和已知结合位点对照。

> **Figure 2/3（v3 里的 conformer swap、Mamba top-k mask）降级为 fallback**：仅当 case-study 匹配失败时启用，作为 "因果但无 ground-truth" 的替代证据。

---

## 1. 仓库现状（必读上下文）

### 1.1 路径约定（远程，linux）

| 类型 | 路径 |
|---|---|
| 训练数据 | `data/<dataset>/<split>/test.csv` 等（e.g. `data/biosnap/random/test.csv`） |
| 特征缓存 | `cache/features/<dataset>/<split>/{protein_features.pt, smiles_features.pt, drug_3d_features.pt}` |
| 主要 checkpoint 目录 | `outputs/interpretability/biosnap_seed42/fp32_state_dict.pt`（已存在，是 BioSNAP seed-42 best） |
| 已抽取的 interpretability CSV | `outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv` |
| 远程 Python | `/home/lsw/miniconda3/envs/ldm-dti/bin/python`（含 torch、rdkit、mamba_ssm、deepspeed 等） |
| 论文目录 | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/` |

### 1.2 BioSNAP test CSV 列结构

通过 `src/data/dataloader.py:145-146` 可知列名为：

- `SMILES`：原始 SMILES 字符串（cache key）
- `Protein`：原始氨基酸序列字符串（**未截断**，cache key）
- `Y`：0/1 label

### 1.3 蛋白特征生成方式 ⭐ 关键

`scripts/pre_extract.py`：

1. `BertModel.from_pretrained(model_dir)`（Rostlab ProtBert，输出 1024-dim per-residue）
2. 序列由 `format_protein_sequence` 仅做 `upper + drop_whitespace + 非标准AA→X`（不改长度）
3. ProtBert forward 后 `pool_token_features(..., output_len=128)` = `F.adaptive_avg_pool1d`，把 L 残基（去 special token 后）池化到 **固定 128 个 token**
4. 输出 shape `PROTEIN_SHAPE = (128, 1024)`，存到 `protein_features.pt`，**dict key = 原始序列字符串**

### 1.4 Token ↔ 残基映射公式 ⭐

由 `adaptive_avg_pool1d` 的语义，第 `t` 个 token（0-indexed）对应原序列残基索引：

```
start = floor(t * L_eff / 128)
end   = ceil((t + 1) * L_eff / 128)         # 半开区间 [start, end)
```

其中 `L_eff = min(L_protein, MAX_PROTBERT_LEN)`。ProtBert max position embedding = 1024（去掉 [CLS]/[SEP] 后约 1022 个有效残基；BioSNAP 配置 `PROTEIN.MAX_LEN = 128` 是池化目标，不是 ProtBert 输入限制）。**实际验证 cutoff 时打印 token_features.shape[0]**。

### 1.5 模型接口契约（关键属性，已实地确认）

| 用途 | 访问路径 | 备注 |
|---|---|---|
| 多构象选择权重 | `model(...)` 返回的 aux dict 里 `aux["conformer_weight"]`（[models.py:296](src/models/models.py#L296)） | shape `[bs, K=8]`，softmax 输出 |
| 多构象编码器对象 | `model.drug_encoding`（**不是** `target_aware_conf_enc`） | `TargetAwareConformerEncoder` 实例，forward 输入 `feature, coor, conf_mask, energy, protein_tokens, protein_mask` |
| 蛋白编码器 | `model.protein_extractor`（ProteinACmix） | FiLM 在 [models.py:497-500](src/models/models.py#L497-L500)，本 plan 不用 |
| Cross-attention 融合（含 Mamba） | `model.cross_intention` | 内部含 `protein_mamba_gate_logit` 标量（已被 [extract 脚本](scripts/extract_stage2_interpretability_biosnap.py) 使用） |
| Mamba 输入/输出 token 张量 | 在 `model.cross_intention` 的某个 `protein_mamba` 子模块上 `register_forward_hook` — 见 [extract_stage2_interpretability_biosnap.py:139-160](scripts/extract_stage2_interpretability_biosnap.py#L139-L160) |
| 推理调用 | `score, aux = model(feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, mode="eval", return_aux=True)` | 见 [main_ds.py](src/core/main_ds.py) 的 batch unpacking |

### 1.6 已有 interp CSV 列

[outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv) 每行 5,493 条 BioSNAP test 样本，含：

`row_index, case_type (TP/TN/FP/FN), label, pred, threshold (0.60), conf_1..conf_8, conformer_entropy, conformer_entropy_norm, conformer_top1, conformer_margin, mamba_refine_mean, mamba_refine_max, mamba_top_token, mamba_gate, film_l*_strength`

**Figure 1（Plan A）只需要这个 CSV**，零远程开销。

### 1.7 已存在的脚本

- [scripts/extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py)：跑全 test 推理 + 抽 interp CSV。**复用其 hook 机制（FiLM hook、Mamba forward hook）和 batch unpack**。
- [scripts/plot_stage2_interpretability_biosnap.py](scripts/plot_stage2_interpretability_biosnap.py)：当前画 Fig 8/9/10 的脚本。**只重写 conformer heatmap 部分**。
- [scripts/run_stage2_interpretability_biosnap.sh](scripts/run_stage2_interpretability_biosnap.sh)：runner，含远程 python 路径。
- [scripts/plot_theme.py](scripts/plot_theme.py)：提供 `GROUP_COLORS`、`SEQUENTIAL_CMAP`、`TAMR`、`apply_paper_theme`、`style_axes`。**所有新画图脚本必须复用**。

---

## 2. Figure 1 — Plan A（替换当前 Fig 8）

**性质**：观察层，零远程开销，仅本地重画。

### 2.1 想回答的问题

模型在全部 5,493 样本上对 8 个 conformer 是怎么分配权重的？是硬选、软加权还是均匀？错误样本和正确样本分布形状有差异吗？

### 2.2 数据

直接读 [biosnap_seed42_interpretability_samples.csv](outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv) 的 `conf_1..conf_8` + `case_type` 列。

### 2.3 设计

- 每条样本：8 个 conformer 权重**按降序排序**得 5,493 × 8 矩阵
- 按 `case_type ∈ {TP, TN, FP, FN}` 分 4 个 facet
- 每个 facet：
  - **median 折线 + IQR 阴影 + 5-95% 阴影**
  - **dashed 水平线 = 1/K = 0.125**（uniform baseline）
  - x 轴 ticks = 1..8（rank）
- 配色：用 `GROUP_COLORS[case_type]`，调色板已在 `plot_theme.py`

### 2.4 已验证的 mock

[tmp/interp_mocks/mock_conformer_plots.py](tmp/interp_mocks/mock_conformer_plots.py) 已用 Dirichlet 合成数据校准到论文统计量（top1=0.255 vs 0.241, entropy=0.923 vs 0.931），输出 [tmp/interp_mocks/1_plan_A.png](tmp/interp_mocks/1_plan_A.png)。**直接照搬 `plan_A()` 函数的画法**，把数据源从 Dirichlet 改成读 CSV 即可。

### 2.5 写作要点

新 Fig 8 的 caption 和正文段落要明确写出来：

> The sorted-weight decay shows that target-aware aggregation behaves as **graded reweighting rather than hard selection**: the median top-1 weight (~0.24) is only ~2× the uniform baseline (0.125), and the curve crosses 1/K near rank-4. The reweighting becomes visibly sharper on error cases (FP/FN rank-1 medians ~0.28), consistent with over-concentration being one failure pattern of the model.

旧 Fig 9 里的 conformer entropy boxplot **可以删除**，Plan A 的曲线斜率已经隐式编码了熵的信息。

### 2.6 新文件

`scripts/plot_conformer_distribution.py`（**本地可跑**，不需要 GPU）

输入：`outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv`
输出：`<paper>/figures/interpretability_conformer_weights.pdf`（**直接覆盖原 Fig 8**，避免改 tex 里的 `\includegraphics{...}` 路径）。同时输出 png/svg 副本到 `outputs/interpretability/biosnap_seed42/`。

---

## 3. Figure 4 — Case study（新 must-have）

**性质**：因果证据 + 生物学锚定，需要远程小规模 forward（仅 N 个匹配样本，分钟级）。

### 3.1 想回答的问题

对**已知 PDB 共晶 + UniProt binding-site 注释**的具体 drug-target 对：
1. Mamba refinement R_t 的高峰是否**集中在已知结合位点对应的 token**？
2. 模型选出的 top-1 conformer 是否**接近 PDB 共晶 ligand pose**？

### 3.2 候选已准备好

[tmp/case_study_candidates/](tmp/case_study_candidates/) 下：

- [candidates.json](tmp/case_study_candidates/candidates.json)：11 个候选对，每条含 `drug_name, drug_smiles_raw, drug_smiles_canonical, drug_inchikey, target_uniprot, target_label, target_sequence (UniProt 全长), binding_sites (UniProt feature locations), active_sites, pdb_refs (共晶 PDB ID)`
- [build_candidates.py](tmp/case_study_candidates/build_candidates.py)：构建脚本（从 UniProt JSON 抽 binding 注释）
- [match_cases.py](tmp/case_study_candidates/match_cases.py)：匹配脚本

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

### 3.3 Step 1 — 匹配（gating，远程跑）

```bash
cd <repo_root>
/home/lsw/miniconda3/envs/ldm-dti/bin/python tmp/case_study_candidates/match_cases.py \
  --test_csv data/biosnap/random/test.csv \
  --candidates tmp/case_study_candidates/candidates.json \
  --interp_csv outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv
```

输出 `tmp/case_study_candidates/match_hits.csv`，每行：`drug, target, uniprot, row_index, label, match_type, test_protein_length, uniprot_length`。控制台同时打印每个命中的 `pred` 和 `case_type`。

#### Gating 决策

- **≥ 2 个命中且至少 1 个是 TP** → 走 Figure 4，跳到 3.4
- **< 2 个命中** → 回退到 Section 5（fallback Figure 2/3）

### 3.4 Step 2 — Mamba R_t 残基级映射（远程小规模 forward）

写 `scripts/case_study_run.py`（基于 [extract_stage2_interpretability_biosnap.py](scripts/extract_stage2_interpretability_biosnap.py) 改造）：

1. 加载 BioSNAP best checkpoint（同 extract 脚本，从 `fp32_state_dict.pt`）
2. 从 `match_hits.csv` 读匹配到的 row_index 列表
3. 仅对这些 row 跑 forward（batch_size=1 也可以，N 才十几条）
4. 复用 extract 脚本里的 `register_mamba_hook` 拿 `mamba_buffer["input"], ["output"]`
5. 算 `R_t = gate * ||h^M_t - h_t||_2`（128 维向量）
6. 对每条 case，按 1.4 公式把 `R_t` 沿 128 个 token 反向投影到原序列残基坐标：
   ```python
   residue_R = np.zeros(L_eff)
   for t in range(128):
       start = int(np.floor(t * L_eff / 128))
       end   = int(np.ceil((t + 1) * L_eff / 128))
       residue_R[start:end] = R_t[t]   # 该 token 的强度均匀填充其覆盖残基
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

### 3.5 Step 3 — Conformer vs PDB pose RMSD（本地可跑）

新文件 `scripts/case_study_conformer_rmsd.py`：

1. 对每个 case：用 RDKit 生成药物 SMILES 的 8 个 ETKDGv3 conformer（**必须用和 pre_extract 一致的 seed 和参数**，见 [scripts/pre_extract.py](scripts/pre_extract.py) 里 conformer 生成段落，约 line 200-280；UFF 能量排序）
2. 从 RCSB PDB 拉对应 ligand pose：`https://files.rcsb.org/download/<PDB>.pdb` → 抽 ligand HETATM → RDKit 重建
3. 用 `rdkit.Chem.rdMolAlign.GetBestRMS` 算 8 个 conformer 各自对 PDB ligand 的 RMSD
4. 从 interp CSV 拿到该 case 的 `conf_1..conf_8` 权重
5. 输出表：每个 conformer 的 (rank by weight, RMSD)，标出 model top-1 的 RMSD vs 8 个里的最小 RMSD

写入 `outputs/case_study/biosnap_seed42_conformer_rmsd.csv`。

> **注意**：RDKit 生成的 conformer 是 randomized（即使固定 seed，浮点也可能跨平台不同）。**严格说应该把 pre_extract 阶段缓存的 `drug_3d_features.pt` 直接读出来用**（包含已生成 conformer 的 coor）。但 pre_extract 把坐标存为 (64, 3) 池化后形式，可能丢了原子身份信息。**优先方案：直接读 cache 的 coor 和 atom feature，反推原始 conformer 用于 RMSD 计算**。Fallback：用相同 ETKDGv3 参数重新生成，承认轻微不一致。

### 3.6 Step 4 — 画 Figure 4（本地）

新文件 `scripts/plot_case_study.py`，输出 `<paper>/figures/interpretability_case_study.pdf`。

布局（2 行 × 3 列，约 14 × 7 cm）：

| | 列 1：药物结构 + 构象对比 | 列 2：蛋白序列 R_t 追踪 | 列 3：8 conformer 权重 |
|---|---|---|---|
| 行 1：Methotrexate + DHFR | 3D 叠加（RDKit top-1 vs PDB ligand），标 RMSD | x = residue 1..L，蓝线 = R_t，红色竖条 = binding sites | bar，标 "model top-1"（star）和 "PDB-closest"（diamond） |
| 行 2：另一 case | 同上 | 同上 | 同上 |

底部加 caption：富集 p-value、TP/FP class、pred prob。

### 3.7 写作要点

- 段落 1：定性观察。"In the methotrexate-DHFR case, the residues with highest Mamba refinement intensity (top-10%) include positions {…}, overlapping by N out of M with UniProt-annotated folate binding pocket (Fisher's exact p = …)."
- 段落 2：构象。"The conformer selected by the target-aware module (top-1, weight=…) deviates from the PDB co-crystal pose by RMSD=…, which ranks first/second/… among the 8 ETKDGv3-generated conformers."
- 段落 3：失败案例（如果有 FP 命中）。"In contrast, the FP case (X+Y) shows R_t peaks dispersed across non-binding regions, consistent with the lower top-1 conformer weight observed."
- 把 limitation 写进去：ProtBert 128-token 池化粒度 → 每 token 覆盖 ⌈L/128⌉ 残基，因此 residue-level 富集应该解释为 token-range-level overlap。

---

## 4. 论文改写

### 4.1 删 / 改 / 留

| 现状 | 操作 |
|---|---|
| Fig 8 conformer heatmap（[experiments.tex:389-394](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L389-L394)） | **替换** 为 Plan A 输出（同路径同 label，无需改 tex 的图引用） |
| Fig 9 module statistics 3 子图（entropy + Mamba magnitude + FiLM）（[experiments.tex:414-419](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L414-L419)） | **删除整个图**：entropy 被 Plan A 覆盖；Mamba magnitude 和 FiLM 子图作为 sanity stat 移到 appendix 或正文一句话 |
| Fig 10 Mamba refinement heatmap（[experiments.tex:423-428](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L423-L428)） | **删除**：被 Figure 4 的 R_t residue track 取代且更有说服力 |
| 小节标题 "Interpretability analysis"（[experiments.tex:377](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex#L377)） | 保留，但第一句改成 "We complement the ablation results with a **distribution-level and biology-grounded** interpretability analysis." |
| [discussion.tex:33](TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/discussion.tex#L33) "visualizations are best interpreted as diagnostic checks" | **改语气**：现在有 case study 锚定了，可以更直接说 "internal-state analysis with limited biological grounding via 2 case studies on …" |

### 4.2 新小节结构

```
\subsubsection{Interpretability analysis}

[段 1] Distribution-level conformer behavior.
  - 引用新 Fig 8（Plan A），说"graded reweighting rather than hard selection"
  - 报告 rank-1 / rank-2 / rank-4 数值
  - 报告 FP/FN vs TP/TN 形状差异

[段 2] Biological grounding via case studies on methotrexate-DHFR and <X>-<Y>.
  - 引用新 Fig 4，分两个 case 描述
  - 每个 case：R_t top-k 富集 p-value + conformer RMSD rank
  - 一个 TP + 一个对照（FP 或弱 TP），构成 narrative 张力

[段 3，可选] Module sanity statistics.
  - 一句话：FiLM 强度 / Mamba gate 数值，引 appendix
```

---

## 5. Fallback — Figure 2/3（仅当 Section 3.3 gating 失败时）

如果 case-study 至少 2 个 TP 都凑不齐，启用 v3 plan 的因果干预图：

### 5.1 Figure 2 — Conformer 反事实交换

干预设计：
- 对每条样本，记 top-1 conformer 索引为 k\*，bottom-1 为 k₋
- **Swap top↔bottom**：把 `feature[:, k*, :, :]` 和 `coor[:, k*, :, :]` 替换为第 k₋ 个 conformer 的；其余不动
- **Random swap baseline**：k\* 替换为 mask 内任意非 k\* 的 conformer，重复 3 次取均值

实现方式：write a context manager that patches the input tensors before `model.drug_encoding(...)` is called — 或者更简单：**在 batch unpack 之后、`model.forward` 之前，直接修改 `feature` 和 `coor` 张量的 K 维**。不需要改模型代码。

输出 CSV：`logit_full, logit_swap_top, logit_swap_rand_mean` per sample。
画图：violin by TP/TN/FP/FN + paired bar (top swap vs random swap)。

### 5.2 Figure 3 — Mamba 位置因果性

干预设计：
- 挑 ~100 条高置信 TP（按 `pred > 0.95 & label == 1` 筛 interp CSV）
- 对每条：
  - 跑原始 forward 拿 `R_t`（128 dim）
  - **Top-k mask**：把 `R_t` 最大的 k 个 token 在 `protein_mask` 里置 0，重跑 forward
  - **Random-k mask**：随机选 k 个非 padding token，重复 5 次取均值
  - k ∈ {2, 4, 8}

实现：单独改 `protein_mask` 张量，模型其他都不动。

输出图：line plot，x=k, y=mean |Δlogit|，两条线（top-k vs random-k）带 95% CI。

---

## 6. 工程落地

### 6.1 文件清单

```
scripts/
  plot_conformer_distribution.py        # 图 1（Plan A），本地，纯重画
  case_study_run.py                     # 图 4 step 2，远程，对匹配 case 跑 forward
  case_study_conformer_rmsd.py          # 图 4 step 3，本地，RDKit + PDB RMSD
  plot_case_study.py                    # 图 4 step 4，本地

# fallback 路径（如启用）
  run_conformer_swap.py                 # 图 2 远程
  plot_conformer_swap.py                # 图 2 本地
  run_mamba_position.py                 # 图 3 远程
  plot_mamba_position.py                # 图 3 本地

tmp/case_study_candidates/              # 已存在
  candidates.json
  build_candidates.py
  match_cases.py
  *.json                                # UniProt entries
```

### 6.2 设备/环境

- **本地（Mac, darwin）**：用 `./.venv-pptx/bin/python`（已装 matplotlib/numpy/pandas/rdkit），跑图 1、case-study 画图、RMSD 计算
- **远程（linux, GPU）**：用 `/home/lsw/miniconda3/envs/ldm-dti/bin/python`，跑 matching、Figure 4 forward、（fallback）干预 forward

### 6.3 远程计算成本估算

- Figure 1：**0**（纯本地）
- Figure 4 step 1（matching）：< 1 分钟
- Figure 4 step 2（case forward）：N × 1 forward，N < 20，秒级
- Fallback Fig 2：5,493 × 5 forward ≈ 28k forward，10-30 分钟
- Fallback Fig 3：100 × (1+18) ≈ 1,900 forward，分钟级

---

## 7. 论文图编号变更总结

| 旧 | 新 |
|---|---|
| Fig 8 (conformer heatmap, 18 cherry-pick + PowerNorm) | **Fig 8 (Plan A 分位带)** |
| Fig 9 (module statistics 3 子图) | **删除**（部分进 appendix） |
| Fig 10 (Mamba refinement heatmap) | **替换为 Fig 9 (Case study, 2 行)** |

最终 interpretability section 只剩 2 张图。

---

## 8. TODO 清单（带 gating）

### 第一阶段（无依赖，今天就能做）

- [ ] **本地**：写 `scripts/plot_conformer_distribution.py`，照搬 [tmp/interp_mocks/mock_conformer_plots.py](tmp/interp_mocks/mock_conformer_plots.py) 的 `plan_A()` 函数，数据源换为 CSV，输出覆盖 `<paper>/figures/interpretability_conformer_weights.pdf`
- [ ] **远程**：跑 Section 3.3 的 `match_cases.py`，把 `match_hits.csv` 和控制台输出回传

### 第二阶段（依赖 gating 决策）

**如果 ≥ 2 个 case 命中且包含 ≥ 1 个 TP：**
- [ ] 远程：写 + 跑 `scripts/case_study_run.py`，拿到 R_t residue-level CSV
- [ ] 本地：写 `scripts/case_study_conformer_rmsd.py`，需要先下载 PDB 文件
- [ ] 本地：写 `scripts/plot_case_study.py`，组合图

**否则（fallback）：**
- [ ] 远程：写 `src/models/hooks.py`（conformer swap + mamba mask 两个 context manager）
- [ ] 远程：写 + 跑 `run_conformer_swap.py` 和 `run_mamba_position.py`
- [ ] 本地：写两个 plot_*.py

### 第三阶段（无 gating）

- [ ] 论文：按 Section 4 改 `sec/experiments.tex` 和 `sec/discussion.tex`
- [ ] 论文：检查 `\ref{fig:...}` 和 caption 是否和新图对得上

---

## 9. 已知风险 / 待澄清

1. **SMILES encoding 差异**：BioSNAP CSV 里 SMILES 可能是 DrugBank 原始而非 RDKit canonical。`match_cases.py` 已用 InChIKey 匹配（鲁棒）。如果命中率低，可放宽到 InChIKey without stereo。
2. **Protein sequence isoform 差异**：UniProt canonical 可能和 BioSNAP 用的不一样。`match_cases.py` 已有 substring fallback。
3. **EGFR (1210 aa) 超过 ProtBert 标称 max length**：需在 case-study run 时打印 `token_features.shape[0]` 确认 cutoff，并用真实 `L_eff` 做 token-残基映射，不能盲用 UniProt L。
4. **Conformer 复现**：cache 里 `drug_3d_features.pt` 的 coor 已是 64-atom pool 后的，**可能不可逆**回到原始 atom 数。如果 RMSD 算不准，case-study 的 conformer-vs-PDB 部分要简化为"top-1 是否落在能量排序前列"+ "8 个候选最小 RMSD 是否就是 top-1"。
5. **`drug_encoding` vs `target_aware_conf_enc`**：v3 plan 里写错了属性名，实际是 `model.drug_encoding`。所有新脚本 follow 这个。
