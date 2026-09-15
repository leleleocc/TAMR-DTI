# LDM-DTI 改进实验上下文（精简版）

更新口径：2026-05-13

## 1. 使用约定

- 本文件是当前实验判断的主记录。
- `docs/experiment_records.md` 是旧版记录，不作为当前决策依据。
- 正式实验优先采用 `formal-*` 命名的 SwanLab / output 结果。
- 探索 run 只用于方向判断，不直接进入论文主结果表，除非后续按正式口径重跑。

## 2. 当前任务定位

基于原论文 `LDM-DTI.pdf` 做方法改进与复现实验。目标是在保持 LDM-DTI 主体框架可比较的前提下，验证新编码模块和融合模块是否能达到或超过原论文及本地 baseline。

原论文结构：

- Drug 1D: ChemBERTa
- Protein sequence: ProtBERT
- Drug 2D: GCN
- Drug 3D: EGNN
- Protein local context: DCNN + self-attention
- Fusion: DIAM / BiIntention
- Classifier: MLP

原论文 BindingDB 结果：

| Model | AUROC | AUPRC | Acc | F1 |
| --- | ---: | ---: | ---: | ---: |
| LDM-DTI | 0.960 ± 0.002 | 0.945 ± 0.002 | 0.904 | 0.902 |

本地历史 baseline `swanlog/v0` 基本复现原论文，但正式比较以 E0 rerun 为准。

## 3. 当前方法定义

### 第一创新点：药物-蛋白互相调制的构象编码

代码位置：

```text
src/models/models.py
TargetAwareConformerEncoder
ProteinACmix
LDMDTI.forward
```

机制：

- 蛋白质上下文参与药物构象选择，得到 target-aware drug 3D representation。
- 药物融合表示通过 FiLM 调制蛋白质残基编码，得到 ligand-conditioned protein representation。

推荐表述：

```text
Bidirectional drug-target modulation with conformation-aware molecular encoding.
```

当前判断：C1 结果支持“互相调制编码”有潜力，但 n 构象数量本身并非越多越好，多构象叙事需要谨慎。

### 第二创新点候选状态

已测试的第二创新点候选包括：

- MambaFusion 完全替换 BiIntention；
- Mamba residual + BiIntention；
- FieldEnhancedBiIntention；
- Protein-only Mamba residual before BiIntention；
- Protein-Mamba direct dynamic gate before BiIntention；
- 当前 CF regularization。

当前结论：

- `MambaFusion`、`MambaEnhancedBiIntention`、`FieldEnhancedBiIntention`、当前 `CF` 都不成立；
- `M1-g1-n1` 低于 `M1-g2-n1`，说明固定 gate 继续增大到 `-1` 会扰动泛化；
- `M1-dg-n1` 低于固定 gate 版本，直接动态 gate 暂不作为主线；
- `protein_mamba_biintention` 已经超过 E0 / E1 / C1，是当前最有希望的第二创新点方向。

## 4. 正式实验总表

| Run | 主要设置 | best epoch | Test AUROC | Test AUPRC | Test Acc | Test F1 | 关键结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v0 | 历史本地 baseline，BiIntention，无特殊编码 | - | 0.9576 | 0.9436 | 0.8992 | - | 历史参考，不再作为唯一正式 baseline |
| E0 | v0 formal rerun，BiIntention，无特殊编码 | 69 | 0.9619 | 0.9494 | 0.9079 | 0.8890 | 当前正式 baseline |
| E1 | 双向调制编码，n=8，BiIntention | 66 | 0.9615 | 0.9502 | 0.9074 | 0.8889 | 相比 E0 仅 AUPRC 微升，整体基本持平 |
| E1a | E1 no-CF | 66 | 0.9615 | 0.9502 | 0.9074 | 0.8889 | 与 E1 完全一致，说明当前 BiIntention 下 CF 未生效 |
| E2 | 双向调制编码，MambaFusion 替换 BiIntention | 42 | 0.9422 | 0.9207 | 0.8730 | 0.8425 | 明显下降，纯 MambaFusion 不适合作主线 |
| E2a | E2 稳定训练版，低 LR，best=AUROC，长 patience | 91 | 0.9420 | 0.9202 | 0.8757 | 0.8504 | 未改善，排除学习率/早停为主因 |
| E3 | 双向调制编码，FieldEnhancedBiIntention | 53 | 0.9573 | 0.9444 | 0.8972 | 0.8766 | 低于 E0/E1，不作为主线 |
| E4 | 双向调制编码，MambaEnhancedBiIntention | 41 | 0.9521 | 0.9357 | 0.8892 | 0.8627 | 低于 E0/E1/E3，Mamba residual 不成立 |
| M1-g2-n8 | 双向调制编码，n=8，protein-Mamba + BiIntention，gate=-2 | 69 | 0.9664 | 0.9554 | 0.9110 | 0.8929 | 当前 AUROC/AUPRC 最好正式结果 |
| M1-g2-n1 | 双向调制编码，n=1，protein-Mamba + BiIntention，gate=-2 | 77 | 0.9640 | 0.9533 | 0.9129 | 0.8938 | 当前 Acc/F1 最好正式结果 |
| M1-g1-n1 | 双向调制编码，n=1，protein-Mamba + BiIntention，gate=-1 | 61 | 0.9608 | 0.9483 | 0.9058 | 0.8876 | 低于 M1-g2-n1/C1/E0，gate 过大不成立 |
| M1-dg-n1 | 双向调制编码，n=1，protein-Mamba + direct dynamic gate + BiIntention | 52 | 0.9632 | 0.9523 | 0.9059 | 0.8844 | 低于 M1-g2-n1/C1，直接动态 gate 暂不成立 |
| C1 | 双向调制编码，n=1，BiIntention | 69 | 0.9640 | 0.9517 | 0.9100 | 0.8922 | 当前最好纯 BiIntention 结果，超过 E0 |
| C2 | 双向调制编码，n=2，BiIntention | 61 | 0.9625 | 0.9518 | 0.9083 | 0.8888 | 接近 C1，AUPRC 略高于 C1 |
| C3 | 双向调制编码，n=4，BiIntention | 54 | 0.9623 | 0.9493 | 0.9066 | 0.8871 | 低于 C1/C2 |
| C4 | 双向调制编码，n=8，BiIntention | 41* | 0.9589* | 0.9470* | 0.9020* | 0.8832* | C4 未正常写出 `best_metrics.txt`，该结果来自 swanlog，暂作 provisional |

注：C4 日志只到 epoch 46，`early_stop=5/10`，没有正式 `best_metrics.txt`，不能作为最终 n=8 结论。E1 是当前可引用的正式 n=8 主模型结果。

## 5. 每个实验的关键结论

### v0：历史 baseline

```text
test_acc  = 0.8992
test_auc  = 0.9576
test_aupr = 0.9436
```

结论：基本复现原论文 BindingDB 表现，但后续正式比较以 E0 rerun 为准。

### E0：正式 baseline rerun

```text
best_epoch = 69
test_acc   = 0.9079
test_auroc = 0.9619
test_auprc = 0.9494
test_f1    = 0.8890
```

结论：当前最重要 baseline。E1/E2/E3/E4 都必须优先和 E0 对比。

### E1：双向调制编码 + BiIntention

```text
best_epoch = 66
test_acc   = 0.9074
test_auroc = 0.9615
test_auprc = 0.9502
test_f1    = 0.8889
```

结论：相比 E0，AUPRC +0.0008，但 Acc/AUROC/F1 略低。第一创新点不能只靠 E1 宣称成立，需要 C1-C3 和拆分消融补证。

### E1a：E1 no-CF

```text
best_epoch = 66
test_acc   = 0.9074
test_auroc = 0.9615
test_auprc = 0.9502
test_f1    = 0.8889
```

结论：E1a 与 E1 指标完全一致，best checkpoint 也一致。BiIntention 当前返回 `att=None`，导致 CF loss 为 0；当前 CF 不能作为第二创新点。

### E2：MambaFusion 替换 BiIntention

```text
best_epoch = 42
test_acc   = 0.8730
test_auroc = 0.9422
test_auprc = 0.9207
test_f1    = 0.8425
```

结论：明显低于 E0/E1。直接把 drug/protein tokens 拼成序列并用 Mamba 替换 BiIntention 不适合当前 DTI 跨模态对齐任务。

### E2a：MambaFusion 训练策略排查

```text
best_epoch = 91
test_acc   = 0.8757
test_auroc = 0.9420
test_auprc = 0.9202
test_f1    = 0.8504
```

结论：降低 LR、改用 AUROC 选 best、拉长 patience 后仍无改善。MambaFusion 下降主要来自架构不匹配，而非简单训练策略问题。

### E3：FieldEnhancedBiIntention

```text
best_epoch = 53
test_acc   = 0.8972
test_auroc = 0.9573
test_auprc = 0.9444
test_f1    = 0.8766
```

结论：低于 E0/E1。显式 interaction field residual 没有形成稳定收益，只能作为融合对照或负结果。

### E4：MambaEnhancedBiIntention

```text
best_epoch = 41
test_acc   = 0.8892
test_auroc = 0.9521
test_auprc = 0.9357
test_f1    = 0.8627
```

结论：低于 E0/E1/E3。当前 pooled Mamba residual 设计不成立，不能作为第二创新点。

### M1-g2-n8：protein-Mamba + BiIntention，n=8

```text
best_epoch = 69
test_acc   = 0.9110
test_auroc = 0.9664
test_auprc = 0.9554
test_f1    = 0.8929
```

结论：显著超过 E0 / E1 / C1，是当前 AUROC/AUPRC 最好的正式结果。说明 Mamba 更适合作为 protein sequence enhancer，而不是直接替换跨模态 BiIntention。

### M1-g2-n1：protein-Mamba + BiIntention，n=1

```text
best_epoch = 77
test_acc   = 0.9129
test_auroc = 0.9640
test_auprc = 0.9533
test_f1    = 0.8938
```

结论：相比 C1 继续提升 Acc/F1/AUPRC，是当前 Acc/F1 最好的正式结果。说明在单构象底座上，protein-Mamba 同样有效。

### M1-g1-n1：protein-Mamba + BiIntention，n=1，gate=-1

```text
best_epoch = 61
test_acc   = 0.9058
test_auroc = 0.9608
test_auprc = 0.9483
test_f1    = 0.8876
```

结论：低于 M1-g2-n1、C1 和 E0。学到的 gate sigmoid 约为 0.2707，明显高于 M1-g2 的约 0.14；说明 protein-Mamba 注入比例过大时会扰动 BiIntention 的稳定对齐，当前最佳固定 gate 仍是 `-2` 附近。

### M1-dg-n1：protein-Mamba + direct dynamic gate，n=1

```text
best_epoch = 52
test_acc   = 0.9059
test_auroc = 0.9632
test_auprc = 0.9523
test_f1    = 0.8844
```

结论：低于 M1-g2-n1，也低于 C1 的 Acc/F1。直接用 `[drug_ctx, protein_ctx, drug_ctx * protein_ctx]` 预测样本级 gate 没有带来收益，当前不作为主线；如继续探索 gate，应优先限制动态偏移并 detach gate 输入。

### C1-C4：纯 BiIntention 的 n 构象对比

| Run | n | Test AUROC | Test AUPRC | Test Acc | Test F1 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 1 | 0.9640 | 0.9517 | 0.9100 | 0.8922 | 当前最好纯 BiIntention 结果 |
| C2 | 2 | 0.9625 | 0.9518 | 0.9083 | 0.8888 | 接近 C1 |
| C3 | 4 | 0.9623 | 0.9493 | 0.9066 | 0.8871 | 低于 C1/C2 |
| C4 | 8 | 0.9589* | 0.9470* | 0.9020* | 0.8832* | 未正常收尾，仅作临时参考 |

结论：

- n 构象数量不是越多越好。
- C1 说明双向调制编码在单构象下已经能超过 E0。
- 纯 BiIntention 下，`n=1` 更偏 `Acc/F1/AUROC`，`n=2` 的 `AUPRC` 略高；更大 n 没有继续提升。
- 加入 protein-Mamba 后，`n=8` 当前更偏 `AUROC/AUPRC`，`n=1` 当前更偏 `Acc/F1`。
- 论文中不宜直接声称“更多构象带来稳定收益”；应把主线表述为 drug-target modulation / conformation-aware encoding，而不是单纯 multi-conformer 数量优势。

## 6. 当前总判断

1. 当前正式 baseline 是 E0。
2. 当前最好正式结果分成两类：
   `M1-g2-n8` 是 AUROC/AUPRC 最强；
   `M1-g2-n1` 是 Acc/F1 最强。
3. 第一创新点有希望，但应依赖 C1-C3 和后续拆分消融，而不是只引用 E1。
4. 第二创新点当前最有希望的正式方向是 `protein_mamba_biintention`。
5. 纯 MambaFusion、pooled Mamba residual、FieldEnhancedBiIntention、过大固定 gate、直接动态 gate 均不成立。
6. 当前 CF regularization 在 BiIntention 下未生效，不能作为第二创新点。

## 7. 论文结果口径

主结果表建议优先包含：

- Original LDM-DTI paper；
- Local historical baseline v0；
- E0 formal baseline；
- C1 当前最好纯 BiIntention 模型；
- M1-g2-n8 当前最好 AUROC/AUPRC 模型；
- M1-g2-n1 当前最好 Acc/F1 模型；
- E1 n=8 双向调制主模型；
- E2/E2a/E3/E4 作为融合模块负结果或消融。

写作注意：

- E1 相比 E0 不是全面提升，只能说 AUPRC 微升、整体持平。
- C1 相比 E0 有更清楚的提升，是第一创新点在纯 BiIntention 下的强结果。
- M1-g2-n8 和 M1-g2-n1 说明 protein-Mamba 可以作为第二创新点主线，其中不同 n 对不同指标偏好不同。
- E2/E2a/E4 可以用于说明直接 Mamba 替换或 pooled residual Mamba 不适合当前任务。
- E3 可以用于说明显式 field residual 没有稳定带来泛化收益。

## 8. 待补实验

当前优先待跑：

| Run | 配置 | 目的 | 状态 |
| --- | --- | --- | --- |
| M1-dg-lite-n1 | `configs/experiments/formal_m1dg_lite_n1_bimod_protein_mamba_direct_gate_bi_bindingdb_random.yaml` | direct dynamic gate 的受限版，detach gate 输入并限制 logit 偏移 | 已生成配置，但优先级不高 |

必要补证：

- `w/o target-aware conformer selection`
- `w/o ligand-conditioned protein FiLM`
- C4 若要进入正式表，需要正常收尾或重跑并生成 `best_metrics.txt`
- `protein_mamba_biintention` 建议至少补一个新 seed

多 seed：

- 主模型确定后再做 3-5 个随机种子，报告 mean ± std。

## 9. Overleaf

本地论文 / Overleaf 项目目录：

```text
/home/lsw/lv/LDM-DTI/DTI
```

常用命令：

```bash
cd /home/lsw/lv/LDM-DTI/DTI
latexmk -pdf -interaction=nonstopmode -halt-on-error ldm_dti.tex
OLCLI_TIMEOUT_MS=120000 olcli pull
OLCLI_TIMEOUT_MS=120000 olcli push
```

当前论文状态：

- `DTI/ldm_dti.tex` 已补充实验图表、诊断表、剩余验证计划、实现细节和 run registry。
- 本地 `latexmk` 编译通过，输出 `DTI/ldm_dti.pdf`，共 12 页。
- 本地日志未发现 overfull、未定义引用或 LaTeX error；剩余为 underfull 排版提示。
- 旧 ICLR 模板名前缀已统一改为正式短名 `ldm_dti.*`；Overleaf compile/output 接口此前会超时，本地编译结果为准。
