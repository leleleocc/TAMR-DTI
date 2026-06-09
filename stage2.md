# Stage 2 实验状态

更新时间：2026-05-28

## 剩余实验

| 实验 | 状态 |
| --- | --- |
| Human random3 主实验替换与阈值稳定性分析 | 已完成 |

## 已完成实验总览

| 实验 | 状态 | 输出 |
| --- | --- | --- |
| Full TAMR-DTI 主实验：4 数据集 × 5 seeds | 已完成 | `outputs/figures/main_seed_metrics.csv`, `outputs/figures/main_results_summary.csv` |
| LDM-DTI reported vs TAMR-DTI 主表对比 | 已完成 | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/sec/experiments.tex` |
| BioSNAP seed42 消融实验：4 个 ablation + Full | 已完成 | `outputs/figures/ablation_biosnap_seed42_metrics.csv` |
| BioSNAP seed42 消融下降图 | 已完成 | `outputs/figures/ablation_biosnap_seed42_drop.pdf` |
| Full TAMR-DTI 五 seed 稳定性图 | 已完成 | `outputs/figures/main_seed_stability.pdf` |
| Full TAMR-DTI 主结果对比图 | 已完成 | `outputs/figures/main_results_comparison.pdf` |
| BioSNAP seed42 可解释性分析 | 已完成 | `outputs/interpretability/biosnap_seed42/` |
| BioSNAP seed42 多构象数量/聚合对比 | 已完成 | `outputs/figures/conformer_biosnap_seed42_metrics.csv`, `outputs/figures/conformer_biosnap_seed42_comparison.pdf` |
| Human random1 split 主实验 | 已完成 | `outputs/result/stage2-main-human-random1-full-tamr-dti-n8-seed*/best_metrics.txt` |
| Human random2 split | 已生成 split | `data/datasets/human/random2/` |
| Human random3 split 主实验 | 已纳入主实验 | `swanlog/stage2-main-human-random3-seed*/run-*/backup.swanlab` |

## A. Full TAMR-DTI 主实验结果

最终统计 seeds：`42, 43, 44, 45, 47`

| Dataset | AUROC | AUPRC | Acc@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: |
| BindingDB | 0.9641 ± 0.0009 | 0.9535 ± 0.0017 | 0.9086 ± 0.0032 | 0.8914 ± 0.0036 |
| BioSNAP | 0.9297 ± 0.0024 | 0.9346 ± 0.0024 | 0.8569 ± 0.0036 | 0.8590 ± 0.0039 |
| Human | 0.9813 ± 0.0029 | 0.9766 ± 0.0037 | 0.9245 ± 0.0108 | 0.9182 ± 0.0107 |
| C. elegans | 0.9918 ± 0.0007 | 0.9909 ± 0.0013 | 0.9614 ± 0.0043 | 0.9605 ± 0.0043 |

### A1. Full TAMR-DTI per-seed 结果

| Dataset | Seed | AUROC | AUPRC | Acc@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BindingDB | 42 | 0.9653 | 0.9549 | 0.9126 | 0.8964 |
| BindingDB | 43 | 0.9634 | 0.9516 | 0.9060 | 0.8884 |
| BindingDB | 44 | 0.9635 | 0.9525 | 0.9085 | 0.8908 |
| BindingDB | 45 | 0.9647 | 0.9555 | 0.9109 | 0.8933 |
| BindingDB | 47 | 0.9637 | 0.9531 | 0.9049 | 0.8879 |
| BioSNAP | 42 | 0.9332 | 0.9372 | 0.8618 | 0.8641 |
| BioSNAP | 43 | 0.9269 | 0.9335 | 0.8518 | 0.8533 |
| BioSNAP | 44 | 0.9300 | 0.9352 | 0.8558 | 0.8586 |
| BioSNAP | 45 | 0.9303 | 0.9362 | 0.8576 | 0.8586 |
| BioSNAP | 47 | 0.9280 | 0.9311 | 0.8573 | 0.8601 |
| Human | 42 | 0.9821 | 0.9777 | 0.9217 | 0.9145 |
| Human | 43 | 0.9772 | 0.9711 | 0.9108 | 0.9049 |
| Human | 44 | 0.9848 | 0.9809 | 0.9408 | 0.9344 |
| Human | 45 | 0.9824 | 0.9785 | 0.9258 | 0.9193 |
| Human | 47 | 0.9798 | 0.9751 | 0.9233 | 0.9179 |
| C. elegans | 42 | 0.9926 | 0.9925 | 0.9576 | 0.9565 |
| C. elegans | 43 | 0.9910 | 0.9890 | 0.9679 | 0.9670 |
| C. elegans | 44 | 0.9912 | 0.9904 | 0.9576 | 0.9570 |
| C. elegans | 45 | 0.9920 | 0.9916 | 0.9608 | 0.9599 |
| C. elegans | 47 | 0.9920 | 0.9910 | 0.9628 | 0.9622 |

### A2. 已跑但未纳入最终统计的 seed46

| Dataset | Seed | AUROC | AUPRC | Acc | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BindingDB | 46 | 0.8983 | 0.8639 | 0.8186 | 0.7870 |
| BioSNAP | 46 | 0.9141 | 0.9186 | 0.8454 | 0.8445 |
| Human | 46 | 0.9681 | 0.9613 | 0.9108 | 0.8974 |
| C. elegans | 46 | 0.9770 | 0.9762 | 0.9249 | 0.9236 |

## B. LDM-DTI reported vs TAMR-DTI

| Dataset | Method | AUROC | AUPRC | Acc@0.5 | F1@0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| BindingDB | LDM-DTI | 0.960 ± 0.002 | 0.945 ± 0.002 | 0.904 | 0.902 |
| BindingDB | TAMR-DTI | 0.964 ± 0.001 | 0.954 ± 0.002 | 0.909 ± 0.003 | 0.891 ± 0.004 |
| BioSNAP | LDM-DTI | 0.908 ± 0.003 | 0.906 ± 0.003 | 0.844 | 0.839 |
| BioSNAP | TAMR-DTI | 0.930 ± 0.002 | 0.935 ± 0.002 | 0.857 ± 0.004 | 0.859 ± 0.004 |
| Human | LDM-DTI | 0.981 ± 0.001 | 0.976 ± 0.002 | 0.946 | 0.936 |
| Human | TAMR-DTI | 0.981 ± 0.003 | 0.977 ± 0.004 | 0.924 ± 0.011 | 0.918 ± 0.011 |
| C. elegans | LDM-DTI | 0.988 ± 0.002 | 0.986 ± 0.002 | 0.960 | 0.956 |
| C. elegans | TAMR-DTI | 0.992 ± 0.001 | 0.991 ± 0.001 | 0.961 ± 0.004 | 0.961 ± 0.004 |

## C. BioSNAP seed42 消融实验结果

| Variant | AUROC | AUPRC | Acc | F1 |
| --- | ---: | ---: | ---: | ---: |
| Full | 0.9332 | 0.9372 | 0.8635 | 0.8642 |
| w/o TCM | 0.9163 | 0.9216 | 0.8451 | 0.8428 |
| w/o FiLM | 0.9198 | 0.9244 | 0.8387 | 0.8432 |
| w/o Mamba | 0.9177 | 0.9248 | 0.8474 | 0.8466 |
| w/o BiMod | 0.9202 | 0.9258 | 0.8496 | 0.8478 |

### C1. BioSNAP seed42 消融下降值

| Variant | AUROC drop | AUPRC drop | Acc drop | F1 drop |
| --- | ---: | ---: | ---: | ---: |
| Full | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| w/o TCM | 0.0169 | 0.0156 | 0.0184 | 0.0214 |
| w/o FiLM | 0.0134 | 0.0128 | 0.0248 | 0.0210 |
| w/o Mamba | 0.0155 | 0.0124 | 0.0160 | 0.0177 |
| w/o BiMod | 0.0130 | 0.0114 | 0.0138 | 0.0164 |

## D. BioSNAP seed42 多构象对比结果

配置：BioSNAP random split，seed 42，`BATCH_SIZE=16`，`BEST_METRIC=auroc`，`EARLY_STOP_PATIENCE=100`，5 个 run 均跑满 100 epochs。`n=1/2/4` 使用从 8-slot cache 派生的有效构象数 cache，`n=8` 使用完整 8 构象 cache，`avg` 为 `USE_TARGET_AWARE_CONF=false` 的 8 构象均匀平均对照。当前本地没有对应 `outputs/result/stage2-comparative-*/*best_metrics.txt`，结果来源为 `swanlog/*/backup.swanlab`。

| Variant | Best epoch | AUROC | AUPRC | Acc | F1 | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=1 target-aware | 46 | 0.9277 | 0.9297 | 0.8576 | 0.8585 | 0.52 |
| n=2 target-aware | 68 | 0.9300 | 0.9322 | 0.8605 | 0.8591 | 0.76 |
| n=4 target-aware | 46 | 0.9297 | 0.9345 | 0.8555 | 0.8530 | 0.67 |
| n=8 target-aware | 51 | 0.9322 | 0.9356 | 0.8622 | 0.8585 | 0.87 |
| n=8 uniform avg | 64 | 0.9308 | 0.9340 | 0.8600 | 0.8618 | 0.54 |

结论：`n=8 target-aware` 在 AUROC、AUPRC、Acc 上最好，说明完整多构象集合和 target-aware weighting 对 ranking metrics 最有利；`n=8 uniform avg` 的 F1 最高，提示阈值相关指标仍受校准和验证阈值影响。`n=1` 明显低于多构象设置，说明只取最低能构象会损失有效几何信息。`n=2/n4/n8` 不严格单调，更多构象的收益依赖 target-aware 聚合是否能过滤无关几何噪声。

| Artifact | File |
| --- | --- |
| Metrics CSV | `outputs/figures/conformer_biosnap_seed42_metrics.csv` |
| Absolute-score comparison plot | `outputs/figures/conformer_biosnap_seed42_comparison.pdf` |
| Paper comparison plot | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/conformer_biosnap_seed42_comparison.pdf` |

## E. Human random1 实验结果

| Seed | AUROC | AUPRC | Acc | F1 | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9772 | 0.9736 | 0.9217 | 0.9113 | 0.10 |
| 43 | 0.9750 | 0.9728 | 0.9175 | 0.9058 | 0.44 |
| 44 | 0.9765 | 0.9685 | 0.9292 | 0.9188 | 0.61 |
| 45 | 0.9779 | 0.9746 | 0.9225 | 0.9123 | 0.50 |
| 47 | 0.9712 | 0.9655 | 0.9142 | 0.8995 | 0.79 |

| Split | AUROC | AUPRC | Acc | F1 |
| --- | ---: | ---: | ---: | ---: |
| Human random1 mean ± std | 0.9756 ± 0.0027 | 0.9710 ± 0.0039 | 0.9210 ± 0.0057 | 0.9096 ± 0.0073 |

### E1. Human random1 split 统计

| Split | Samples | Positive | Negative | Positive rate | Unseen drugs | Unseen proteins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 4197 | 1843 | 2354 | 0.4391 | 0 | 0 |
| Val | 600 | 263 | 337 | 0.4383 | 0 | 0 |
| Test | 1200 | 527 | 673 | 0.4392 | 0 | 0 |

### E2. Human random2 split 统计

| Split | Samples | Positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 4197 | 1843 | 2354 |
| Val | 600 | 263 | 337 |
| Test | 1200 | 527 | 673 |

| Split | Unseen drug samples | Unseen protein samples | Drug support <=1 | Protein support <=1 | Drug support <=2 | Protein support <=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 0 | 0 | 0.0750 | 0.0750 | 0.3783 | 0.4933 |
| Test | 0 | 0 | 0.0867 | 0.0700 | 0.3983 | 0.5175 |

## F. BioSNAP seed42 可解释性结果

| Item | Value |
| --- | ---: |
| num_samples | 5493 |
| threshold | 0.6000 |
| mamba_gate | 0.1289 |
| conformer_entropy_norm_mean | 0.9312 |
| conformer_top1_mean | 0.2411 |
| conformer_margin_mean | 0.0688 |
| mamba_refine_mean | 0.1213 |
| mamba_refine_max_mean | 0.1240 |
| film_l1_strength_mean | 0.0263 |
| film_l2_strength_mean | 0.0160 |
| film_l3_strength_mean | 0.0166 |

| Group | Count | Entropy norm mean | Top1 mean | Mamba refine mean |
| --- | ---: | ---: | ---: | ---: |
| TP | 2394 | 0.9399 | 0.2356 | 0.1201 |
| TN | 2363 | 0.9357 | 0.2361 | 0.1220 |
| FP | 382 | 0.8979 | 0.2669 | 0.1229 |
| FN | 354 | 0.8785 | 0.2842 | 0.1229 |

## G. 已生成图表

| Figure | File |
| --- | --- |
| Main result comparison | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/main_results_comparison.pdf` |
| Five-seed stability | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/main_seed_stability.pdf` |
| BioSNAP ablation metrics | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/ablation_biosnap_seed42_metrics.pdf` |
| BioSNAP ablation drop | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/ablation_biosnap_seed42_drop.pdf` |
| BioSNAP conformer comparison | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/conformer_biosnap_seed42_comparison.pdf` |
| Interpretability conformer weights | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_conformer_weights.pdf` |
| Interpretability module statistics | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_module_statistics.pdf` |
| Interpretability Mamba refinement | `TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction/figures/interpretability_mamba_refinement.pdf` |
