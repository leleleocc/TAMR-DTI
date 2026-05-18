  一、主实验

1. 四个数据集 AUROC / AUPRC 对比
表 3、表 4。和 SVM、RF、KNN、LR、GraphDTA、DrugBAN、IIFDTI、TransformerCPI、MolTrans、BINDTI 对比。


| 数据集       | AUROC         | AUPRC         | Acc   | F1    |
| --------- | ------------- | ------------- | ----- | ----- |
| BindingDB | 0.960 ± 0.002 | 0.945 ± 0.002 | 0.904 | 0.902 |
| BioSNAP   | 0.908 ± 0.003 | 0.906 ± 0.003 | 0.844 | 0.839 |
| Human     | 0.981 ± 0.001 | 0.976 ± 0.002 | 0.946 | 0.936 |
| C.elegans | 0.988 ± 0.002 | 0.986 ± 0.002 | 0.960 | 0.956 |


1. 五折随机实验的 Accuracy 箱线图
Fig. 2：四个子图分别是 BindingDB、BioSNAP、Human、C.elegans。
结论是 LDM-DTI 在四个数据集上都有最高或接近最高的 accuracy median，并且箱体较窄，作者用来说明性能稳定、
方差小、泛化能力强。
2. 统计显著性检验
Fig. 3 和 Table 5。
Fig. 3 是四个数据集上的五次随机实验统计检验热力图；Table 5 是 BindingDB 上 LDM-DTI 相比各 baseline 的p-value。


| Baseline       | P-value(AUPRC) | P-value(Acc) | P-value(F1) |
| -------------- | -------------- | ------------ | ----------- |
| SVM            | 1.39E-08       | 3.92E-08     | 2.07E-09    |
| RF             | 4.14E-06       | 8.90E-06     | 1.28E-07    |
| KNN            | 1.68E-07       | 2.49E-08     | 5.65E-08    |
| LR             | 1.07E-06       | 3.06E-06     | 1.51E-06    |
| GraphDTA       | 2.51E-04       | 2.05E-05     | 6.13E-03    |
| DrugBAN        | 4.05E-01       | 6.69E-02     | 1.15E-02    |
| IIFDTI         | 1.38E-06       | 2.32E-07     | 8.21E-10    |
| TransformerCPI | 1.22E-07       | 3.64E-06     | 1.39E-07    |
| MolTrans       | 5.21E-05       | 1.65E-03     | 7.26E-07    |
| BINDTI         | 4.15E-02       | 7.03E-03     | 3.58E-01    |


  注意：DrugBAN 的 AUPRC/Acc 和 BINDTI 的 F1 不显著或不强显著，原文说“majority of cases”显著，不是所有项都显
  著。

  CI 是“我这个方法重复跑 5 次，平均性能大概落在哪个区间”。
  p-value 是“两个方法重复跑 5 次，差距是不是大到不像随机波动”。

1. 95% Confidence interval
Table 6，BindingDB 上 AUPRC、Acc、F1 的置信区间。LDM-DTI 行是：


| Method  | AUPRC Lower | AUPRC Upper | Acc Lower | Acc Upper | F1 Lower | F1 Upper |
| ------- | ----------- | ----------- | --------- | --------- | -------- | -------- |
| LDM-DTI | 94.41       | 94.63       | 90.14     | 90.74     | 90.09    | 94.20    |


  这部分结论是：LDM-DTI 的 AUPRC 和 Acc 置信区间整体较高，作者据此强调稳定性和可靠性。不过 F1 上 DrugBAN、
  MolTrans、BINDTI 的上界也很高，所以写作时不能说所有 CI 全面碾压，只能按原文谨慎说“整体上更稳定、更优”。

  二、消融实验

  消融实验只在 BioSNAP 上做，Fig. 4 展示 ROC / PR 曲线，比较完整模型和四个去模块版本：


| Variant      | AUROC | AUPRC |
| ------------ | ----- | ----- |
| Full LDM-DTI | 0.908 | 0.906 |
| Without GCN  | 0.765 | 0.714 |
| Without EGNN | 0.796 | 0.735 |
| Without DCNN | 0.807 | 0.757 |
| Without DIAM | 0.859 | 0.834 |


  所以原论文的“主实验”应包括：四数据集 AUROC/AUPRC 表、五折 Accuracy 箱线图、统计显著性 p-value、95% CI。消融实验单独是 BioSNAP 上的 GCN / EGNN / DCNN / DIAM 去除实验。