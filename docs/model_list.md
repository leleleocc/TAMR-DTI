1. 预训练语言模型 (Pretrained Language Models)
   主要用于提取分子和蛋白质的语义级特征（作为外部特征输入到网络中）：

ChemBERTa: 用于提取药物（SMILES）的预训练语义特征。
ProtBERT: 用于提取靶点蛋白质（氨基酸序列）的预训练语义特征。 2. 药物特征提取模型 (Drug Representation)
用于从图结构和空间结构中提取药物特征：

GCN (Graph Convolutional Network): 在 MolecularGCN 模块中使用（调用了 dgllife 库），用于提取药物的 2D 分子拓扑图特征。
EGNN (Equivariant Graph Neural Network): 等变图神经网络，用于处理药物分子的 3D 几何坐标（coor），提取空间结构特征。 3. 蛋白质特征提取模型 (Protein Representation)
用于从一维氨基酸序列中提取局部和全局特征：

ACmix (CNN + Transformer 混合架构): 在 ProteinACmix 模块中实现。它结合了卷积神经网络（提取局部特征）和 Transformer 编码器（提取全局依赖关系）。 4. 特征融合模型 (Feature Fusion)
用于将药物特征和蛋白质特征进行深度交互：

Self-Attention (自注意力机制): 在交叉融合前，对药物和蛋白质各自的特征进行内部增强。
Bi-Cross-Attention (双向交叉注意力网络): 在 DWFusion.py 的 BiIntention 模块中实现。药物和蛋白质特征互为 Query、Key、Value 进行双向注意力计算，实现多模态特征的深度对齐与融合。 5. 分类解码器 (Classifier / Decoder)
用于输出最终的预测结果：

MLP (Multi-Layer Perceptron): 在 MLPDecoder 模块中实现，由多层全连接层、BatchNorm 和 ReLU 激活函数组成，用于最终的药物-靶点相互作用（DTI）二分类打分。
