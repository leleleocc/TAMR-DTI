from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
# 2D DGL 图节点输入维度；dataloader 中原子特征 74 维 + virtual node 标记 1 维。
_C.DRUG.NODE_IN_FEATS = 75

# 是否让 MolecularGCN 的输入线性层忽略 virtual node 标记位；开启后 padding 节点不直接贡献原子语义。
_C.DRUG.PADDING = True

# 2D GCN 每层 hidden 维度；层数越多/维度越大，2D 拓扑分支容量越强，也更容易过拟合。
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
# 2D 原子特征进入 GCN 前投影到的 embedding 维度，需要和后续融合维度保持一致。
_C.DRUG.NODE_IN_EMBEDDING = 128
# 每个分子图补齐到的最大节点数；当前 dataloader 默认也是 290，超过会报错。
_C.DRUG.MAX_NODES = 290
# 每个 SMILES 缓存/使用的 3D 构象数；越大几何信息更丰富，但缓存、训练显存和 EGNN 计算开销线性增加。
_C.DRUG.NUM_CONFORMERS = 8
# 每个构象池化后的 3D 原子 token 数；和 drug3d cache shape 以及 2D+3D 拼接长度相关。
_C.DRUG.CONFORMER_ATOMS = 64
# 是否用蛋白表示动态选择多构象；关闭后构象权重退化为非靶标感知的均匀/默认选择。
_C.DRUG.USE_TARGET_AWARE_CONF = True
# 1D ChemBERTa 分支与 2D+3D 几何分支的融合门初始 bias。
# sigmoid(bias) 是初始 1D 权重；-2.0≈12% 1D、88% 2D/3D，用于加强几何分支。
_C.DRUG.FUSION_GATE_BIAS = -2.0
# Drug-3D 原子特征变体（仅决定离线 cache 文件名以及 atom_features 内容）：
# - "vanilla": 原 74 维原子特征（与 conformer 无关，feature[k] 跨 K 完全相同）
# - "geo_v1":  原 74 维 + 10 维 SE(3)-invariant 全分子几何描述符
#              (per-atom d_min/mean/std/max + 6 维 RBF 求和), broadcast 到全部 atom；
#              使 feature[k] 沿 K 维不同，但 feature[k, i] 在 geo block 上跨 i 常数。
# - "geo_v2":  原 74 维 + 12 维 SE(3)-invariant per-atom 局部几何描述符
#              (dist_to_COM, densities, nn1+RBFs, triplet angles, planarity, gyration),
#              使 feature[k, i] 沿 K 和 i 两个维度都真正不同。
_C.DRUG.DRUG3D_FEATURE_VARIANT = "vanilla"
# TargetAwareConformerEncoder.score MLP 的输入构成模式：
# - "with_protein": cat([conf_global, protein_global(broadcast), energy_emb]) -- 旧行为；
#                   protein_global 沿 K 方差为 0，零方差子空间是塌陷的结构性放大器。
# - "no_protein":   cat([conf_global, energy_emb]) -- 去掉零方差子空间。
# - "protein_ctx":  cat([conf_global, protein_ctx(per-K), energy_emb])
#                   protein_ctx[k] 由轻量 cross-attn(query=conf_global[k], key/value=protein_tokens) 得到，
#                   让蛋白上下文随 conformer 变化。
_C.DRUG.CONF_SCORE_INPUT_MODE = "with_protein"


# Protein feature extractor
_C.PROTEIN = CN()
# 旧 CNN 配置，当前 LDMDTI 未直接读取；保留用于兼容旧实验配置。
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
# 旧多尺度卷积核配置，当前 ProteinACmix 构造处固定 conv_kernel_size=3，未直接读取。
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
# ProtBERT 1024 维投影后的蛋白 token 维度；必须和 CROSSINTENTION.EMBEDDING_DIM 对齐。
_C.PROTEIN.EMBEDDING_DIM = 128
# 旧蛋白注意力头数配置，当前 ProteinACmix 构造处固定 num_attention_heads=8，未直接读取。
_C.PROTEIN.NUM_HEAD = 8
# 蛋白序列离线池化后的 token 数；决定交互场蛋白侧长度和计算量。
_C.PROTEIN.MAX_LEN = 128
# 旧 padding 开关，当前蛋白有效位主要由 protein_mask 控制。
_C.PROTEIN.PADDING = True
# 是否使用药物全局向量对蛋白编码器做 FiLM 调制；开启后蛋白特征会变成 ligand-conditioned。
_C.PROTEIN.USE_LIGAND_FILM = True
# FiLM 调制强度；越大药物对蛋白表征影响越强，过大可能扰乱 ProtBERT 通用表示。
_C.PROTEIN.FILM_SCALE = 0.05

# CrossIntention feature fusion
_C.CROSSINTENTION = CN()
# 交互层数；BiIntention 中生效，InteractionFieldFusion 当前实现不使用该层数。
_C.CROSSINTENTION.LAYER = 1
# 交互/自注意力头数；影响药物-token 与蛋白-token 交互建模容量。
_C.CROSSINTENTION.NUM_HEAD = 8
# 融合模块内部 embedding 维度；需与 DRUG/PROTEIN 投影后的 128 维一致。
_C.CROSSINTENTION.EMBEDDING_DIM = 128
# 融合模块模式:
# - biintention: 原论文双向 intention 主干，性能稳定但没有显式 interaction field。
# - interaction_field: 直接用显式 drug-token x protein-token field 替换主干；已验证随机划分上性能下降。
# - field_enhanced_bi: 保留 BiIntention 主干，用 interaction field 做小权重残差增强，作为融合对照。
# - mamba: 完全用 Mamba 序列融合替换 BiIntention，用于评估纯状态空间主干效果。
# - mamba_enhanced_bi: 保留 BiIntention 主干，用 Mamba 做小权重残差增强。
# - protein_mamba_biintention: 仅用 Mamba 增强 protein tokens，再进入稳定 BiIntention 主干。
# - protein_mamba_direct_gate_biintention: 用 drug/protein 全局匹配信息动态控制 protein-Mamba 注入强度。
_C.CROSSINTENTION.FUSION_MODE = "biintention"
# 旧布尔开关，仅在 FUSION_MODE 为空/旧配置时用于兼容；新实验优先使用 FUSION_MODE。
_C.CROSSINTENTION.USE_INTERACTION_FIELD = False
# FieldEnhancedBiIntention 中 field 残差门初始 bias；sigmoid(-2.0)≈0.12，保证早期主要依赖 BiIntention。
_C.CROSSINTENTION.FIELD_ENHANCE_GATE_BIAS = -2.0
# 纯 Mamba 融合主干配置；FUSION_MODE='mamba' 时生效。
_C.CROSSINTENTION.MAMBA_D_STATE = 16
_C.CROSSINTENTION.MAMBA_D_CONV = 4
_C.CROSSINTENTION.MAMBA_EXPAND = 2
_C.CROSSINTENTION.MAMBA_BIDIRECTIONAL = True
_C.CROSSINTENTION.MAMBA_DROPOUT = 0.1
# Mamba 残差门初始 bias；mamba_enhanced_bi / protein_mamba_biintention 中生效。
# direct gate 的 MLP 最后一层零初始化，因此初始 gate 仍等价于 sigmoid(MAMBA_ENHANCE_GATE_BIAS)。
_C.CROSSINTENTION.MAMBA_ENHANCE_GATE_BIAS = -2.0
# protein_mamba_direct_gate_biintention 的 gate MLP 隐层维度。
_C.CROSSINTENTION.MAMBA_GATE_HIDDEN_DIM = 128
# direct gate 动态偏移强度；配合 MAMBA_GATE_BOUNDED=True 时表示 logit 偏移上限。
_C.CROSSINTENTION.MAMBA_GATE_DELTA_SCALE = 1.0
# 是否阻断 gate 输入对 drug/protein 编码器的反向梯度。
_C.CROSSINTENTION.MAMBA_GATE_DETACH_CONTEXT = False
# 是否将 gate MLP 输出限制为 delta_scale * tanh(delta)，避免动态 gate 大幅漂移。
_C.CROSSINTENTION.MAMBA_GATE_BOUNDED = False
# 当前 mamba-ssm 2.2.2 fast path 直接调用旧 causal_conv1d_cuda 签名；配合 causal-conv1d 1.6.1 时需关闭。
_C.CROSSINTENTION.MAMBA_USE_FAST_PATH = False
# 当前机器 GPU 低于 sm70 时不能跑 Mamba2 Triton backward；默认用 Mamba1 CUDA 扩展路径。
_C.CROSSINTENTION.MAMBA_BACKEND = "mamba1"
# 第几轮开始加入反事实一致性损失；越早正则越强，可能压制主任务收敛。
_C.CROSSINTENTION.CF_START_EPOCH = 30
# 反事实遮蔽时选取 top 关键蛋白 token 的比例；越大遮蔽区域越宽，约束更强。
_C.CROSSINTENTION.CF_TOP_RATIO = 0.15
# 关键区域遮蔽后的降分 margin；越大要求关键区域被遮蔽后分数下降越明显。
_C.CROSSINTENTION.CF_MARGIN = 0.1
# 关键区域遮蔽降分损失权重；过大可能牺牲分类 BCE 主目标。
_C.CROSSINTENTION.CF_KEY_WEIGHT = 0.02
# 非关键区域遮蔽稳定性损失权重；过大可能让模型过度保守。
_C.CROSSINTENTION.CF_STABLE_WEIGHT = 0.01
# 交互场熵正则权重；越大越鼓励交互分布平滑，过大可能削弱关键位点聚焦。
_C.CROSSINTENTION.FIELD_ENTROPY_WEIGHT = 0.0

# MLP decoder
_C.DECODER = CN()
# 解码器名称标记；当前代码固定使用 MLPDecoder，主要用于记录配置。
_C.DECODER.NAME = "MLP"
# 融合后的输入维度；当前为 drug_pool 128 + protein_pool 128。
_C.DECODER.IN_DIM = 256
# MLP 分类头 hidden 维度；越大分类头容量越强，也更容易过拟合。
_C.DECODER.HIDDEN_DIM = 256
# MLP 分类头倒数第二层输出维度；影响最终 logits 前的压缩表示。
_C.DECODER.OUT_DIM = 128
# 二分类输出维度；BCEWithLogitsLoss 需要 1 个 logit。
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
# 最大训练轮数；实际可能被 early stopping 提前终止。
_C.SOLVER.MAX_EPOCH = 100
# 每张 GPU 的 micro batch size；有效 batch = BATCH_SIZE * GPU数 * gradient_accumulation_steps。
_C.SOLVER.BATCH_SIZE = 16
# DataLoader worker 数；过大占 CPU/内存，过小可能喂不满 GPU。
_C.SOLVER.NUM_WORKERS = 8
# 初始学习率；v1 新增模块较多时偏大可能导致验证集波动。
_C.SOLVER.LR = 1e-4
# 旧开关，当前训练主流程未直接读取；保留用于兼容旧配置。
_C.SOLVER.USE_LD = True
# 学习率衰减倍数；plateau 调度触发时 lr *= LR_DECAY。
_C.SOLVER.LR_DECAY = 0.5
# 旧固定间隔衰减参数；当前 LR_SCHEDULER='plateau' 时未直接使用。
_C.SOLVER.DECAY_INTERVAL = 25
# 学习率调度策略；当前训练代码实现 plateau，根据验证 AUC 停滞降 lr。
_C.SOLVER.LR_SCHEDULER = "plateau"
# 验证集指标连续多少轮无提升后降低学习率。
_C.SOLVER.LR_PATIENCE = 5
# 学习率下限，防止 plateau 调度无限降低。
_C.SOLVER.MIN_LR = 1e-6
# Adam weight decay；越大正则越强，可缓解过拟合但可能欠拟合。
_C.SOLVER.WEIGHT_DECAY = 1e-4
# 是否将 drug_encoding.score.* 单独建一个 optimizer param group。
# 配合 SOLVER.CONF_SCORE_WEIGHT_DECAY / SOLVER.CONF_SCORE_LR_MULT 使用；
# 解释性变体推荐: GROUP=True, WEIGHT_DECAY=0.0, LR_MULT=5.0，破除 collapse 第 3 因子（无对抗 shrinkage）。
_C.SOLVER.CONF_SCORE_PARAM_GROUP = False
# 当 CONF_SCORE_PARAM_GROUP=True 时，score MLP 单独的 weight_decay；<0 表示沿用全局 WEIGHT_DECAY。
_C.SOLVER.CONF_SCORE_WEIGHT_DECAY = 0.0
# 当 CONF_SCORE_PARAM_GROUP=True 时，score MLP 单独的 lr 倍数（基于 SOLVER.LR）。
_C.SOLVER.CONF_SCORE_LR_MULT = 5.0
# Conformer 多样性辅助 loss 权重（基于 UFF energy 的 margin ranking）：
# 让低能量 conformer 的 conf_weight 显著高于高能量 conformer。0 表示关闭。
_C.SOLVER.CONF_DIVERSITY_RANK_WEIGHT = 0.0
# Conformer rank loss margin（在 softmax 概率空间）：要求 w[k_low_E] >= w[k_high_E] + margin。
_C.SOLVER.CONF_DIVERSITY_RANK_MARGIN = 0.05
# 弱负熵正则权重：鼓励 conf_weight 远离 uniform；>0 时即使无 rank 监督也避免 softmax 退化。
# 注意应保持较小（~1e-3）否则会过 confident。0 表示关闭。
_C.SOLVER.CONF_DIVERSITY_ENTROPY_WEIGHT = 0.0
# 是否按训练集正负样本比例自动/手动设置 BCE pos_weight。
_C.SOLVER.USE_POS_WEIGHT = True
# 正类权重；<=0 时由训练集自动计算，=1.0 基本等价不重加权。
_C.SOLVER.POS_WEIGHT = 0.0
# 验证集 AUC 连续多少轮无有效提升后 early stopping。
_C.SOLVER.EARLY_STOP_PATIENCE = 10
# 用验证集哪个指标选择 best checkpoint；支持 auc/auroc、aupr/auprc、acc/accuracy、f1、loss。
_C.SOLVER.BEST_METRIC = "acc"
# 判断 BEST_METRIC 有效提升的最小增量。
_C.SOLVER.MIN_DELTA = 1e-4
# 随机种子，影响参数初始化、shuffle 和可复现实验。
_C.SOLVER.SEED = 42

# RESULT
_C.RESULT = CN()
# 默认输出根目录；实际训练可用 --save_dir 覆盖。
_C.RESULT.OUTPUT_DIR = r"outputs/result/"
# 是否保存模型；当前 DeepSpeed 训练流程每轮保存 checkpoint。
_C.RESULT.SAVE_MODEL = True

def get_cfg_defaults():
    return _C.clone()
