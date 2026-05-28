# 环境迁移记录

## 目标环境
- PyTorch 2.4.0 + CUDA 12.4
- DeepSpeed 0.18.9
- mamba_ssm 2.2.2
- causal_conv1d 1.6.1
- conda 虚拟环境: ldm-dti

## 步骤

### 1. 创建 conda 环境并安装 PyTorch

```bash
conda create -n ldm-dti python=3.10 -y
conda activate ldm-dti
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2. 安装 CUDA 12.4 Toolkit（编译依赖）

系统 nvcc 为 11.8，需要安装 12.4 用于编译 mamba_ssm 和 causal_conv1d。

```bash
# 下载（阿里云镜像）
wget -c https://mirrors.aliyun.com/opsx/ecs/linux/binary/nvidia/cuda/12.4.1/cuda_12.4.1_550.54.15_linux.run

# 安装（只装 toolkit，不装驱动）
sh cuda_12.4.1_550.54.15_linux.run --toolkit --toolkitpath=/paddle/cuda-12.4 --silent --no-opengl-libs

# 设置环境变量
export CUDA_HOME=/paddle/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证
nvcc --version  # 应输出 release 12.4
```

### 3. 编译安装 causal_conv1d

```bash
cd /paddle/lv/TAMR-DTI/causal-conv1d-1.6.1.post4
pip install --no-build-isolation . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 编译安装 mamba_ssm

```bash
cd /paddle/lv/TAMR-DTI/mamba-2.2.2
pip install --no-build-isolation . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 安装 DeepSpeed

```bash
pip install --no-build-isolation . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6. 安装 DGL

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

### 7. 安装其他依赖

```bash
pip install dgllife einops yacs rdkit transformers scikit-learn shap pandas numpy swanlab -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 8. 持久化环境变量（写入 bashrc）

```bash
echo 'export CUDA_HOME=/paddle/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## 硬件信息

- 8x Tesla V100-SXM2-32GB
- 驱动: 550.90.07
- CUDA: 12.4
