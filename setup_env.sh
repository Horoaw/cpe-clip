#!/bin/bash

# 1. 定义环境名称
ENV_NAME="cpe-clip"

echo "开始配置 $ENV_NAME 环境..."

# 2. 创建 Conda 环境 (Python 3.10.6)
conda create -y -n $ENV_NAME python=3.10.6

# 3. 安装 PyTorch 相关组件 (使用 1.13.1 以避免 Ubuntu 上的 libtorch 兼容性报错)
# 适配 CUDA 11.6 (兼容绝大多数 Ubuntu + NVIDIA 驱动环境)
conda run -n $ENV_NAME pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 4. 安装核心科研依赖库
conda run -n $ENV_NAME pip install \
    transformers==4.26.1 \
    pandas \
    numpy \
    matplotlib \
    pillow \
    tqdm \
    scikit-learn \
    pyyaml \
    gdown

# 5. 下载并准备数据 (CUB200 和 CIFAR100)
echo "正在下载数据集和划分文件..."

# 创建目录
mkdir -p data splits

# 下载 CUB200 数据集 (Google Drive ID: 1PumwrWQCNZTBbgW6bbZ6NDD4xyA0wGfH)
echo "下载 CUB200 数据集..."
conda run -n $ENV_NAME gdown 1PumwrWQCNZTBbgW6bbZ6NDD4xyA0wGfH -O data/cub200.zip
unzip -o data/cub200.zip -d data/
rm data/cub200.zip

# 下载 CIFAR100 划分文件 (Google Drive ID: 1TpDUpUoy6pHUShbmnaRYFs1TfydV65-e)
echo "下载 CIFAR100 划分文件..."
conda run -n $ENV_NAME gdown 1TpDUpUoy6pHUShbmnaRYFs1TfydV65-e -O splits/cpe-clip-cifar100-splits.zip
unzip -o splits/cpe-clip-cifar100-splits.zip -d splits/
rm splits/cpe-clip-cifar100-splits.zip

# 下载 CUB200 划分文件 (Google Drive ID: 1JEtSvUTJfVaycysCiIgoqLexHlEeubb1)
echo "下载 CUB200 划分文件..."
conda run -n $ENV_NAME gdown 1JEtSvUTJfVaycysCiIgoqLexHlEeubb1 -O splits/cpe-clip-cub200-splits.zip
unzip -o splits/cpe-clip-cub200-splits.zip -d splits/
rm splits/cpe-clip-cub200-splits.zip

# 6. 安装论文特有的学习率调度器 (从 GitHub 直接拉取)
echo "正在安装特殊的学习率调度器..."
conda run -n $ENV_NAME pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git

# 6. 环境验证
echo "-----------------------------------"
echo "验证安装结果:"
conda run -n $ENV_NAME python -c "import torch; import transformers; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Transformers: {transformers.__version__}')"

echo "-----------------------------------"
echo "配置完成！请运行以下命令激活环境："
echo "conda activate $ENV_NAME"
