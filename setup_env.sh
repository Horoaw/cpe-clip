#!/bin/bash
# ==========================================
# CPE-CLIP 终极全自动部署脚本 (路径直达版)
# ==========================================

set -e  # 遇到任何错误立即停止

echo ">>> [1/7] 安装系统基础工具 (需要 sudo)..."
sudo apt-get update
sudo apt-get install -y unzip wget curl git

# 1. 确定安装路径
CONDA_ROOT="$HOME/miniconda"
ENV_NAME="cpe-clip"
ENV_PATH="$CONDA_ROOT/envs/$ENV_NAME"

# 2. 安装 Miniconda (如果尚未安装)
if [ ! -d "$CONDA_ROOT" ]; then
    echo ">>> [2/7] 正在下载并安装 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh
    bash miniconda_installer.sh -b -p "$CONDA_ROOT"
    rm miniconda_installer.sh
    # 初始化
    "$CONDA_ROOT/bin/conda" init bash
    echo "Miniconda 安装完成。"
    else
    echo ">>> [2/7] 检测到 Miniconda 已存在，跳过安装。"
    fi

    # 2.5 接受 Anaconda 服务条款 (解决 CondaToSNonInteractiveError)
    echo ">>> [2.5] 正在接受 Anaconda 服务条款..."
    "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

    # 3. 创建环境 (直接调用二进制文件，确保成功)
if [ ! -d "$ENV_PATH" ]; then
    echo ">>> [3/7] 正在创建 Conda 环境: $ENV_NAME ..."
    "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" python=3.10.6
else
    echo ">>> [3/7] 环境 $ENV_NAME 已存在。"
fi

# 定义环境内的二进制路径，后续操作全部“路径直达”
PYTHON_EXE="$ENV_PATH/bin/python"
PIP_EXE="$ENV_PATH/bin/pip"

# 4. 安装 PyTorch (1.13.1 + cu116)
echo ">>> [4/7] 正在安装 PyTorch (适配 CUDA 11.6)..."
"$PIP_EXE" install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 5. 安装科研库和下载工具
echo ">>> [5/7] 正在安装科研库和 gdown..."
"$PIP_EXE" install transformers==4.26.1 pandas numpy matplotlib pillow tqdm scikit-learn pyyaml gdown

# 6. 下载并解压数据 (Google Drive ID)
echo ">>> [6/7] 正在准备数据集和划分文件..."
mkdir -p data splits

# 函数：尝试下载，失败则提示手动上传
prepare_data() {
    local FILE_PATH=$1
    local GID=$2
    local TARGET_DIR=$3

    if [ -f "$FILE_PATH" ]; then
        echo "检测到 $FILE_PATH 已存在，跳过下载。"
    else
        echo "尝试从 Google Drive 下载 $FILE_PATH ..."
        if ! "$PYTHON_EXE" -m gdown "$GID" -O "$FILE_PATH"; then
            echo "错误：无法连接到 Google Drive。请从本地下载并上传到 $FILE_PATH"
            return 1
        fi
    fi
    echo "正在解压缩 $FILE_PATH 到 $TARGET_DIR ..."
    unzip -o "$FILE_PATH" -d "$TARGET_DIR"
}

# 处理 CUB200 数据集
prepare_data "data/cub200.zip" "1PumwrWQCNZTBbgW6bbZ6NDD4xyA0wGfH" "data/" || echo "请手动处理 data/cub200.zip"

# 处理 CIFAR100 划分
prepare_data "splits/cifar100.zip" "1TpDUpUoy6pHUShbmnaRYFs1TfydV65-e" "splits/" || echo "请手动处理 splits/cifar100.zip"

# 处理 CUB200 划分
prepare_data "splits/cub200.zip" "1JEtSvUTJfVaycysCiIgoqLexHlEeubb1" "splits/" || echo "请手动处理 splits/cub200.zip"

# 7. 安装特殊调度器
echo ">>> [7/7] 正在安装特殊的学习率调度器..."
"$PIP_EXE" install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git

echo "=========================================="
echo "所有任务已圆满完成！"
echo "请执行以下两条命令进入工作状态："
echo "1. source ~/.bashrc"
echo "2. conda activate $ENV_NAME"
echo "然后你可以直接运行训练脚本："
echo "python train.py --L_g 2 --deep_g 12 --dataset_name cifar100"
echo "=========================================="
