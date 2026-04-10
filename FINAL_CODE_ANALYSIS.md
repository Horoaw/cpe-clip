# CPE-CLIP 系统架构与代码逻辑分析

## 1. 项目结构概述
CPE-CLIP 采用模块化设计，核心逻辑位于 `CoLeLib` 库中。系统分为数据加载 (Datasets)、模型定义 (Models)、训练策略 (Strategies) 和指标评估 (Evaluation) 四个主要模块。

## 2. 核心模块逻辑实现

### 2.1 提示注入机制 (`CoLeLib/models/clip_models.py`)
模型通过对 `CLIPModel` 的 `forward` 路径进行拦截来实现 Prompt 注入：
*   **PromptLearner**：定义了视觉 (Vision) 和文本 (Text) 两个维度的可学习 Parameter 矩阵。
*   **注入实现**：在 Transformer 的每一个 Block 处理输入序列前，利用 `torch.cat` 将 Prompt Tokens 与原始序列 (Class Token + Patch Tokens) 进行拼接。由于 CLIP 骨干网络被设置为 `requires_grad_(False)`，训练过程仅优化这些注入的 Tokens。

### 2.2 增量分类器 (`CoLeLib/models/incremental_classifier.py`)
为了应对类别数的动态增长，项目设计了 `IncrementalClassifier`：
*   **余弦相似度度量**：分类逻辑基于输入特征向量与类别原型 (Class Prototypes) 的余弦相似度，而非传统的点积线性映射。这有助于消除特征幅值差异带来的分类偏差。
*   **动态维度扩展**：在每个 Session 开始前，分类器的输出维度会根据当前可见的总类别数进行动态调整。

### 2.3 训练与原型管理 (`CoLeLib/training/strategies/clip_pe.py`)
训练策略类 `CLIPPE` 负责协调整个增量学习过程：
*   **原型计算**：在每个阶段训练完成后，模型会遍历当前 Session 的样本，计算其特征均值作为该类的原型 (Prototype)。
*   **正则化约束**：在增量阶段，策略引入了平衡正则化。通过计算当前训练特征与已存储旧类原型的距离，约束模型权重的更新方向，从而缓解灾难性遗忘。

### 2.4 数据流转逻辑
1.  **输入端**：`CoLeLib/datasets/` 通过索引 `.pkl` 文件精确控制每个 Session 可见的数据子集。
2.  **处理端**：图像经过冻结的 CLIP 编码器及可学习的深度提示层，提取出高维特征向量。
3.  **输出端**：特征向量在 `IncrementalClassifier` 中进行相似度比对，产生类别预测，并将 Acc/Loss 指标序列化为 JSON 供分析。

## 3. 设计评价
项目在工程实现上具有较高的严谨性。通过冻结大模型参数并仅训练轻量化提示层，显著降低了小样本场景下的过拟合风险。模块化的解耦设计也使得算法能够快速在 CIFAR100、CUB200 和 miniImageNet 之间切换。
