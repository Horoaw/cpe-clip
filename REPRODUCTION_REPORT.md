# CPE-CLIP 复现实验报告

## 1. 研究背景与任务定义
本报告记录了对 **CPE-CLIP (Class-Prompt-Enhanced CLIP)** 在小样本类增量学习 (FSCIL) 任务上的复现过程。FSCIL 要求模型在仅有少量样本的情况下，不断学习新类而不遗忘旧类 [Tao et al., 2020]。传统的增量学习方法在极低样本下容易发生严重的过拟合与灾难性遗忘。

## 2. 核心创新点分析
CPE-CLIP 的核心在于将大规模预训练模型 CLIP [Radford et al., 2021] 的零下泛化能力通过 **深度提示工程 (Deep Prompt Engineering)** 引入到增量学习中：

### 2.1 深度双模态提示注入 (Deep Dual-Modal Prompting)
不同于传统的线性探针 (Linear Probing)，CPE-CLIP 在 CLIP 的视觉编码器和文本编码器中同时注入了可学习的提示词 (Prompts)。通过参数 `--L_g` 控制提示词数量，`--deep_g` 控制注入的 Transformer 层数（实验中使用 12 层全量注入）。
> **引用**：这种做法借鉴了 VPT (Visual Prompt Tuning) [Jia et al., 2022]，但将其扩展到了双模态对齐空间。

### 2.2 提示词累加策略 (Prompt Accumulation Strategy)
在增量阶段，项目实现了 `replace` 和 `accumulate` 两种策略。实验发现 `accumulate` 策略能更好地整合不同 Session 的知识，通过 `--vision_deep_replace_method accumulate` 确保视觉特征在增量过程中保持连续性。

### 2.3 平衡正则化 (Balance Regularization)
在 `CLIPPE` 策略类中，项目引入了 `regularization_method="balance"`。该方法通过约束新类原型与旧类原型在 Embedding 空间中的距离，防止分类头向新类过度偏移。

## 3. 工程实现与复现路径
### 3.1 环境配置
*   **基础环境**：Python 3.10.6, PyTorch 1.13.1+cu116。
*   **关键依赖**：`transformers` (用于调用 CLIP), `pytorch-cosine-annealing-with-warmup` (用于平滑学习率)。

### 3.2 数据流水线
复现采用了标准的 FSCIL 划分：
*   **Base Session**: 60 类 (CIFAR100)。
*   **Incremental Sessions**: 8 个阶段，每阶段 5 类，每类仅 5 个样本。
项目通过 `CoLeLib.datasets` 中的 `.pkl` 索引文件精确控制了每一阶段的可见数据。

## 4. 复现结果分析
在 CIFAR100 任务上，经过 5 轮 (Runs) 实验取平均值，得到以下结果：
*   **平均准确率 (AvgAcc)**: **83.1%**
*   **平均遗忘率 (AvgForgetting)**: **2.36%**

**结论**：复现结果表明，CPE-CLIP 在保持高准确率的同时，遗忘率极低，充分证明了 CLIP 预训练特征在小样本增量场景下的鲁棒性。
