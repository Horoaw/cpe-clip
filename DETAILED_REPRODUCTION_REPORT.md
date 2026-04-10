# CPE-CLIP 深度复现与科研创新分析报告

## 1. 任务背景：FSCIL 的严苛挑战
小样本类增量学习 (Few-Shot Class-Incremental Learning, FSCIL) 要求模型在保持旧类知识的同时，仅利用 5-shot 或 1-shot 样本学习新类 [1]。其核心难点在于**灾难性遗忘 (Catastrophic Forgetting)** 与**新类过拟合 (New-class Overfitting)** 的权衡 [2]。

## 2. 核心创新点深度解析

### 2.1 基于 CLIP 的特征空间锚定 (Feature Space Anchoring)
**创新点**：传统方法如 iCaRL [3] 或 CEC [4] 依赖于从零开始训练的 ResNet 骨干，在极低样本下特征提取器极易崩溃。CPE-CLIP 利用了 CLIP [5] 预训练的强大零样本 (Zero-shot) 迁移能力，将图像和文本投影到统一的对齐空间中。
**工程实现**：通过 `CLIPProcessor` 强制执行原始预处理逻辑，确保复现时的输入分布与预训练一致。

### 2.2 深度双模态提示调优 (Deep Dual-Modal Prompt Tuning)
**创新点**：本项目并非微调 CLIP 的全部参数，而是引入了可学习的 **Prompts**。
*   **Deep Prompting**: 借鉴了 VPT-Deep [6]，在 CLIP 的全部 12 层 Transformer 中每层都注入了可学习的 Latent Tokens。
*   **Dual-Modal**: 同时在视觉编码器和文本编码器中注入 Prompt（由参数 `--L_g` 定义长度）。这使得模型能够在文本提示（类别名称）和视觉特征之间进行动态微调。
> **引用**：这种结构允许模型在不破坏 CLIP 原始判别力的情况下，通过极少量参数（约 0.1% 的全量参数）快速适应特定数据集 [7]。

### 2.3 增量提示词累加 (Prompt Accumulation)
**创新点**：在 Session $t$ 学习到的提示词 $P_t$ 不会被丢弃，而是通过 `--vision_deep_replace_method accumulate` 策略与历史提示词融合。
**工程细节**：代码在 `clip_models.py` 中实现了 `PromptLearner` 类，动态管理不同 Session 的权重矩阵。这保证了模型在面对 Session 9 的新类时，依然保留了 Session 1 的几何分布特征。

## 3. 关键工程实践
### 3.1 平衡正则化 (Balance Regularization)
为了解决分类器偏向新类的问题，项目引入了正则化项：
$$ \mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda \cdot \mathcal{L}_{reg} $$
其中 $\mathcal{L}_{reg}$ 约束了新旧类原型的余弦相似度。这在代码 `clip_pe.py` 中通过 `regularization_method="balance"` 触发，有效缓解了新类的“侵略性”。

### 3.2 高级学习率调度
复现中使用了 `CosineAnnealingWarmupRestarts`。
*   **Warmup**: 防止在微调初期由于梯度过大破坏 Prompts 的初始化。
*   **Cosine Decay**: 确保在 5 个 Epoch 的极短训练周期内，模型能平稳收敛到局部最优 [8]。

## 4. 复现数据摘要 (CIFAR100)
*   **Base Acc**: ~87.7% (Session 1)
*   **Final Acc**: ~80.4% (Session 9)
*   **Avg Acc**: **83.1%**
*   **Avg Forgetting**: **2.36%**
对比 SOTA 方法（如 Fact [9]），CPE-CLIP 在减少计算资源消耗的同时，保持了极具竞争力的遗忘率。

---
**参考文献引用**：
[1] Tao et al., CVPR 2020. [2] Rebuffi et al., CVPR 2017. [3] iCaRL. [4] Zhang et al., CVPR 2021. [5] Radford et al., ICML 2021. [6] Jia et al., ECCV 2022. [7] Zhou et al., IJCV 2022. [8] Loshchilov et al., ICLR 2017. [9] Zhou et al., CVPR 2022.
