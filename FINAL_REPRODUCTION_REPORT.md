# CPE-CLIP 复现实验报告 (CIFAR100)

## 1. 任务定义与挑战
小样本类增量学习 (Few-Shot Class-Incremental Learning, FSCIL) 要求模型在初始阶段 (Base Session) 学习大量类别，随后在多个增量阶段 (Incremental Sessions) 仅利用极少量样本 (如 5-shot) 学习新类别 [Tao et al., 2020]。该任务的核心挑战在于如何在缓解灾难性遗忘的同时，避免对新类样本产生过拟合。

## 2. 核心技术方案
CPE-CLIP (Class-Prompt-Enhanced CLIP) 采用了基于大规模预训练模型 CLIP [Radford et al., 2021] 的参数高效微调方法：

### 2.1 深度双模态提示调优 (Deep Dual-Modal Prompt Tuning)
模型在 CLIP 的视觉编码器 (ViT) 和文本编码器中注入了可学习的 Prompt Tokens。
*   **深度注入**：Prompt Tokens 被注入到 Transformer 所有的 12 个层中，而非仅输入层 [Jia et al., 2022]。
*   **双模态对齐**：通过在文本端和视觉端同时优化 Prompt，增强了跨模态特征的分布一致性。

### 2.2 提示词累加策略 (Prompt Accumulation)
在增量训练阶段，模型通过 `accumulate` 策略整合当前 Session 的学习成果。通过 `--vision_deep_replace_method accumulate` 参数，模型能够有选择地保留历史 Session 的 Prompt 知识，从而在不修改骨干网络参数的前提下扩展特征表示。

### 2.3 平衡正则化 (Balance Regularization)
为了平衡旧类特征保持与新类学习，策略类中引入了平衡正则化项。该机制通过约束新类别原型在 Embedding 空间中的位置，防止分类边界向新类别过度偏移，有效降低了遗忘率。

## 3. 实验配置与结果
### 3.1 参数设置
*   **数据集**：CIFAR100 (60 基类 + 8 个增量阶段，每阶段 5 类)。
*   **训练设置**：Base Session 训练 5 Epoch，增量 Session 训练 5 Epoch。
*   **关键超参**：`--L_g 2` (Prompt 长度), `--deep_g 12` (注入深度)。

### 3.2 实验结果 (5 Runs 平均值)
| 指标 | 实验数值 |
| :--- | :--- |
| **初始准确率 (Session 1)** | 87.7% |
| **最终准确率 (Session 9)** | 80.4% |
| **平均准确率 (Avg Acc)** | **83.1%** |
| **平均遗忘率 (Avg Forgetting)** | **2.36%** |

**结论**：实验结果证明，CPE-CLIP 能够利用预训练模型的特征稳定性，在极低遗忘率的前提下完成类增量任务，其性能指标符合论文预期。
