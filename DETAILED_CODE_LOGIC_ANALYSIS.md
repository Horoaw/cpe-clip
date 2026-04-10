# CPE-CLIP 代码逻辑与架构深度分析

## 1. 软件架构：三层分发模式
项目代码遵循典型的科研框架模式，从上到下分为：运行控制层、算法策略层、底层模型层。

### 1.1 运行控制层 (`train.py`)
这是整个实验的入口，它完成了以下关键任务：
*   **超参数路由**: 根据 `--dataset_name` 自动配置不同的 Batch Size 和 Epoch（如 CUB200 因为更难，Base Epoch 设为 6）。
*   **多轮实验并行**: 通过 `for run in range(n_runs)` 循环，利用 `seeds` 列表确保结果的统计显著性。
*   **动态策略选择**: 根据 `--ablation` 参数，决定实例化 `CLIPPE` (完整版) 还是 `CLIPPEAblated` (消融实验版)。

### 1.2 算法策略层 (`CoLeLib/training/strategies/`)
这是逻辑最复杂的部分，主要文件是 `clip_pe.py`：
*   **`train()` 方法**: 并非简单的 `optimizer.step()`。在每个 Session 开始前，它会调用 `model.expand_classifier()`，动态增加分类器的输出维度。
*   **`_train_exp()`**: 针对 Base Session 和 Incremental Session 采用了不同的逻辑。在增量阶段，它会计算 **Prototypes**（类中心特征），并将其存储在 `self.prototypes` 中，用于后续的类均衡化。
*   **验证逻辑**: 每个阶段结束后，会对**所有已学过的类**进行全量测试，这是衡量增量学习好坏的标准做法。

### 1.3 底层模型层 (`CoLeLib/models/`)
这是实现“深度提示注入”的核心：
*   **`clip_models.py`**:
    *   通过自定义 `CustomCLIP` 类包装了 OpenAI 的 `CLIPModel`。
    *   **注入点**: 在 Transformer 的 `encoder_layer` 之前拦截输入序列，强行拼入 (concatenate) 长度为 `L_g` 的 Prompt 矩阵。
    *   **维度匹配**: 针对 Vision (d=768) 和 Text (d=512) 编码器分别管理两组不同维度的 Prompts。
*   **`incremental_classifier.py`**:
    *   实现了一个基于 **Cosine Similarity** 的分类层。相比传统的 `nn.Linear`，余弦相似度受特征幅值影响较小，能更好地平衡新旧类。

## 2. 关键业务流程：如何实现“不遗忘”？
1.  **特征冻结**: 在 `strategy` 初始化时，通过 `model.requires_grad_(False)` 冻结了 CLIP 的海量参数，只有不到 1MB 的 Prompts 参与梯度更新。
2.  **原型保持 (Prototype Buffering)**: 模型不存储原始图像（保护隐私且节省空间），而是存储每一类的特征向量均值。
3.  **预测校准**: 在推理阶段，模型计算 `Feature(img)` 与 `Prototypes` 的余弦距离，通过 Softmax 归一化后输出类别概率。

## 3. 数据流转图
1.  **Raw Image** $\to$ `CLIPProcessor` (Resize/Norm)
2.  **Preprocessed Image** $\to$ `VisionEncoder` (与 `VisionPrompts` 融合)
3.  **Context Vector** $\to$ `IncrementalClassifier` (匹配 Class Prototypes)
4.  **Output** $\to$ `JSON Logger` (记录每个 Session 的 Acc/Loss)

## 4. 总结
该项目的优秀之处在于其对 CLIP 内部结构的“外科手术式”改造。通过在正确的位置（Transformer Layers）注入极少量的可学习参数，实现了在极低算力消耗下完成高质量的类增量学习。
