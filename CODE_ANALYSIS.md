# CPE-CLIP 代码架构分析报告

## 1. 项目全景概述
CPE-CLIP 是一个基于 `PyTorch` 和 `HuggingFace Transformers` 构建的 Continual Learning (CL) 研究框架。其核心逻辑封装在 `CoLeLib` (Continual Learning Library) 库中，采用模块化设计，方便扩展新的数据集或训练策略。

## 2. 核心模块详解

### 2.1 数据层 (`CoLeLib/datasets/`)
该模块负责处理 FSCIL 的特殊逻辑：
*   **`cifar100_FSCIL.py` / `cub200_FSCIL.py`**: 不同于标准 Dataset，这些类实现了“会话式”加载。它读取 `splits/` 下的 `.pkl` 文件，确保模型在第 $N$ 个 Session 只能看到当前类的 Few-shot 样本及旧类的原型数据。
*   **预处理**: 集成了 `transformers.CLIPProcessor`，确保输入图像尺寸和标准化方式与 CLIP 预训练时完全一致。

### 2.2 模型层 (`CoLeLib/models/`)
*   **`clip_models.py`**: 这是项目的“手术刀”。它通过对 `CLIPModel` 进行 Monkey Patch 或封装，在 `forward` 过程中动态注入可学习的 Prompt Tensors。
*   **`incremental_classifier.py`**: 实现了一个动态生长的分类头。随着 Session 增加，分类器的输出维度从 60 逐渐扩展到 100，并使用余弦相似度作为分类依据，这在小样本学习中比线性层更稳定。

### 2.3 策略层 (`CoLeLib/training/strategies/`)
这是项目的核心算法实现：
*   **`CLIPPE` (CLIP Prompt Engineering)**: 继承自基类模板，实现了 `train()` 和 `eval()` 循环。
    *   **知识保持**: 在每个 Session 结束后，它会计算当前类的类原型 (Prototypes) 并存储，用于后续 Session 的正则化或回放。
    *   **优化器设置**: 仅针对注入的 Prompts 进行微调，冻结 CLIP 的骨干网络 (Backbone)，极大地减少了训练参数量，防止过拟合。

### 2.4 评估层 (`CoLeLib/evaluation/`)
*   **指标计算**: 不仅计算 Top-1 Accuracy，还通过 `metrics.py` 计算 **Forgetting**（定义为：某类在初始学习时的准确率与当前准确率之差）。

## 3. 运行流程分析
1.  **初始化**: `train.py` 解析命令行参数（如提示词深度 `deep_g`）。
2.  **Base 训练**: 在第一个 Session (Exp 1) 进行大样本量训练，构建基础特征空间。
3.  **增量学习**: 进入 Exp 2-9，模型通过少量样本微调 Prompts。
4.  **指标汇总**: 每个 Session 结束后进行全量测试（旧类+新类），最后由 `results.ipynb` 读取 JSON 生成可视化图表。

## 4. 设计模式评价
*   **高度解耦**: 策略 (Strategy) 与模型 (Model) 分离，若要研究新的增量算法，只需在 `strategies/` 下新建文件，无需修改模型结构。
*   **可复现性**: 通过硬编码种子列表 `[42, 13, 50, 24, 69]` 确保了科研实验的严谨性。
