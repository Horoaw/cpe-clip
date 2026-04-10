# CPE-CLIP 代码是怎么跑起来的？(技术细节拆解)

这套代码写得比较模块化，主要逻辑全在 `CoLeLib` 文件夹里。

## 1. 整个流程的入口：`train.py`
这个脚本其实就是一个“参数转发器”。它读取 `--n_runs`（实验轮数）和 `--dataset_name`，然后循环跑实验。
*   它会根据你给的参数去实例化 `CLIPPE` 策略类。
*   最值得注意的是 `seeds` 的处理，它确保了实验是可复现的，不是靠运气撞出来的结果。

## 2. 模型里的“手术”：`CoLeLib/models/clip_models.py`
这是最硬核的部分。它不是直接调库，而是重写了 CLIP 的 `forward` 过程。
*   **PromptLearner**：这个类在模型初始化时造了两堆 Tensor（视觉和文本的 Prompts）。
*   **注入逻辑**：在每一层 Transformer 处理之前，它用 `torch.cat` 把这些 Prompts 强行拼到图像 Patch 或者文本 Token 的前面。这意味着 CLIP 在计算 Self-Attention 时，不得不考虑这些额外注入的信息。

## 3. 动态分类头：`incremental_classifier.py`
增量学习最麻烦的就是类别数会变。
*   这里的 `IncrementalClassifier` 会根据 Session 进度动态扩展。
*   它保存了每一类的 **Prototypes（原型）**。简单理解就是每一类的特征平均值。
*   推理时，模型把图片过一遍得到特征，看这个特征离哪个“原型”近，就判给哪一类。

## 4. 训练策略：`clip_pe.py`
这里控制了“怎么训练”。
*   **冻结参数**：代码里有一行 `self.model.requires_grad_(False)`，然后只把 Prompts 送进优化器。
*   **Session 切换**：每当一个 Session 结束，它会调用 `model.get_prototypes()` 把当前类的特征中心存下来。在下个 Session 训练新类时，它会拿着这些旧原型去做正则化，防止模型“喜新厌旧”。

## 5. 总结
这个项目写得挺地道的。它避开了全量微调大模型的坑，只去磨合那一丁点 Prompts 权重。代码结构很清晰：数据归数据 (`datasets/`)，模型归模型 (`models/`)，算法逻辑全在 `strategies/`。
