# 手把手带你实战 Transformers

![手把手带你实战Transformers](./imgs/1.png)

---

## 详细介绍

本仓库是 **手把手带你实战 Transformers** 课程的配套代码库，面向希望系统学习 [Hugging Face Transformers](https://github.com/huggingface/transformers) 的开发者与研究者。内容从环境搭建、基础 API 使用，一直延伸到 NLP 多任务实战、参数高效微调（PEFT）、低精度/量化训练以及分布式训练，形成一条完整的学习与实战路径。

### 适合谁

- **初学者**：有 Python 和基本深度学习基础，想从零掌握 Transformers 的用法
- **进阶者**：已会用 Pipeline，希望深入 Tokenizer、Model、Datasets、Trainer 等组件
- **实战者**：需要做文本分类、NER、阅读理解、摘要、对话等 NLP 任务并落地
- **大模型相关**：关注 LoRA/QLoRA、低精度训练、多卡/分布式训练与加速

### 你会学到什么

- **基础篇**：Pipeline、Tokenizer、Model、Datasets、Evaluate、Trainer 的用法，并用一个完整的文本分类项目串起来
- **实战篇**：命名实体识别、机器阅读理解、多项选择、文本相似度、检索式/生成式对话、语言模型、文本摘要等任务的建模与代码实现
- **高效微调篇**：基于 PEFT 的 BitFit、Prompt-Tuning、P-Tuning、Prefix-Tuning、LoRA、IA3 等方法的原理与实战
- **低精度篇**：半精度（FP16）、8bit、4bit（QLoRA）训练，结合 LLaMA2、ChatGLM3、InternLM 等模型
- **分布式篇**：Data Parallel、DDP、Accelerate 以及 Accelerate + DeepSpeed 的配置与使用

### 仓库特点

- **按章节组织**：目录与课程章节一一对应（01–33），便于按顺序学习或按需查阅
- **Jupyter + Python**：以 Notebook 为主，配合少量 `.py` 脚本，适合边看边跑、修改实验
- **数据与脚本齐全**：各任务配备相应数据集（或加载脚本）和评估脚本（如 `metric_accuracy.py`、`seqeval_metric.py`、`cmrc_eval.py` 等），可直接复现
- **覆盖主流生态**：涉及 `transformers`、`datasets`、`peft`、`accelerate`、`bitsandbytes` 等常用库，与当前社区实践一致

学习建议：先完成 **01-Getting Started** 打牢基础，再按兴趣选择 **02-NLP Tasks** 中的任务实战，然后根据需要学习 **03-PEFT**、**04-Kbit Training** 和 **05-Distributed Training**。**Others** 中的 Optuna 超参搜索可作为补充技能。

---

## 项目结构

```
learn-huggingface/
├── 01-Getting Started/          # 基础入门：Pipeline、Tokenizer、Model、Datasets、Evaluate、Trainer
│   ├── 01-introduction/         # 环境与入门示例
│   ├── 02-pipeline/
│   ├── 03-tokenizer/
│   ├── 04-model/                # 模型使用 + 文本分类示例（ChnSentiCorp）
│   ├── 05-datasets/
│   ├── 06-evaluate/
│   └── 07-trainer/
├── 02-NLP Tasks/                # NLP 实战：分类、NER、阅读理解、摘要、对话等
│   ├── 08-transformers_solution/
│   ├── 09-token_classification/ # 命名实体识别
│   ├── 10-question_answering/   # 机器阅读理解（含截断/滑动窗口）
│   ├── 11-multiple_choice/     # 多项选择（C3）
│   ├── 12-sentence_similarity/ # 文本相似度（交互/匹配）
│   ├── 13-retrieval_chatbot/   # 检索式对话（FAQ）
│   ├── 14-language_model/      # 因果/掩码语言模型
│   ├── 15-text_summarization/  # 文本摘要（T5/GLM）
│   └── 16-generative_chatbot/  # 生成式对话（Alpaca 格式）
├── 03-PEFT/                     # 参数高效微调：BitFit、Prompt/P/Prefix-Tuning、LoRA、IA3
│   ├── 17-bitfit/
│   ├── 18-prompt_tuning/
│   ├── 19-p_tuning/
│   ├── 20-prefix_tuning/
│   ├── 21-lora/
│   ├── 22-ia3/
│   └── 23-peft/                 # PEFT 进阶（多适配器管理等）
├── 04-Kbit Training/           # 低精度/量化训练
│   ├── 24-llm_download/         # 模型下载与加载
│   ├── 25-16bits_training/     # 半精度（LLaMA2、ChatGLM3）
│   ├── 26-8bits_training/      # 8bit
│   └── 27-4bits_training/      # 4bit QLoRA（含权重分布可视化）
├── 05-Distributed Training/    # 分布式训练
│   ├── 28-remote ssh/          # 远程与 DP 示例
│   ├── 29-data parallel/
│   ├── 30-distributed data parrallel/
│   ├── 31-accelerate ddp/
│   ├── 32-accelerate advanced/
│   └── 33-accelerate_deepspeed/ # DeepSpeed ZeRO-2/3 配置
├── Others/
│   └── 01-hyp_tune/            # Optuna 超参搜索
├── imgs/                       # 图片资源
├── pptx/                       # 课程 PDF 讲义
└── README.md
```

---

## 环境要求

推荐 Python 3.8+，主要依赖示例版本如下（可按需调整）：

| 依赖         | 推荐版本                          |
| ------------ | --------------------------------- |
| torch        | 2.2.1+cu118（或与 CUDA 版本匹配） |
| transformers | 4.42.4                            |
| peft         | 0.11.1                            |
| datasets     | 2.20.0                            |
| accelerate   | 0.32.1                            |
| bitsandbytes | 0.43.1（低精度/量化章节需要）     |
| faiss-cpu    | 1.7.4（检索式对话等需要）         |
| tensorboard  | 2.14.0                            |

安装示例：

```bash
pip install torch transformers peft datasets accelerate
# 低精度训练
pip install bitsandbytes
# 检索等
pip install faiss-cpu tensorboard
```

部分章节会从 Hugging Face Hub 或本地加载模型与数据集，请保证网络可用或已提前下载好模型/数据。

---

## 快速开始

1. **克隆仓库**

   ```bash
   git clone https://github.com/<your-org>/learn-huggingface.git
   cd learn-huggingface
   ```

2. **按顺序学习**

   - 从 `01-Getting Started/01-introduction` 开始，运行其中的 `demo.ipynb` 或 `demo.py` 检查环境
   - 然后依次学习 `02-pipeline`、`03-tokenizer`、`04-model` 等，`04-model` 和 `05-datasets` 起会用到 ChnSentiCorp 等数据
   - 每个子目录下的 `.ipynb` 为对应章节的主讲内容，可直接用 Jupyter 或 VS Code 打开运行

3. **跑通一个完整任务**

   - 例如文本分类：在 `01-Getting Started/04-model` 或 `07-trainer` 中运行 `classification_demo.ipynb`
   - 或选一个你感兴趣的任务（如 `02-NLP Tasks/09-token_classification` 的 NER），按照 Notebook 中的步骤执行即可

课程讲义位于 `pptx/` 目录，可与代码结合使用。

---

## 课程与致谢

本仓库配套 **手把手带你实战 Transformers** 系列课程，视频与代码会持续更新。若对你有帮助，欢迎 Star 或参与贡献。

如有问题或建议，可通过仓库 Issue 反馈。
