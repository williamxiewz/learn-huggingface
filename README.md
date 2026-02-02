# 手把手带你实战 Transformers

> 从零入门 Hugging Face Transformers，系统掌握 Pipeline、模型微调、低精度训练与分布式训练。

## 目录

- [简介](#简介)
- [环境要求](#环境要求)
- [课程规划](#课程规划)
- [课程地址](#课程地址)
- [Star History](#star-history)

---

## 简介

本仓库为 **手把手带你实战 Transformers** 课程的配套代码，涵盖 Hugging Face 生态从入门到进阶的完整实践，包括基础组件、NLP 实战、PEFT 微调、低精度训练与分布式训练等。

---

## 环境要求

推荐使用以下依赖版本（以 `torch==2.2.1+cu118` 为例）：

| 依赖         | 版本        |
| ------------ | ----------- |
| torch        | 2.2.1+cu118 |
| transformers | 4.42.4      |
| peft         | 0.11.1      |
| datasets     | 2.20.0      |
| accelerate   | 0.32.1      |
| bitsandbytes | 0.43.1      |
| faiss-cpu    | 1.7.4       |
| tensorboard  | 2.14.0      |

或使用 `requirements.txt` 安装（如有）：

```bash
pip install -r requirements.txt
```

---

## 课程规划

| 篇章           | 内容概要                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------- |
| **基础入门**   | Pipeline、Tokenizer、Model、Datasets、Evaluate、Trainer，以及文本分类完整流程             |
| **实战演练**   | 命名实体识别、机器阅读理解、多项选择、文本相似度、检索式/生成式对话、语言模型、文本摘要等 |
| **高效微调**   | 以 PEFT 为核心：BitFit、Prompt-tuning、P-tuning、Prefix-Tuning、LoRA、IA3                 |
| **低精度训练** | 基于 bitsandbytes：半精度、8bit、4bit（QLoRA），含 LLaMA2、ChatGLM 等实战                 |
| **分布式训练** | 基于 accelerate：Data Parallel、DDP、与 DeepSpeed 集成                                    |
| **对齐训练**   | （规划中）                                                                                |
| **性能优化**   | （规划中）                                                                                |
| **系统演示**   | （规划中）                                                                                |

---

## 课程地址

课程视频同步更新于 **B 站** 与 **YouTube**，代码与视频持续维护。

- [Bilibili 合集](https://www.bilibili.com/video/BV1ma4y1g791)
- [YouTube 频道](https://www.youtube.com/@lunatic-zzz)

### 基础入门篇（已更新完成）

| 课时  | 内容                            | 视频链接                                                                                                               |
| ----- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 01    | 基础知识与环境安装              | [Bilibili](https://www.bilibili.com/video/BV1ma4y1g791) \| [YouTube](https://www.youtube.com/watch?v=ddCfxkCh-O8)      |
| 02    | 基础组件之 Pipeline             | [Bilibili](https://www.bilibili.com/video/BV1ta4y1g7bq) \| [YouTube](https://www.youtube.com/watch?v=Xeu3qFTP9qY&t=7s) |
| 03    | 基础组件之 Tokenizer            | [Bilibili](https://www.bilibili.com/video/BV1NX4y1177c) \| [YouTube](https://www.youtube.com/watch?v=G4JmQu-VWrU)      |
| 04 上 | 基础组件之 Model：基本使用      | [Bilibili](https://www.bilibili.com/video/BV1KM4y1q7Js) \| [YouTube](https://www.youtube.com/watch?v=xK-6VcLqa94)      |
| 04 下 | 基础组件之 Model：BERT 文本分类 | [Bilibili](https://www.bilibili.com/video/BV18T411t7h6) \| [YouTube](https://www.youtube.com/watch?v=nkwOQQDCDvc)      |
| 05    | 基础组件之 Datasets             | [Bilibili](https://www.bilibili.com/video/BV1Ph4y1b76w) \| [YouTube](https://www.youtube.com/watch?v=LRhcUjbSOEk)      |
| 06    | 基础组件之 Evaluate             | [Bilibili](https://www.bilibili.com/video/BV1uk4y1W7tK) \| [YouTube](https://www.youtube.com/watch?v=tpE2bleqk6A)      |
| 07    | 基础组件之 Trainer              | [Bilibili](https://www.bilibili.com/video/BV1KX4y1a7Jk) \| [YouTube](https://www.youtube.com/watch?v=YzS-BvHeSGE)      |

### 实战演练篇（已更新完成）

| 课时  | 内容                              | 视频链接                                                                                                          |
| ----- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 08    | 基于 Transformers 的 NLP 解决方案 | [Bilibili](https://www.bilibili.com/video/BV18N411C71F) \| [YouTube](https://www.youtube.com/watch?v=WRBPd86T1Fc) |
| 09    | 命名实体识别                      | [Bilibili](https://www.bilibili.com/video/BV1gW4y197CT) \| [YouTube](https://www.youtube.com/watch?v=3xQR-7sly_I) |
| 10 上 | 机器阅读理解（过长截断策略）      | [Bilibili](https://www.bilibili.com/video/BV1rs4y1k7FX) \| [YouTube](https://www.youtube.com/watch?v=-rzKZIpELOk) |
| 10 下 | 机器阅读理解（滑动窗口策略）      | [Bilibili](https://www.bilibili.com/video/BV1uN411D7oy) \| [YouTube](https://www.youtube.com/watch?v=oTlpbISOkaE) |
| 11    | 多项选择                          | [Bilibili](https://www.bilibili.com/video/BV1FM4y1E77w) \| [YouTube](https://www.youtube.com/watch?v=xHM1PjIihJs) |
| 12 上 | 文本相似度（交互策略）            | [Bilibili](https://www.bilibili.com/video/BV1Tm4y1J7EF) \| [YouTube](https://www.youtube.com/watch?v=SElN5_LqZls) |
| 12 下 | 文本相似度（匹配策略）            | [Bilibili](https://www.bilibili.com/video/BV13P411C7UD) \| [YouTube](https://www.youtube.com/watch?v=7zxNXBBDqwA) |
| 13    | 检索式对话机器人                  | [Bilibili](https://www.bilibili.com/video/BV1Lh4y117KJ) \| [YouTube](https://www.youtube.com/watch?v=gHOUoqqXb8I) |
| 14    | 预训练模型                        | [Bilibili](https://www.bilibili.com/video/BV1B44y1c7x2) \| [YouTube](https://www.youtube.com/watch?v=jHRo2qgtE7Y) |
| 15 上 | 文本摘要（T5）                    | [Bilibili](https://www.bilibili.com/video/BV1Kp4y137ar) \| [YouTube](https://www.youtube.com/watch?v=5AusJJbpWaA) |
| 15 下 | 文本摘要（GLM）                   | [Bilibili](https://www.bilibili.com/video/BV1CF411y7hw) \| [YouTube](https://www.youtube.com/watch?v=BK2wUNZZbRg) |
| 16    | 生成式对话机器人（Bloom）         | [Bilibili](https://www.bilibili.com/video/BV11r4y197Ht) \| [YouTube](https://www.youtube.com/watch?v=McE0XUG5Gw4) |

### 参数高效微调篇（已更新完成）

| 课时 | 内容                       | 视频链接                                                                                                          |
| ---- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 17   | 参数高效微调与 BitFit 实战 | [Bilibili](https://www.bilibili.com/video/BV1Xu4y1k7Ls) \| [YouTube](https://www.youtube.com/watch?v=ynBE40yVTSk) |
| 18   | Prompt-Tuning 原理与实战   | [Bilibili](https://www.bilibili.com/video/BV1Fu4y1C7tJ) \| [YouTube](https://www.youtube.com/watch?v=aAbVsm6tWIM) |
| 19   | P-Tuning 原理与实战        | [Bilibili](https://www.bilibili.com/video/BV17V411N7Ld) \| [YouTube](https://www.youtube.com/watch?v=xNC12IhNuw4) |
| 20   | Prefix-Tuning 原理与实战   | [Bilibili](https://www.bilibili.com/video/BV1Ru411g7Qa) \| [YouTube](https://www.youtube.com/watch?v=EYd-sJHXCio) |
| 21   | LoRA 原理与实战            | [Bilibili](https://www.bilibili.com/video/BV13w411y7fq) \| [YouTube](https://www.youtube.com/watch?v=-xVJtu9pyoA) |
| 22   | IA3 原理与实战             | [Bilibili](https://www.bilibili.com/video/BV1Y8411k7yD) \| [YouTube](https://www.youtube.com/watch?v=WOrHqOkMqxY) |
| 23   | PEFT 进阶操作              | [Bilibili](https://www.bilibili.com/video/BV1YH4y1o7rg) \| [YouTube](https://www.youtube.com/watch?v=KJljAinRXs8) |

### 低精度训练篇（已更新完成）

| 课时  | 内容                   | 视频链接                                                                                                          |
| ----- | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 24    | 低精度训练与模型下载   | [Bilibili](https://www.bilibili.com/video/BV1y34y1M7t1) \| [YouTube](https://www.youtube.com/watch?v=mWiXtVs9ZzY) |
| 25 上 | 半精度训练（LLaMA2）   | [Bilibili](https://www.bilibili.com/video/BV1CB4y1R78v) \| [YouTube](https://www.youtube.com/watch?v=Is4T8u1Astk) |
| 25 下 | 半精度训练（ChatGLM3） | [Bilibili](https://www.bilibili.com/video/BV1aw411M7Cv) \| [YouTube](https://www.youtube.com/watch?v=8SmlpNuY_pU) |
| 26    | 量化与 8bit 模型训练   | [Bilibili](https://www.bilibili.com/video/BV1EN411g7Yn) \| [YouTube](https://www.youtube.com/watch?v=XKImkaWv7-Y) |
| 27    | 4bit 量化与 QLoRA 训练 | [Bilibili](https://www.bilibili.com/video/BV1DQ4y1t7e8) \| [YouTube](https://www.youtube.com/watch?v=CY0jTExZlKE) |

### 分布式训练篇（已更新完成）

| 课时  | 内容                                 | 视频链接                                                                                                          |
| ----- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 28    | 分布式训练基础与环境配置             | [Bilibili](https://www.bilibili.com/video/BV1cK4y1z7Mv) \| [YouTube](https://www.youtube.com/watch?v=eNOoIlUCX6Q) |
| 29    | Data Parallel 原理与应用             | [Bilibili](https://www.bilibili.com/video/BV1qN4y1n7iG) \| [YouTube](https://www.youtube.com/watch?v=WiRpMjHL79s) |
| 30    | Distributed Data Parallel 原理与应用 | [Bilibili](https://www.bilibili.com/video/BV1wS421w7ug) \| [YouTube](https://www.youtube.com/watch?v=hoa-AIE_yxk) |
| 31    | Accelerate 分布式训练入门            | [Bilibili](https://www.bilibili.com/video/BV12Z421t74R) \| [YouTube](https://www.youtube.com/watch?v=eDaT_bBoiJ4) |
| 32 上 | Accelerate 使用进阶（上）            | [Bilibili](https://www.bilibili.com/video/BV1vq421F7Cf) \| [YouTube](https://www.youtube.com/watch?v=IhpuxmYoKgI) |
| 32 下 | Accelerate 使用进阶（下）            | [Bilibili](https://www.bilibili.com/video/BV1Lp421975B) \| [YouTube](https://www.youtube.com/watch?v=WmZ94u9QDME) |
| 33    | Accelerate + DeepSpeed               | [Bilibili](https://www.bilibili.com/video/BV1hb421E7WY) \| [YouTube](https://www.youtube.com/watch?v=Vegqv1PDboY) |

### 番外技能篇

| 内容                                     | 视频链接                                                                                                          |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 基于 Optuna 的 Transformers 模型自动调参 | [Bilibili](https://www.bilibili.com/video/BV1NN4y1S7i8) \| [YouTube](https://www.youtube.com/watch?v=ugiAW2ukZZw) |

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zyds/transformers-code&type=Date)](https://star-history.com/#zyds/transformers-code&Date)

---

## 请作者喝杯奶茶

如果课程对你有帮助，欢迎请作者喝杯奶茶～

![](./imgs/wx.jpg)
