# DPLM Family Lab：从读懂蛋白序列到真实生成式微调

这是一个面向生物 AI 开发工程学习的项目：使用 **DPLM-1 150M**，在 **PF00194（α 型碳酸酐酶）家族序列**上继续训练，比较原始模型和适配后模型生成序列的家族特征、多样性与近邻相似性。这里训练的是蛋白序列生成模型，不是活性回归模型。

项目仓库：[babylovestudy/dplm-family-lab](https://github.com/babylovestudy/dplm-family-lab)。

学习预算为 12 周、每周 10–15 小时。已有的生物统计与科学工作流经验将用于数据审计、实验设计和结果解释；Python、张量和梯度从小例子开始练习。12 周是工作安排，不是就业或模型效果保证。

## 当前状态

真实DPLM-1 150M已在NVIDIA L4上完成**100步训练pilot**：训练报告记录193.69秒、峰值已分配显存2.78 GiB，编码器梯度和权重更新均已验证；导出后在CPU重载的logits最大误差为0。源码身份、权重revision与原始文件SHA已核对。当前数据拆分为train **2354**、validation **294**、test **294**条。

CPU合成演示、18项基础测试及远程环境中的9项模型/训练/采样测试已通过；微型网络测试与真实150M运行分别记录。**100步pilot不代表三种子正式实验完成；尚未验证生成蛋白的结构或酶活。** CPU演示仍仅为合成教学数据。正式对照及生成指标以已有结果汇总为准，不预填效果提升。

同一L4上的20步硬件profiling表明：`micro_batch_size=4`、累积4次时约0.508秒/优化步、2.83 GiB；原始1×16配置约1.8秒/优化步。两者有效序列batch均为16。这是短时实测，不是完整实验耗时保证；仓库默认仍保留较保守的1×16，调整方法见[远程GPU说明](docs/remote-gpu.md)。

## 今天从这里开始

1. 阅读 [第一周任务](docs/week01.md)，先完成环境、FASTA 与张量练习。
2. 运行 [CPU 教学 notebook](notebooks/01_fasta_and_tensors.ipynb)。它不下载模型、不需要 GPU；PyTorch 部分可选。
3. 查看 [12 周路线](docs/learning-roadmap.md)，用每周的验收问题判断自己是否掌握。
4. 准备进入训练时，依次读 [模型原理](docs/model-guide.md)、[远程 GPU](docs/remote-gpu.md) 和 [实验协议](docs/experiment-protocol.md)。

## 安装与轻量检查

以下命令在项目根目录运行。CPU 学习环境只需 Python 3.10–3.12；远程训练采用单独环境。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m dplm_family_lab.cli doctor
.venv\Scripts\python -m dplm_family_lab.cli demo --output-dir artifacts/demo
```

激活环境后也可以使用 `dplm-lab doctor`。`doctor` 中 GPU 或外部工具缺失不等于 CPU 学习无法进行；正式家族筛选需要HMMER或`.[hmm]`提供的PyHMMER。聚类/拆分审计使用Biopython严格全局比对、无损剪枝与连通分量，不要求MMseqs2；真实训练使用远程GPU。检查输出应逐项阅读。

## 实验入口

正式数据与GPU流程需要联网和相应工具，运行前先看实验协议。`run-experiment`负责训练、配对生成、测试去噪与生成评估；`report`只读取已有结果，不触发训练。

```bash
dplm-lab fetch --config configs/pf00194.yaml
dplm-lab prepare-data --config configs/pf00194.yaml
dplm-lab audit-data --config configs/pf00194.yaml
dplm-lab run-experiment --config configs/pf00194.yaml --pilot
dplm-lab report --config configs/pf00194.yaml --output-dir artifacts/report
```

完整pilot使用一个训练种子、100步训练及baseline/适配模型各32条配对生成，并执行评估，使用独立pilot目录。已通过的100步训练只证明训练阶段，整个pilot是否完成需查看状态记录。

正式协议为3个训练种子各1000步，生成种子101/102/103与baseline及3个适配模型完整交叉：共12组、每组256条，复用长度计划。首次20步检查采用独立smoke配置；正式流程见[实验协议](docs/experiment-protocol.md)。

```bash
dplm-lab run-experiment --config configs/pf00194.yaml
dplm-lab report --config configs/pf00194.yaml --output-dir artifacts/report
```

汇总输出`artifacts/report/report.md`、`summary.csv`和带来源SHA的`summary.json`，严格区分tiny、smoke、pilot与正式结果；缺少真实结果时显示pending。

## 会产出什么

| 产出 | 应有的证据 |
| --- | --- |
| 对蛋白生成模型的理解 | 自己画出的加噪、去噪、训练和采样图；源码阅读笔记 |
| 可复现的开源项目 | 固定数据与版本、命令行工作流、审计报告、测试和失败记录 |
| 真实微调能力 | 编码器参数更新、有限损失与梯度、检查点及恢复训练记录 |
| 科学评估能力 | 基线对照、相同生成预算、HMM 与多样性指标、统计不确定性及边界 |
| 可写进简历的事实 | 可链接的代码、报告和真实测量值，见 [求职证据模板](docs/career-evidence.md) |

## 范围与来源

项目筛选 180–256 aa 的完整序列，以减少首个单卡实验的计算负担；这只是 PF00194 的受限子集。家族筛选和 HMM 命中不证明催化活性，也不保证折叠或实验可用。80% identity / 80% coverage 的聚类划分降低近重复泄漏，不能证明没有更远的同源关系或预训练语料重叠。

官方上游固定为 `bytedance/dplm@8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d`，权重为 `airkingbd/dplm_150m@49b7125a5d28c6418fcc2f3c4fe799352ac1488b`。源码、权重和数据授权需分别记录；本仓库的许可证不能覆盖外部资产。核查链接与公开范围见 [上游审计](docs/upstream-audit.md)。
