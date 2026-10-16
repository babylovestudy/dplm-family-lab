# 上游公开范围与来源核查

核查日：2026-10-16。训练主线是**DPLM-1 150M 的家族定向生成式继续训练**。固定源码commit为`8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d`，权重revision为`49b7125a5d28c6418fcc2f3c4fe799352ac1488b`；不以主分支未来变化替代本次依据。

## 实际公开的训练链路

| 内容 | 官方固定源码/资源 | 核查结论 |
| --- | --- | --- |
| 模型与加噪/采样 | [DPLM模型](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/src/byprot/models/dplm/dplm.py) | 有`q_sample`、`compute_loss`、特殊token排除和迭代生成实现 |
| 实际训练task | [DPLM训练任务](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/src/byprot/tasks/lm/dplm.py) | 有训练/验证步骤、criterion调用与梯度记录 |
| 训练配置 | [150M配置](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/configs/experiment/dplm/dplm_150m.yaml) | 有模型、随机掩码、linear weighting、AdamW和学习率计划 |
| 优化执行入口 | [训练pipeline](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/src/byprot/training_pipeline.py) | 实际调用`trainer.fit`并支持检查点路径 |
| 150M权重 | [官方权重页](https://huggingface.co/airkingbd/dplm_150m) | 公开checkpoint；运行时必须另外记录实际模型revision |
| 预训练数据 | [UniRef50数据页](https://huggingface.co/datasets/airkingbd/uniref50)与[固定README](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/README.md) | 提供处理后的数据及下载说明；不属于本项目必须下载的PF00194快照 |
| 原论文 | [ICML 2024论文](https://arxiv.org/abs/2402.18567) | 用于理解目标与实验设置，不把论文数字当作本项目结果 |

因此选择DPLM的依据不是“有模型权重”而是可读的训练与生成链路。公开代码仍不等于作者每一次内部实验记录、每个预处理环节和精确数据快照都已经逐项重现。本项目不做UniRef50规模的从零预训练。

## 本地复用与修改

本项目选择性保留DPLM-1网络、加噪和采样实现，移除与当前任务无关的注册器、DPLM-2、OpenFold等依赖，增加单卡工作流与审计。具体移植范围和兼容性修改见[本地来源说明](../src/dplm_family_lab/_vendor/SOURCE.md)。不能把本地应用层称为官方原封不动的训练pipeline。

本地训练使用生成式去噪目标，包含对零mask情况的安全处理。真实150M已完成100步pilot及42/43/44三个种子各1000步的正式训练，确认编码器更新和导出重载一致性；源码身份、固定权重revision及原始文件SHA已核对。[正式训练证据](../results/engineering/formal_training_summary.json)记录总wall time 32.66分钟，最佳验证loss依次为0.617952/0.619555/0.620384，峰值已分配显存依次为2.85/5.64/5.65 GiB。这里报告验证指标，不据此声称生成效果提升。

[真实150M恢复证据](../results/engineering/real_150m_resume.json)验证连续10步与第5步停止后恢复至第10步的模型及优化器张量逐位一致，调度器、采样器、随机状态与验证噪声一致。证据记录本地提前停止/验证边界修复与源码SHA；不把修改后的本地训练流程称为官方原封不动的pipeline，也不外推为跨硬件逐位复现保证。远程13项GPU集成测试已通过；本机当前全量96 passed、2 skipped（CUDA与Windows符号链接相关条件），最终版本数量以后续完整记录为准。

微型网络权重不是预训练150M模型，CPU demo也不是正式家族数据。正式12组生成评估仍在进行，完整协议及最终生成结果保持pending；训练完成不能替代该对照，更不能证明结构或酶活。

## 代码、权重、数据分别记录授权

上游[源码LICENSE](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/LICENSE)为Apache-2.0；本地移植保留其版权头和许可证。该判断不能自动延伸为对所有外部权重和数据的重新授权。

核查时`airkingbd/dplm_150m`模型卡显示缺失/空的YAML授权元数据，未据此确认独立权重授权；发布或分发权重前应保存实际版本的许可说明，必要时向权利方核实。本项目代码开源不等于需要把外部大权重复制进仓库。

PF00194序列来自[UniProt](https://www.uniprot.org/)，HMM来自[InterPro/Pfam PF00194](https://www.ebi.ac.uk/interpro/entry/pfam/PF00194/)。数据快照应保留原站点链接、查询、下载日期、版本信息、hash和适用条款。参考[UniProt使用许可](https://www.uniprot.org/help/license)及[InterPro数据下载说明](https://www.ebi.ac.uk/interpro/download/)；如当次下载返回不同条款，以实际资产核查为准。

## 生物学范围

PF00194是α型碳酸酐酶家族的profile。本项目用180–256 aa、非fragment、HMM覆盖要求来选择受限子集；全长蛋白、结构域和家族命中不是同义词。HMM检测回答序列是否具备家族profile特征，不能证明表达、折叠、催化效率、稳定性或用途。

DPLM已在大量公开序列上预训练，PF00194样本可能已出现于预训练语料。80/80聚类拆分只控制本项目适配阶段的近邻泄漏风险，不能支持“从未见过该蛋白家族”或“无预训练重叠”的声明。
