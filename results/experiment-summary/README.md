# 最终实验完整性汇总

`report.md`、`summary.csv`、`summary.json` 是在远程正式流程结束后，由最终发布版 `dplm-lab report` 生成的原始输出，未手工改写。正式状态为 `complete`，缺失阶段为 0；模型、数据、生成及评测记录通过一致性核对。

汇总共 27 行，其中也保留先前的试跑和恢复验证；它们不是额外的正式训练重复。正式对照是 3 个训练种子、每臂 3 个生成种子，以及原模型和 3 个微调模型的独立测试去噪评测。完整正式证据在 [formal](../formal/index.json)，指标分析在 [analysis/formal](../analysis/formal/analysis.json)。

原始 Markdown 汇总使用统一列展示不同记录类型，缺少该列时显示 `pending`。例如训练记录没有生成命中率，去噪记录没有生成种子，生成评测没有去噪 CE；这些单元格不表示正式阶段缺失。完成情况以 `summary.json` 的 `formal_status`、`missing_expected_stages` 和各自类型的实际指标为准；便于阅读的对照表见 [技术报告](../technical-report.md)。

`auxiliary/` 保存这份汇总引用的 3 个工程试跑训练报告。其余逐行来源保存在 `results/pilot/` 和 `results/formal/`。`manifest.json` 将每行的原始路径、SHA-256 与发布位置对应起来；保留报告中的原始工作目录，不据此推断本机存在相同目录。

运行 `python scripts/verify_evidence.py` 可检查发布文件的字节校验值。这是记录完整性检查，不会重新执行训练、全库最近邻搜索或生物学实验。
