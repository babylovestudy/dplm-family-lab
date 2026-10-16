# 正式生成结果的补充诊断

本目录从已公开的 [完整正式结果](../../formal/) 重算记录核对、分位数、完全相同序列的合并计数和确定性案例。这些诊断在结果检查阶段定义，属于 **post hoc supplementary diagnostics**，不作为预先指定的主要终点；没有修改采样、阈值或原始评测。

默认重算只读既有 CSV、FASTA、metrics，不加载模型、不调用 HMMER，也不执行 GPU 计算：

```bash
python scripts/diagnose_generations.py --input-root results/formal --output-dir artifacts/diagnostics-reproduction
```

缺少任一组时，默认拒绝完整分析。只有显式加上 `--allow-incomplete` 才能输出标记为 incomplete 的暂时结果。正式数据的 12 组均有 **256 条有效标准氨基酸序列**，总计 3,072 条；重复序列保留在分母中。

## 完整结果

下表按每个模型的三个生成 seed 合并，nearest 分位数保留所有有效记录，包括重复序列。

| 模型 | 家族通过 / n | GA 通过但覆盖不足 | nearest 中位数 | nearest 95% 分位 | nearest 最大值 | 合并 unique / n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 / 768 | 0 | 0.279937 | 0.305360 | 0.324841 | 766 / 768 |
| train seed 42 | 760 / 768 | 8 | 0.577604 | 0.626062 | 0.702929 | 768 / 768 |
| train seed 43 | 765 / 768 | 3 | 0.570866 | 0.614450 | 0.703557 | 768 / 768 |
| train seed 44 | 758 / 768 | 10 | 0.578642 | 0.609940 | 0.700389 | 768 / 768 |

基线全部未通过 GA。三个微调模型的序列均通过 GA，21 条未达到单 domain 覆盖率要求。微调模型九组合并后为 **2,304 / 2,304 unique**，没有跨生成 seed 或跨训练 seed 的完全相同序列。所有模型的训练集精确拷贝数均为 0，所报 nearest identity ≥80% 和 ≥90% 的记录数均为 0。

每组内部重复数均为 0；基线跨生成 seed 有两条重复序列、共四条记录，因此合并 unique 为 766。其 ID、长度、字符组成和最长同聚物记录在重复案例表中。组成信息只描述观察到的序列，不能据此推断折叠或其他功能是否存在。

三个生成 seed 是同一模型的采样重复；不同模型也共享相同采样 seed。合并后的数量用于描述，不增加独立训练重复数。这里的独立训练 seed 数为 3。

## 文件与来源链

- [diagnostics.json](diagnostics.json)：完整聚合、方法、限制、案例和实际使用的 **36 个输入文件 SHA-256**；保留执行脚本 SHA-256。
- [group_checks.csv](group_checks.csv)：12 组的分母、GA、覆盖、nearest、重复计数核对。
- [arm_diagnostics.csv](arm_diagnostics.csv)：每个模型的 nearest 分位数、≥80% / ≥90% 比例及重复统计；JSON 另有只针对 family 通过记录的分布。
- [sequence_diagnostics.csv](sequence_diagnostics.csv)：全部 3,072 条记录的可追溯 ID、序列 SHA 和诊断字段。
- [examples.csv](examples.csv)：按确定性规则选取的最高 nearest、覆盖失败、普通通过或 GA 未通过案例。
- [cross_generation_seed_duplicates.csv](cross_generation_seed_duplicates.csv)：同一模型跨生成 seed 的完全重复案例。
- [all_adapted_pooled_duplicates.csv](all_adapted_pooled_duplicates.csv)：九个微调生成组的合并重复案例；本次只有表头，表示没有重复案例。

案例规则是最高 nearest、覆盖失败中最接近原阈值的一条，以及最接近相应记录集 nearest 中位数的普通通过 / GA 未通过序列；并列时按生成 seed 和 ID 排序。案例是补充说明，不据此调参或改变结果选择。

## 独立核对所报配对

默认脚本仅核对记录中的数值和标记。另一次运行使用已冻结的训练 FASTA，对每条生成记录独立核对了精确拷贝标记，并重新计算生成序列与 **所报训练参考 ID** 的全局配对 identity，3,072 条均在 1e-12 绝对及相对容差内一致。小型验证摘要见 [reported_pair_verification.json](reported_pair_verification.json)，其中包含训练参考、实现、执行脚本和输入文件哈希，没有收录训练 FASTA。

该核对验证所报配对的数值，没有重新遍历全部训练候选来证明其为全局最近邻。原评测的全局搜索算法及多进程一致性另有实现测试。恢复冻结训练数据后，可选执行：

```bash
python scripts/diagnose_generations.py --input-root results/formal --train-fasta data/processed/train.fasta --output-dir artifacts/diagnostics-named-pair-check
```

[冻结数据恢复说明](../../dataset-snapshot/)可重建相同的训练参考。默认纯记录模式与这次附加核对使用的 36 个公开输入文件哈希完全相同，聚合和案例也完全相同；默认报告会如实标记没有执行附加配对计算。

## 解释边界

nearest identity 和精确拷贝只针对微调训练集，不能证明基础模型预训练数据或其他数据库中不存在这些序列。家族 GA 与覆盖率支持 PF00194 序列兼容性；同一个 HMM 也用于训练数据完整性筛选，因此它是目标相关的 profile 指标。催化活性、折叠、表达和稳定性仍需独立验证。本诊断没有计算新的 p 值。
