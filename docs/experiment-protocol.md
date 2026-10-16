# PF00194生成式适配实验协议

本协议先固定比较规则，再查看正式测试结果。主张只限于“家族profile特征、序列多样性与训练近邻关系”；没有湿实验，不作活性提升或新酶发现声明。真实150M的100步pilot和正式三种子/12组生成对照均已完成，已完成结果见[技术报告](../results/technical-report.md)；下面保留复现实验的方法与条件。

## 完整流程入口

完成下述数据制备与审计后，`dplm-lab run-experiment --config configs/pf00194.yaml --pilot`编排100步训练、baseline/适配模型各32条配对生成与评估。在新工作目录用公开L4配置运行`dplm-lab run-experiment --config configs/pf00194-l4.yaml`，执行三训练种子各1000步和12组每组256条的正式协议；相应汇总命令为`dplm-lab report --config configs/pf00194-l4.yaml --output-dir artifacts/report`。

本次原始远程运行使用只将`model.source`替换为已核验本地缓存的`configs/pf00194-l4-runtime.yaml`，恢复和汇总也必须保持原始配置。公开profile与原始runtime的区别及重建前提见[远程GPU说明](remote-gpu.md#本次原始运行与公开配置)。单个工作目录只承载一套正式协议；不要将两份配置依次写入同一`outputs/formal`。

下面保留单阶段命令供理解和排查。手工生成例子的目录不属于编排器正式布局；即使手工凑齐数量，也不自动形成完整stage journal。初学者复现整套协议优先使用`run-experiment`，不要依次把所有教学例子写进已有正式目录。

## 1. 数据与拆分

数据来源为UniProt中的PF00194注释记录。配置限定非fragment、长度180–256 aa，并以PF00194 HMM进行覆盖筛选；保留原始记录、查询、下载时间、来源版本与hash。长度上限是计算预算选择，会排除部分家族成员，不代表完整家族分布。

```bash
dplm-lab fetch --config configs/pf00194.yaml
dplm-lab prepare-data --config configs/pf00194.yaml
dplm-lab audit-data --config configs/pf00194.yaml
```

已有真实快照和必需资产时可用`prepare-data --config configs/pf00194.yaml --offline`；offline不应凭空生成真实数据，也不能用教学数据填补缺失资产。

清洗只接受约定的标准氨基酸，去掉精确重复并保留排除原因。使用Biopython全局配对（match=1、mismatch=-1、gap open=-2、extend=-0.5），identity为相同残基数/全部比对列，双向coverage为两侧都有残基的列数/各自完整长度；端部gap也计入比对列。以identity≥0.8且双向coverage≥0.8的配对连边，通过无损剪枝和连通分量聚类，再以cluster为单位按配置目标80/10/10分配train/validation/test。该流程不需要MMseqs2。聚类不可分割，因此实际序列条数不一定精确满足比例。

正式审计至少检查：ID/序列重复、split间cluster重叠、长度和词表、HMM覆盖、cross-split近邻identity及覆盖。80/80只控制该阈值下的近邻，不排除低于阈值的同源性。发现越界近邻或工具失败必须留下证据并修复制备过程，不能静默改阈值。

**进入训练的条件：** 每个split非空且可解释；审计通过；来源与hash可追溯。数据过少或cluster过少时先报告事实和调整计划，不能凭空承诺足够样本，更不能换成合成序列后仍标PF00194。

## 2. 训练与检查点

初始化为真实`airkingbd/dplm_150m`，源码固定commit见上游审计；配置已锁定权重revision`49b7125a5d28c6418fcc2f3c4fe799352ac1488b`，运行报告还须核对实际载入版本。模型本体参与反向传播，在家族训练集上继续优化加权掩码去噪目标，不训练回归头，也不使用活性标签。

起始设置：microbatch=1、梯度累积16、学习率1e-5、最多1000个优化步、梯度检查点、自动选择兼容精度，验证/保存间隔100步；最终以运行保存的配置为准。一次优化步包含多次microbatch；不要把它与生成时的100次采样迭代混淆。

先完成20步短试跑，确认模型身份、有限损失、梯度、参数变化、检查点恢复和保存后重载。先复制`configs/pf00194.yaml`为`configs/pf00194-smoke.yaml`，仅将`train.output_dir`改为`runs/pf00194-smoke/seed_42`，以免占用正式seed42目录。训练/验证损失用于诊断；检查点选择仅依据预先固定的验证去噪指标，不能依据测试集或挑选生成样本。

```bash
dplm-lab train --config configs/pf00194-smoke.yaml --max-steps 20 --seed 42
```

正式训练种子为42、43、44。`run-seeds --pilot`若使用，只执行seed42、最多100步的预算试跑；它不能冒充完整三种子实验。需要改变学习率等协议时另存配置，在看测试结果之前记录理由及试验次数。

```bash
dplm-lab train --config configs/pf00194.yaml --seed 42
dplm-lab train --config configs/pf00194.yaml --seed 43
dplm-lab train --config configs/pf00194.yaml --seed 44
```

`--resume`使用命令输出中报告的实际训练检查点，保持原始数据、输出目录与包括`max_steps`在内的协议。已经完成最大20步的smoke不能改为1000步继续；恢复练习可用不改变目标最大步数的`train.stop_after_steps`临时停止字段。短试跑、pilot和正式运行分开，避免把恢复训练与重新初始化混为一谈。

## 3. 配对生成对照

比较原始模型与各适配模型，生成种子101、102、103，每个种子256条。训练种子差异与生成种子差异分别保留；对每个适配模型，均与相同生成种子的baseline配对。两组必须使用相同长度列表、条数、采样策略`gumbel_argmax`和迭代预算100。

采样的两个臂一致限制为20种标准氨基酸，额外屏蔽ESM null等非标准词表项；训练logits保持原有词表。该规则在pilot与正式实验前已固定，生成manifest保留alphabet policy。虽然复用官方迭代调度与gumbel策略，本项目采样输出并非完全未修改的官方实现；这项预先定义的输出约束不是事后按HMM分数选择性过滤。

先让baseline从训练集长度分布取样，生成的`length_schedule.json`再交给适配模型；不从测试集选取生成长度。下面以一对seed演示，`MODEL_PATH`应替换为训练输出中报告的真实导出目录或受支持检查点。

```bash
dplm-lab generate --config configs/pf00194.yaml --output-dir outputs/baseline/seed_101 --seed 101 --count 256
MODEL_PATH="/absolute/path/reported-by-train"
dplm-lab generate --config configs/pf00194.yaml --model-path "$MODEL_PATH" --output-dir outputs/finetuned/train_42/seed_101 --seed 101 --count 256 --lengths-file outputs/baseline/seed_101/length_schedule.json
```

生成目录包含`sequences.fasta`、`manifest.json`和`length_schedule.json`。逐项核对manifest，不能只看FASTA条数。没有有效长度文件时先解决问题，不用新抽样的另一组长度冒充配对。

pilot可以先生成32条检查流程，使用如`outputs/pilot/baseline/seed_101`的独立目录；其指标和长度计划与正式256条实验分开。失败、非法或空输出保留在记录中；不因HMM低分而删掉样本再计算“命中率”。

## 4. 指标与统计

```bash
dplm-lab evaluate --config configs/pf00194.yaml --fasta outputs/baseline/seed_101/sequences.fasta --output-dir outputs/evaluation/baseline_101
dplm-lab evaluate --config configs/pf00194.yaml --fasta outputs/finetuned/train_42/seed_101/sequences.fasta --output-dir outputs/evaluation/train_42_gen_101
dplm-lab denoise-eval --config configs/pf00194.yaml --model-path "$MODEL_PATH"
```

正式seed42的导出模型位于`runs/pf00194/seed_42/export_best`，训练检查点为同目录的`best.pt`和`last.pt`；仍以该次训练报告为准。独立smoke与pilot位于各自目录。`denoise-eval`会在配置的`evaluation.output_dir`写入`denoising_baseline.json`或`denoising_adapted.json`并拒绝覆盖；评估另一个模型时使用一份只改变该输出目录的评估配置，不要删除旧结果再重跑。

| 评估面 | 应报告内容 | 能支持与不能支持的结论 |
| --- | --- | --- |
| 基本有效性 | 请求/成功/失败条数、合法字母、长度符合率 | 流程与序列格式；不能证明生物学功能 |
| 家族profile | HMM命中与覆盖、分数分布、明确阈值和分母 | PF00194家族特征；不是催化活性 |
| 多样性 | 精确唯一率、生成集内相似性或聚类、氨基酸组成 | 是否塌缩或高度重复；不等于功能多样性 |
| 训练近邻 | 最近训练序列identity与ID、精确重复数；必要时另查配对覆盖 | 适配数据记忆/新颖性线索；不是全数据库新颖性证明 |
| 去噪 | 相同数据/时间步/随机噪声下的加权掩码去噪loss | 受控去噪比较；不是真实完整序列perplexity |
| 资源 | GPU、实际精度、优化步、峰值显存、耗时及失败 | 当前设置的成本；不能外推到任意GPU |

HMM工具及命中阈值以实际实现、版本和manifest为准，并在报告中明确写出。没有正式HMM后端结果时将相关指标标为未完成，不能用随机分数冒充正式结果。正式相似性统一使用项目规定的全局配对定义，不混用近似局部检索或不同identity分母。

评估器的HMM比例以“已输出且只含20种标准氨基酸的非空序列”为条件分母，保留其中的重复与超长/过短序列。报告必须同时给出原始请求数、失败数、非法数，以及`family_pass / 原始请求数`的整体产出率，防止条件命中率掩盖生成失败；不要把未输出序列或非法序列悄悄当成从未请求过。

按每次训练/生成运行报告原始值，再报告均值和区间。配置的1000次bootstrap只能描述其重采样单位下的不确定性；生成序列可能相关，不能把数百条序列当作数百次独立训练。三个训练种子仍然很少，避免强显著性或跨家族泛化声明。只有种子不同但长度配对的对照才能减少长度混杂；相同种子也不保证跨硬件位级一致。

如果HMM命中提高同时最近训练邻居相似性上升、唯一率下降，应优先讨论记忆/塌缩；如简单基线并不更差，如实报告。预训练UniRef50可能包含本家族及测试序列，本协议不宣称排除全部预训练重叠。

## 5. 实验报告由学习者填写

```text
状态：待运行 / 短试跑 / pilot / 正式
问题与事先固定的指标：
数据来源、版本、hash、清洗后数量与各split cluster数：
模型来源revision、上游commit、训练/生成种子、完整命令：
实际GPU、可用显存、软件、精度、训练成本：
验证选择规则与最终检查点：
每个运行的指标及分母、baseline差值、失败数量：
最高分与失败案例的最近训练邻居：
局限、未验证事项和下一步：
```

仓库应保存可复查的表格、manifest、日志与代表性失败案例。学习者须能独立解释数据与每个指标；自动跑完命令不等于已掌握实验设计。

## 6. 正式结束后读取结果

在完成实验的机器上，使用与该次运行一致的配置汇总。公开L4 profile的新运行使用：

```bash
dplm-lab report --config configs/pf00194-l4.yaml --output-dir artifacts/report
```

本次原始远程运行把`--config`改为`configs/pf00194-l4-runtime.yaml`。这份机器专用配置未随Git发布；只阅读已公开的报告不需要拥有它。重做完整性检查则需同时保留原始配置、数据和被引用产物，不能把公网模型ID临时替换进旧run。

1. 先看`outputs/formal/experiment_status.json`的阶段记录，再看`artifacts/report/summary.json`。只有`formal_status: complete`且`missing_expected_stages: []`，才表示三次1000步训练、12组256条配对生成评估和4组测试去噪的已有产物及hash检查完整；它不是生物学有效性结论。
2. 在`summary.csv`筛选`source_type=training`，核对42/43/44各一份正式报告、1000步及资源成本。这些行的`denoising_loss`是最佳验证loss。
3. 筛选`source_type=generation`，正式区域应有3组baseline和9组适配模型。对相同生成种子作差；保留各训练种子的结果，不把9组适配生成当成9次独立训练。
4. 筛选`source_type=denoising`，核对baseline和3个适配模型共4组相同test数据、固定噪声种子的评估；这里的loss才是固定噪声测试去噪CE。
5. 回到每行`source_json`读取完整区间与分母。生成明细在`outputs/formal/<arm>/seed_<seed>/evaluation/`：`metrics.json`是汇总，`per_sequence.csv`含HMM命中与最近训练邻居，`pairwise_diversity.csv`含生成集配对距离。模型与长度计划在上一级`manifest.json`、`length_schedule.json`中。
6. 把最终结论、失败案例与局限填入README“正式结果”表和自己的实验报告，再决定哪些事实可以写进简历；最终指标未出齐时保持pending。

如果显示partial，先读`missing_expected_stages`和`formal_validation_issues`，核对是否尚未运行、文件未同步、配置不一致或hash变化。只下载一张汇总表不能在另一台机器重做完整性核查；保留原机器完整产物，或连同被引用的数据与结果一起同步。
