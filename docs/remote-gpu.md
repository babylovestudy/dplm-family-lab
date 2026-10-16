# 远程单卡训练

真实150M模型已在Python 3.10、NVIDIA L4上完成42/43/44三个种子各1000步，含三次导出的总wall time为32.66分钟，编码器更新与CPU导出重载一致性均已验证。12组正式生成评估、4组test去噪和严格来源检查均已完成，结果见[技术报告](../results/technical-report.md)；复用已有训练的正式编排阶段wall time为44.05分钟，GPU生成与CPU评估重叠。当前Linux L4全量108项测试通过、无skip。此前100步训练pilot为193.69秒、2.78 GiB，单独标记，不混入正式结果。

L4总显存23034 MiB；初次检查时另有任务占用约7670 MiB。这是当时的设备状态，不是未来可用显存保证。PyTorch峰值已分配显存也不等于整张卡的总占用。下面保留可迁移的部署步骤，主机地址与账号由使用者自行配置，不写入公开文档。

## 目标环境

- Linux x86_64；以Python 3.10、PyTorch 2.2.0、CUDA 12.1 wheel为起始兼容环境。
- 优先16–24 GB显存的兼容NVIDIA GPU；“≤24 GB”只是约束，不代表任意更小显存都能完成训练。
- 需要能够下载Hugging Face模型和UniProt/InterPro数据；家族筛选使用HMMER或PyHMMER。
- 聚类、拆分与审计使用基础依赖中的Biopython严格全局比对、无损剪枝和连通分量，不依赖MMseqs2候选检索。

这套旧PyTorch配置不保证支持所有NVIDIA代际，尤其较新架构必须核对wheel支持范围。驱动报告的CUDA版本不等于安装的PyTorch CUDA runtime版本；不要只看到`nvidia-smi`成功就宣布训练环境可用。

## 上传项目与创建环境

在远程主机放置完整项目目录，保留数据、配置和版本记录。可克隆实际项目仓库：

```bash
git clone https://github.com/babylovestudy/dplm-family-lab.git
cd dplm-family-lab
```

已有工作目录时先确认版本及未提交内容，再同步所需文件；不以重新克隆覆盖实验目录。

以下在远程Linux终端执行，优先使用已配置的uv。示例创建独立`.venv-gpu`；已有实际训练环境时先记录它，不重复覆盖环境。

```bash
cd /path/to/dplm-family-lab
uv venv --python 3.10 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-gpu/bin/python -e '.[train,hmm]'
source .venv-gpu/bin/activate
dplm-lab doctor
```

没有uv时可用Python 3.10创建venv，再用对应解释器的pip执行同样的包版本安装。`.[hmm]`提供PyHMMER后，不必为了本项目额外安装MMseqs2或系统级HMMER。如果选择外部`hmmsearch`后端，按[HMMER官方说明](http://hmmer.org/)安装并确认当前PATH中的版本。

完成安装后记录`python --version`、`uv pip freeze --python .venv-gpu/bin/python`、`nvidia-smi`和HMM后端版本。在Python内确认PyTorch、CUDA可用性、设备名称、BF16支持与实际显存。先做最小GPU运算再加载真实模型；不在CPU上等待完整150M训练来“验证GPU”。

## 正式数据与短训练

```bash
dplm-lab fetch --config configs/pf00194.yaml
dplm-lab prepare-data --config configs/pf00194.yaml
dplm-lab audit-data --config configs/pf00194.yaml
```

先检查数据审计。为首次20步创建独立smoke配置：复制`configs/pf00194.yaml`为`configs/pf00194-smoke.yaml`，将`train.output_dir`改为`runs/pf00194-smoke/seed_42`，其余数据路径不变，再运行：

```bash
dplm-lab train --config configs/pf00194-smoke.yaml --max-steps 20 --seed 42
```

检查模型载入与参数数量、有效损失、梯度、检查点和恢复能力。分别记录加载、稳定训练、导出和生成的耗时，不把训练报告中的会话秒数当成端到端计费时长。smoke与正式运行目录分开，正式配置不从已完成20步的smoke检查点延长到1000步。

`mixed_precision: auto`应依据实际设备选择支持的精度，BF16并非所有GPU可用。先保持`micro_batch_size: 1`和梯度检查点；梯度累积16次表示多次小batch合并一个优化步，不等于扩散采样迭代16次。

## 已测硬件配置与完整实验

同一L4的20步profiling测得下列训练速度；均为短时测量，应结合实际日志复核，不外推为任意GPU或完整任务的保证。

| 配置 | 有效序列batch | 训练速度 | 峰值已分配显存 |
| --- | --- | --- | --- |
| microbatch=1，累积16次 | 16 | 约1.8秒/优化步 | 100步pilot记录2.78 GiB，非同一20步统计窗 |
| microbatch=4，累积4次 | 16 | 约0.508秒/优化步 | 20步profiling记录2.83 GiB |

三次正式1000步训练的实测结果如下；[原始训练汇总](../results/engineering/formal_training_summary.json)保存完整数值、模型来源和源码SHA。会话秒数包含训练、验证与检查点阶段；32.66分钟是从进程启动到三次导出结束的wall time，不包含后续生成评估或全部项目计费时间。

| 训练种子 | 最佳验证去噪loss | 训练/验证/检查点会话秒数 | 峰值已分配显存 |
| --- | --- | --- | --- |
| 42 | 0.617952 | 604.91 | 2.85 GiB |
| 43 | 0.619555 | 627.60 | 5.64 GiB |
| 44 | 0.620384 | 632.75 | 5.65 GiB |

峰值按该进程实际记录保留，不把短时profiling的2.83 GiB替换成完整实验峰值，也不将张量已分配量当作显卡总占用或最低显存需求。验证loss不等于测试loss或生成蛋白质量。

仓库canonical默认仍为1×16。公开的`configs/pf00194-l4.yaml`使用训练4×4、生成batch4与8个评估worker；下面命令用于新工作目录的复现。两份配置指向同一正式命名空间，需在开始前选定，不能交叉运行。编排器固定使用`outputs/formal`和`outputs/pilot`，另做硬件/超参数实验需使用独立项目工作目录，不能只改训练输出父目录。有效batch相同不代表随机噪声、padding和数值轨迹完全相同；不要在断点恢复时静默换配置。

数据已制备的当前快照拆分为2354/294/294条（train/validation/test）；数据定义、清洗与审计以数据报告为准。审计通过后可运行完整pilot：

```bash
dplm-lab run-experiment --config configs/pf00194.yaml --pilot
dplm-lab report --config configs/pf00194.yaml --output-dir artifacts/report
```

它执行100步训练、原始/适配模型各32条配对生成、测试去噪和生成评估。已通过的100步训练不等于这整套pilot已结束；通过`outputs/pilot/experiment_status.json`和已有结果汇总核对阶段。

正式协议使用训练种子42/43/44各1000步；baseline和3个适配模型各用101/102/103生成种子，共12组每组256条：

```bash
dplm-lab run-experiment --config configs/pf00194-l4.yaml
dplm-lab report --config configs/pf00194-l4.yaml --output-dir artifacts/report
```

编排器核对已完成产物后复用，并可从持久训练检查点恢复；中断的生成目录保留失败证据，按错误信息使用新命名空间。正式模式与pilot使用不同目录，不将pilot指标拼成正式结果。汇总也使用L4配置，因为生成batch等协议参数会参加完整性核查；错用canonical配置会被标为未完整匹配。

## 本次原始运行与公开配置

本次远程正式运行使用的原始配置是`configs/pf00194-l4-runtime.yaml`：训练/生成参数与公开L4 profile相同，仅`model.source`为`model_cache/dplm_150m`。缓存记录原始官方模型ID、固定revision与原始权重SHA；路径本身不是模型身份的证明。该runtime配置和大型缓存未纳入Git，不能在新克隆中假定它们已存在。

若要建立同样的离线来源，先取得固定revision的模型及其来源校验记录，核对官方原始权重SHA，再复制公开L4配置为runtime文件，仅将`model.source`改成实际缓存路径。在整个训练、恢复、生成和汇总期间保持这份配置不变。直接联网复现则使用公开L4 profile，无需改成缓存路径。不要把任意同名目录视为已验证的150M模型。

下面两行仅适用于保有本次原始配置与产物的远程工作目录；第二行读取已有结果，不触发训练：

```bash
dplm-lab run-experiment --config configs/pf00194-l4-runtime.yaml
dplm-lab report --config configs/pf00194-l4-runtime.yaml --output-dir artifacts/report
```

原始100步pilot也使用缓存，完整配置保存在该次`runs/pf00194/pilot_seed_42/config.json`中。学习者可直接读已发布的pilot证据，无需在现有pilot目录中重新执行公开配置；新pilot从独立工作目录开始。不要为使汇总通过而改写原始配置或报告。

## 断点、OOM与长任务

[真实150M恢复证据](../results/engineering/real_150m_resume.json)已比较同一环境下连续10步与“第5步停止、恢复至第10步”：522个模型张量和1464个优化器张量逐位一致、最大差0；scheduler、scaler、sampler、RNG和验证噪声也一致。它验证当前设置的优化器边界恢复，不保证换GPU或修改协议后仍逐位相同。证据同时记录了提前停止不触发计划外验证/最佳检查点的修复及源码版本，不能把它当作未修改的上游训练器。

恢复时使用训练输出报告的真实检查点路径，`max_steps`必须与原始运行一致。下面假设原始运行目标就是1000步，在中途停止；路径是待替换占位符：

```bash
CHECKPOINT="/absolute/path/to/a/training/checkpoint"
dplm-lab train --config configs/pf00194-l4.yaml --resume "$CHECKPOINT" --max-steps 1000 --seed 42
```

此例针对公开L4配置的新运行；恢复本次原始运行时把配置替换为其runtime文件。恢复使用相同数据快照、输出目录与训练协议；不能将原本最大20步的run用`--max-steps 1000`续跑，签名校验会拒绝。为了练习恢复，可在原配置保留正式最大步数、使用`train.stop_after_steps`提前停止于优化边界，再移除该临时停止字段恢复；先在独立smoke目录操作。不要把“只载入模型权重重新开始”称为恢复了优化器和随机状态。正式长任务可用平台任务管理器或tmux，保留日志，并在远程会话结束后核对进程状态。

当前恢复实现还要求checkpoint记录的**原始绝对`output_dir`完全一致**。把`last.pt`复制到任意新路径不会自动迁移run，改变目录会被明确拒绝；需保持原目录/挂载路径与原协议。这个限制不影响`export_best`的可移植推理，不能把两种交付混为一谈。

若OOM，记录实际GPU、命令、长度、精度、batch和错误，再确认是否有其它进程占用。优先减小microbatch、保持梯度检查点并调整累积以维持有效batch。必须改变长度范围或目标时，生成新实验配置、重新制备/审计数据并重新跑基线，不能静默截断后继续比较。

## 取回与复现

完整恢复需保留**整个run目录**，包括`last.pt`、选定的`best.pt`、`tokenizer/`、`validation_noise.pt`、`config.json`、环境记录和训练日志；不能只拷贝一个权重文件。还需保持上述原始绝对输出路径、相同数据快照与协议。远程现有文件可直接在对应工作目录使用，公开Git仓库不下载或分发多GB检查点。

`export_best/`用于可移植推理，应连同权重、模型配置、tokenizer、`roundtrip.json`、`export_identity.json`和`family_lab_provenance.json`一起保留；其身份校验还引用原training report的SHA，因此追溯时保留对应小报告。原始预训练模型缓存是固定SHA的来源权重，可以按固定revision重新获取，不能用另一套权重替换后仍沿用旧来源记录。

同时保存数据清单、生成FASTA、长度列表、评估与失败记录。源码commit不能代替权重revision。公开结果入口为[严格汇总](../results/experiment-summary/report.md)、[配对分析](../results/analysis/formal/analysis.json)和[正式证据索引](../results/formal/index.json)。

正式完成后先在远程项目目录运行与该次原始配置一致的`dplm-lab report`命令；本次使用上述runtime命令，公开profile的新运行使用公开L4命令。读取`summary.json`中的`formal_status`与`missing_expected_stages`。不要只看进程结束或最后一条“complete”。`report.md`用于阅读，`summary.csv`用于逐组比较，原始JSON用于核对分母、区间与来源；详见[结果读取清单](experiment-protocol.md#6-正式结束后读取结果)。

当前“CPU已验证”“微型测试已验证”“真实150M训练及恢复已验证”“完整正式实验已完成”均有对应记录；最后一项只表示完整协议产物及来源检查齐备，不能替代结构或酶活验证。
