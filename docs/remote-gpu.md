# 远程单卡训练

已完成真实150M模型的100步训练pilot：Python 3.10、NVIDIA L4，训练报告记录193.69秒、峰值已分配显存2.78 GiB；编码器确有梯度及权重变化，CPU导出重载logits最大误差为0。源码身份、权重revision和原始文件SHA已验证。远程环境中的9项模型/训练/采样测试也已通过，微型网络测试不冒充150M训练结果。

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

仓库canonical默认仍为1×16。要采用4×4，另存实验配置并写入`train.micro_batch_size: 4`与`train.gradient_accumulation_steps: 4`，使用新的训练输出父目录。有效batch相同不代表随机噪声、padding和数值轨迹完全相同；不要在断点恢复时静默换配置。

数据已制备的当前快照拆分为2354/294/294条（train/validation/test）；数据定义、清洗与审计以数据报告为准。审计通过后可运行完整pilot：

```bash
dplm-lab run-experiment --config configs/pf00194.yaml --pilot
dplm-lab report --config configs/pf00194.yaml --output-dir artifacts/report
```

它执行100步训练、原始/适配模型各32条配对生成、测试去噪和生成评估。已通过的100步训练不等于这整套pilot已结束；通过`outputs/pilot/experiment_status.json`和已有结果汇总核对阶段。

正式协议使用训练种子42/43/44各1000步；baseline和3个适配模型各用101/102/103生成种子，共12组每组256条：

```bash
dplm-lab run-experiment --config configs/pf00194.yaml
dplm-lab report --config configs/pf00194.yaml --output-dir artifacts/report
```

编排器核对已完成产物后复用，并可从持久训练检查点恢复；中断的生成目录保留失败证据，按错误信息使用新命名空间。正式模式与pilot使用不同目录，不将pilot指标拼成正式结果。

## 断点、OOM与长任务

恢复时使用训练输出报告的真实检查点路径，`max_steps`必须与原始运行一致。下面假设原始运行目标就是1000步，在中途停止；路径是待替换占位符：

```bash
CHECKPOINT="/absolute/path/to/a/training/checkpoint"
dplm-lab train --config configs/pf00194.yaml --resume "$CHECKPOINT" --max-steps 1000 --seed 42
```

恢复使用相同数据快照、输出目录与训练协议；不能将原本最大20步的run用`--max-steps 1000`续跑，签名校验会拒绝。为了练习恢复，可在原配置保留正式最大步数、使用`train.stop_after_steps`提前停止于优化边界，再移除该临时停止字段恢复；先在独立smoke目录操作。不要把“只载入模型权重重新开始”称为恢复了优化器和随机状态。正式长任务可用平台任务管理器或tmux，保留日志，并在远程会话结束后核对进程状态。

若OOM，记录实际GPU、命令、长度、精度、batch和错误，再确认是否有其它进程占用。优先减小microbatch、保持梯度检查点并调整累积以维持有效batch。必须改变长度范围或目标时，生成新实验配置、重新制备/审计数据并重新跑基线，不能静默截断后继续比较。

## 取回与复现

取回配置快照、环境信息、数据清单及hash、训练日志、选定检查点、生成FASTA、长度列表、评估与失败记录。大型数据和模型留在受控存储，开源仓库提供取得方式与校验信息。保存模型来源与实际revision；上游源码commit不能代替权重revision。

最终报告分别填写“CPU已验证”“微型测试已验证”“真实150M短试跑已验证”“正式实验已完成”。目前前三类已有对应验证，100步运行是pilot；只有正式三训练种子与全部生成/评估产物齐备后，才能填写最后一项。
