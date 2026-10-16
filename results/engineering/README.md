# 真实模型训练与恢复验证

这些文件记录实际 NVIDIA L4 / BF16 运行，不能用来声称生物学功能已经验证。

- `formal_training_summary.json`：三个训练种子各 1,000 步，含梯度/参数更新、固定噪声验证、显存和导出重载证据。训练、验证和保存合计约 31.09 分钟；含加载和导出的三次顺序运行总时长约 32.66 分钟，均不等于全部实验或计费时长。
- `real_150m_resume.json`：真实 148,796,794 参数模型的连续 10 步与 5→10 步恢复对照。522 个模型张量、1,464 个优化器张量逐位一致，最大差为 0；RNG、采样位置、scheduler/scaler 和验证噪声均一致。原始运行脚本以 `verify_real_resume.py.txt` 保存。
- `formal_training_source.py.txt`、`formal_backend_source.py.txt`：正式训练实际使用的源码字节快照，与训练环境记录的 SHA-256 对应。
- `generation_backend_source.py.txt`、`generation_experiment_source.py.txt`：本次正式生成实际使用的加载和编排代码。
- `export_identity_migration.json`、`seal_recorded_exports.py.txt`：对已完成导出补充独立身份旁录的 CPU 检查；4 个导出的实际模型张量均匹配原重载证明，权重、训练报告、重载报告和来源文件字节没有改变。
- `final_pytest_remote.txt`、`final_platform_tests.txt`：最终 Linux / NVIDIA L4 环境全量 108 项测试全部通过，无跳过项；另存 CUDA tiny 模型训练及 Linux 符号链接保护的专项结果。这些测试属于工程验证，不能替代真实训练和生物学评估。
- `final_doctor.json`、`final_cli_help.txt`：实际安装的 `dplm-lab` 命令可直接调用，记录依赖版本、GPU 与最终源码 SHA。
- `final_inference_identity.json`、同名日志及 `final_inference_identity_validation.py.txt`：使用最终加载器在 CPU 上读取基础模型、pilot 和三个正式导出，核对实际权重文件 SHA、加载后完整张量摘要及独立导出身份，再进行有限值 FP32 前向计算；没有重新训练或生成序列。
- `final_validation_summary.json`、`final_package_extract.json`：最终验收摘要与受保护同步证据；252 个原始 runtime 配置、数据、训练和生成文件在解包前后保持不变。验收开始和结束时 GPU 均空闲。
- `wheel_rebuild_validation.json`：最终 wheel 已重建；包内源码逐文件匹配当前工作区及远程验收时记录的 SHA，许可证、上游来源说明、CLI 入口与 wheel 自身校验清单均核对通过。wheel 保留在忽略的 `dist/`，不包含模型权重。
- `index.json`：上述原始证据文件的字节校验值。运行 `python scripts/verify_evidence.py` 可核对。

正式完整训练之后修复了提前停止时的验证节奏：提前停止不再额外触发未计划的验证，恢复时严格核对 `eval_every`，官方权重加载还会持久保存原始来源 SHA。完整 1,000 步运行中的参数更新及验证时点没有改变。恢复对照使用修复后的代码，报告分别记录了两版源码的哈希。

最终发布版进一步核对导出权重文件、训练报告、张量摘要与生成记录的对应关系。当前生成期间没有更新远程源码；已有导出通过独立 CPU 检查新增 `export_identity.json`，不改写任何原始证据或模型权重。后续新训练会自动建立该身份记录，复用旧输出时也会核对实际权重文件，防止同名目录更新后混用实验。身份校验版本与本次执行版本的差异不涉及训练或采样数值计算。

最终加载器还显式选择与所核验文件一致的格式：目录同时存在 `pytorch_model.bin` 与 `model.safetensors` 时选择已核验的 bin；只有 safetensors 时选择已核验的 safetensors。回归测试刻意让两种文件的权重不同，确认实际加载张量与记录的文件 SHA 对应。本次真实模型缓存和导出目录均只有 bin 文件，实际目录清单见 `final_weight_format_inventory.json`，所以这一修复不会改变已完成科学实验所加载的权重；原始记录保留，未重跑训练或生成。

恢复脚本需要已准备的数据、固定版本模型缓存及原始 runtime 配置，且使用独立的 `runs/resume_validation` 命名空间。再次执行前选择全新的项目工作目录或新的验证目录，保留旧证据。`.txt` 只是存档格式；按原报告路径复制成 `artifacts/validation/verify_real_resume.py` 后，使用远程 GPU 环境执行。它会运行真实训练，不属于 CPU 快速检查。

实测各训练 seed 的峰值已分配显存约为 2.85、5.64、5.65 GiB。它们是同进程顺序运行的实测高水位，不是整卡占用，也不能将短时 profiling 的 2.83 GiB 当作所有正式运行的峰值。后续独立检查未发现残留模型对象，重复推理后已分配显存稳定；缓存释放前后的观测及随机状态检查保留在恢复报告中。
