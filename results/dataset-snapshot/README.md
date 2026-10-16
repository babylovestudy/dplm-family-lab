# 可离线复现的冻结数据快照

本目录保留原实验使用的 4,464 条 UniProt 原始查询记录和 PF00194.27 HMM，总数据文件约 1.38 MB。数据库未来更新不会改变这些文件。该快照用于重跑原有清洗、HMM 筛选、精确聚类和划分，不改变选择标准或指标。

| 文件 | 来源与版本 | 许可 |
| --- | --- | --- |
| `raw/uniprot.fasta`、`raw/provenance.json` | UniProtKB 2026_03，2026-09-02 发布，2026-10-16 获取 | CC BY 4.0 |
| `raw/PF00194.hmm`、对应 provenance | Pfam PF00194.27，模型日期 2025-05-29，2026-10-16 获取 | CC0 1.0 |

许可核验依据、归属、修改说明和引用见 [LICENSES.md](LICENSES.md)。各文件来源、版本、字节数与 SHA-256 见 [manifest.json](manifest.json)。数据保留其来源许可，仓库的软件许可证不会替代这些许可。

## 恢复和重算

先安装项目的数据处理依赖。完整 HMM 重算需要 Linux/WSL 上的 pyhmmer，或可用的 HMMER；原实测版本为 Python 3.10.12、NumPy 1.26.4、Biopython 1.85、pyhmmer 0.12.3。依赖安装可以联网，以下数据恢复和准备过程不访问网络：

```bash
python -m pip install -e . numpy==1.26.4 biopython==1.85 pyhmmer==0.12.3
python results/dataset-snapshot/restore.py --verify-only
python results/dataset-snapshot/restore.py --prepare --report artifacts/snapshot-restore.json
```

Windows 环境可以先执行 `--verify-only` 或不带 `--prepare` 的安全恢复，再在已有 HMM 工具的环境中准备数据。脚本不会安装系统工具或修改训练配置。

要在独立空目录验证完整管道，可运行：

```bash
python results/dataset-snapshot/restore.py --destination artifacts/snapshot-rebuild-check --prepare --report artifacts/snapshot-rebuild-check.json
```

脚本逐一验证打包文件，恢复至目标目录的 `data/raw`，调用真实 `prepare(download=False)` 和跨集合 `audit`，最后核对三份 FASTA 的字节哈希以及规范化划分哈希。遇到内容不同的既有原始文件、既有不同划分或已冻结数据时，会拒绝覆盖；可选择另一个空目录。原有训练和正式实验文件不会被更改。

## 哪些哈希应相同

重算应保留 **2,942** 条序列、**1,719** 个连通分量，得到训练 **2,354**、验证 **294**、测试 **294** 条，并通过相同阈值下的跨集合检查。

规范化划分 SHA-256：

```text
b6e07bd3c5809492f1ecbf73ccf8f39d168a2dffb70c0329e02b0d217ec3f330
```

它覆盖完整的科学输入签名，以及按 accession 排序的序列 ID、序列 SHA、长度、分量和分区。序列选择或分配发生变化时，该值随之改变。确切的序列字节和 FASTA 头部还会由三份 FASTA 的文件 SHA-256 独立检查。

原始工作清单的字节 SHA-256 为：

```text
d0611fd93b331a12dab9407597cba246adad2b10115f86e1f26b314c32dbd345
```

新的工作清单包含新的创建时间和实测耗时，其文件字节哈希通常不同。这里不会把重新测量的报告冒充原始报告；复现一致性由规范化划分哈希和 FASTA 字节哈希验证。

## 实际离线复现检查

2026-10-16 已使用最终恢复脚本，在事先不存在的目标目录中执行真实 HMM 筛选、精确聚类、数据划分与审计。测试期间阻断 Python 网络 socket 连接，未导入 Torch；没有使用原来的 HMM 评分、配对缓存或已处理数据。完整准备和审计耗时约 **234.48 秒**。

重建的 **2,942** 条序列、**1,719** 个分量、三份 FASTA 字节哈希和规范化划分哈希全部一致。跨集合检查覆盖 **1,470,588** 对，安全筛选后精确检查 **43,304** 对，违规数为 **0**。Linux 上还实际检查了目录链接不能逃逸到目标目录外。

原始实测记录见 [validation.json](validation.json)，其中保留本次使用的脚本、数据处理源码、基础配置的 SHA-256 和依赖版本。这里的耗时属于一次实测，不是其他机器的运行时间保证。恢复脚本的自动检查还覆盖相同文件复用、拒绝覆盖不同原始文件、拒绝改写不同冻结划分，以及规范化哈希对科学字段变化的检测。

## 来源和使用边界

UniProt 快照来自原项目查询：

```text
(xref:pfam-PF00194) AND (length:[180 TO 256]) AND (fragment:false)
```

REST 返回的 TSV 被转换成按 accession 排序的 FASTA，头部保留 taxid 和 reviewed 标记，序列按 80 列换行；原始导出阶段没有改写残基。后续清洗和过滤由原管道执行。Pfam 响应只做 gzip 解压，HMM 文本未编辑。provenance 中的分页游标来自公开 UniProt API，不是访问凭据。

Pfam GA 与覆盖率支持序列符合该家族的统计特征；这不证明催化活性、折叠、表达或其他真实功能。快照也不证明序列未出现在基础模型的预训练数据中。
