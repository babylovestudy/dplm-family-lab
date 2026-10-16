# 第三方数据许可、归属和引用

许可于 2026-10-16 依据下列资源自身发布的官方说明核验。本目录中的第三方数据保留其来源许可；项目软件的 Apache-2.0 许可不替代它们。

## UniProtKB 数据：CC BY 4.0

归属：**The UniProt Consortium**。资源：[UniProt](https://www.uniprot.org/)；本快照为 UniProtKB **2026_03**（发布日 2026-09-02）中指定查询的子集，获取日期为 2026-10-16。

[官方许可说明](https://www.uniprot.org/help/license)及其[官方 REST 版本](https://rest.uniprot.org/help/license)规定数据库中可受版权保护的部分适用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。REST 许可文档的 `lastModified` 为 2024-12-18。该许可允许在提供适当归属、链接至许可并注明修改的条件下分享和改编。请随再分发保留这些归属和许可信息，不暗示 UniProt 对本项目的认可。

适用文件：`raw/uniprot.fasta`、`raw/provenance.json`。

修改说明：从公开 UniProt REST TSV 响应选取查询结果，按 accession 排序，转换成带 taxid/reviewed 标记、80 列换行的 FASTA；未改写原始导出的残基内容。查询选择和后续数据清洗、HMM 过滤、聚类、划分是本项目的处理步骤。

保留来源声明：UniProt 不保证数据准确性，并说明部分数据可能涉及专利或其他权利。这里核实的是版权许可，不代表其他权利的无限制授权。本项目定位于研究、教学与复现，不把这些记录作为医疗建议；上述说明不为 CC BY 4.0 增加使用限制。

建议引用：The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2025*. Nucleic Acids Research 53(D1):D609–D617. [doi:10.1093/nar/gkae1010](https://doi.org/10.1093/nar/gkae1010)。引用依据为 [UniProt 官方 publications 页面](https://www.uniprot.org/help/publications)。

## Pfam HMM 数据：CC0 1.0

归属：**Pfam / InterPro、EMBL-EBI 及贡献该模型的整理者**。模型为 [PF00194.27](https://www.ebi.ac.uk/interpro/entry/pfam/PF00194/)，源于[官方单模型下载接口](https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00194?annotation=hmm)。获取日期为 2026-10-16；模型头部记录创建日 2025-05-29。完整 Pfam 数据库 release 未在原实验中捕获，因此不补造该版本号；确切模型版本和文件 SHA 已锁定。

[InterPro 官方数据许可说明](https://interpro-documentation.readthedocs.io/en/latest/license.html)和 [Pfam 官方文档](https://pfam-docs.readthedocs.io/en/latest/)明确给其提供的 Pfam 下载数据适用 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)。此处发布的是 HMM 数据，不涉及重新分发 InterProScan 或 HMMER 软件。

适用文件：`raw/PF00194.hmm`、`raw/PF00194.hmm.provenance.json`。

修改说明：对下载接口的 gzip 响应解压，未编辑模型文本。保留模型自身的版本、构建信息和 provenance，不暗示 Pfam、InterPro 或 EMBL-EBI 对项目的认可。

建议引用：Paysan-Lafosse T. et al. *The Pfam protein families database: embracing AI/ML*. Nucleic Acids Research. [doi:10.1093/nar/gkae997](https://doi.org/10.1093/nar/gkae997)。引用依据为 [Pfam 官方 citing 页面](https://pfam-docs.readthedocs.io/en/latest/citing-pfam.html)。
