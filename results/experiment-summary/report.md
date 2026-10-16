# DPLM Family Lab：已有结果汇总

结果状态：**available**；正式协议状态：**complete**（3次正式训练已记录）。

只汇总磁盘上实际存在的结果，不把tiny、smoke或pilot升级为正式实验；缺失值显示pending。
只有正式状态记录、三次1000步训练、12组256条配对生成评估与四组测试去噪的元数据和hash一致，才标记complete。

## 正式协议尚缺阶段

无；预期阶段及来源一致性检查已通过。

## 运行与成本

| 类型 | 阶段 | 来源 | 状态 | 步数/目标 | 实际权重revision | 秒/本会话 | 峰值GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training | pilot | runs/pf00194/pilot_seed_42/training_report.json | complete | 100/100 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 193.69 | 2.7817 |
| training | formal | runs/pf00194/seed_42/training_report.json | complete | 1000/1000 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 604.91 | 2.8484 |
| training | formal | runs/pf00194/seed_43/training_report.json | complete | 1000/1000 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 627.6 | 5.6426 |
| training | formal | runs/pf00194/seed_44/training_report.json | complete | 1000/1000 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 632.75 | 5.6476 |
| training | smoke | runs/pf00194/throughput_mb4_seed_42/training_report.json | complete | 20/20 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 20.129 | 2.8342 |
| training | smoke | runs/resume_validation/control_10/training_report.json | complete | 10/10 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 15.457 | 2.8319 |
| training | smoke | runs/resume_validation/resumed_10/training_report.json | complete | 10/10 | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 12.113 | 2.8308 |
| generation | baseline | outputs/formal/baseline/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 216.52 | 1.1557 |
| generation | baseline | outputs/formal/baseline/seed_102/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 215.69 | 1.1565 |
| generation | baseline | outputs/formal/baseline/seed_103/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 223.59 | 1.1583 |
| denoising | baseline | outputs/formal/baseline/test_denoising/denoising_baseline.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |
| generation | formal | outputs/formal/train_seed_42/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 193.51 | 1.1557 |
| generation | formal | outputs/formal/train_seed_42/seed_102/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 195.56 | 1.1557 |
| generation | formal | outputs/formal/train_seed_42/seed_103/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 194.22 | 1.1557 |
| denoising | formal | outputs/formal/train_seed_42/test_denoising/denoising_adapted.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |
| generation | formal | outputs/formal/train_seed_43/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 193.6 | 1.1557 |
| generation | formal | outputs/formal/train_seed_43/seed_102/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 194.39 | 1.1557 |
| generation | formal | outputs/formal/train_seed_43/seed_103/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 196.27 | 1.1557 |
| denoising | formal | outputs/formal/train_seed_43/test_denoising/denoising_adapted.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |
| generation | formal | outputs/formal/train_seed_44/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 200.01 | 1.1557 |
| generation | formal | outputs/formal/train_seed_44/seed_102/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 194.44 | 1.1557 |
| generation | formal | outputs/formal/train_seed_44/seed_103/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 194.59 | 1.1557 |
| denoising | formal | outputs/formal/train_seed_44/test_denoising/denoising_adapted.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |
| generation | pilot | outputs/pilot/baseline/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 72.207 | 1.1316 |
| denoising | pilot | outputs/pilot/baseline/test_denoising/denoising_baseline.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |
| generation | pilot | outputs/pilot/train_seed_42/seed_101/evaluation/metrics.json | complete | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | 95.85 | 1.1316 |
| denoising | pilot | outputs/pilot/train_seed_42/test_denoising/denoising_adapted.json | recorded | — | 49b7125a5d28c6418fcc2f3c4fe799352ac1488b | pending | pending |

## 指标

| 来源 | 有效N | GA率 | HMM家族率 | 请求总体产出率 | 重复率 | 配对距离 | 最近训练identity | 去噪CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/pf00194/pilot_seed_42/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.66657 |
| runs/pf00194/seed_42/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.61795 |
| runs/pf00194/seed_43/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.61955 |
| runs/pf00194/seed_44/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.62038 |
| runs/pf00194/throughput_mb4_seed_42/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.70335 |
| runs/resume_validation/control_10/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.70622 |
| runs/resume_validation/resumed_10/training_report.json | pending | pending | pending | pending | pending | pending | pending | 0.70622 |
| outputs/formal/baseline/seed_101/evaluation/metrics.json | 256 | 0 | 0 | 0 | 0 | 0.79678 | 0.26635 | pending |
| outputs/formal/baseline/seed_102/evaluation/metrics.json | 256 | 0 | 0 | 0 | 0 | 0.7995 | 0.26953 | pending |
| outputs/formal/baseline/seed_103/evaluation/metrics.json | 256 | 0 | 0 | 0 | 0 | 0.79258 | 0.26872 | pending |
| outputs/formal/baseline/test_denoising/denoising_baseline.json | pending | pending | pending | pending | pending | pending | pending | 0.72171 |
| outputs/formal/train_seed_42/seed_101/evaluation/metrics.json | 256 | 1 | 0.99609 | 0.99609 | 0 | 0.41605 | 0.57093 | pending |
| outputs/formal/train_seed_42/seed_102/evaluation/metrics.json | 256 | 1 | 0.98438 | 0.98438 | 0 | 0.41442 | 0.57464 | pending |
| outputs/formal/train_seed_42/seed_103/evaluation/metrics.json | 256 | 1 | 0.98828 | 0.98828 | 0 | 0.42164 | 0.57484 | pending |
| outputs/formal/train_seed_42/test_denoising/denoising_adapted.json | pending | pending | pending | pending | pending | pending | pending | 0.63236 |
| outputs/formal/train_seed_43/seed_101/evaluation/metrics.json | 256 | 1 | 0.99219 | 0.99219 | 0 | 0.41128 | 0.56223 | pending |
| outputs/formal/train_seed_43/seed_102/evaluation/metrics.json | 256 | 1 | 1 | 1 | 0 | 0.41486 | 0.5661 | pending |
| outputs/formal/train_seed_43/seed_103/evaluation/metrics.json | 256 | 1 | 0.99609 | 0.99609 | 0 | 0.41152 | 0.566 | pending |
| outputs/formal/train_seed_43/test_denoising/denoising_adapted.json | pending | pending | pending | pending | pending | pending | pending | 0.63645 |
| outputs/formal/train_seed_44/seed_101/evaluation/metrics.json | 256 | 1 | 0.98047 | 0.98047 | 0 | 0.3889 | 0.56974 | pending |
| outputs/formal/train_seed_44/seed_102/evaluation/metrics.json | 256 | 1 | 0.98828 | 0.98828 | 0 | 0.38507 | 0.57186 | pending |
| outputs/formal/train_seed_44/seed_103/evaluation/metrics.json | 256 | 1 | 0.99219 | 0.99219 | 0 | 0.37793 | 0.57449 | pending |
| outputs/formal/train_seed_44/test_denoising/denoising_adapted.json | pending | pending | pending | pending | pending | pending | pending | 0.63482 |
| outputs/pilot/baseline/seed_101/evaluation/metrics.json | 32 | 0 | 0 | 0 | 0 | 0.8043 | 0.26508 | pending |
| outputs/pilot/baseline/test_denoising/denoising_baseline.json | pending | pending | pending | pending | pending | pending | pending | 0.72171 |
| outputs/pilot/train_seed_42/seed_101/evaluation/metrics.json | 32 | 0.3125 | 0.3125 | 0.3125 | 0 | 0.75692 | 0.32987 | pending |
| outputs/pilot/train_seed_42/test_denoising/denoising_adapted.json | pending | pending | pending | pending | pending | pending | pending | 0.68994 |

精确训练匹配率及完整字段见summary.csv；各来源JSON保留原始区间与分母，不在此把区间重新解释成显著性。

## 统计与生物学边界

- This summary reads existing files; it does not rerun or independently authenticate an experiment.
- Tiny/synthetic fixtures, smoke tests and pilot runs are not formal 150M results.
- HMM GA/coverage measures family sequence compatibility, not activity, folding, expression or stability.
- HMM rates are conditional on valid emitted sequences; overall yield additionally requires the requested count.
- Sequence bootstrap intervals describe sampling uncertainty, not independent training replicates; related sequences and only three training seeds limit inference.
- The 80/80 adaptation split does not rule out distant homology or foundation-model pretraining overlap.
- Fixed-noise weighted denoising CE is not exact sequence likelihood or perplexity.
- Training elapsed time is for the reported session, not necessarily all resumed sessions or total billing.

## 来源与SHA-256

| 来源 | SHA-256 |
| --- | --- |
| model_cache/dplm_150m/family_lab_provenance.json | 82334a33e2ba1f3269dbdf228e6248b11882bc46b8c6e56e3f46000e8c904784 |
| outputs/formal/baseline/seed_101/evaluation/metrics.json | 232c34c0e42826ceca54e01048d329192960c127942fd2b9e11c1f6056181078 |
| outputs/formal/baseline/seed_102/evaluation/metrics.json | 63fcb6d473ea0d25a692dd118eb4f25f8a0e726693756af13bf233844ad46b0b |
| outputs/formal/baseline/seed_103/evaluation/metrics.json | 403241391917f85c957dce46721ec5e8fcc9f277c268148be7386d607986726b |
| outputs/formal/baseline/test_denoising/denoising_baseline.json | 6eca8ea22e524ee50821ae8ac5eb9f06b691ae575b2e07ab1b4b2501bdf6e8c0 |
| outputs/formal/train_seed_42/seed_101/evaluation/metrics.json | 0b10685a82e38b56c42928e60a448d33436abe0f13fde56070cc09f1b8f3eff9 |
| outputs/formal/train_seed_42/seed_102/evaluation/metrics.json | 49437c3e1be1146757163f30d444ae6124d72cf005524ebc874bb986f69b344a |
| outputs/formal/train_seed_42/seed_103/evaluation/metrics.json | 259f58703d398ae9b668796ad71103485ea0583c1bd24585e7fd15f16c5f2d98 |
| outputs/formal/train_seed_42/test_denoising/denoising_adapted.json | 1796452b953d91ee80188852a7644163a4a0855386f3446b6705bfe0ce59fe25 |
| outputs/formal/train_seed_43/seed_101/evaluation/metrics.json | 0d16343cc0724f33102bacd920db524cc75e4242d63a85226339be3f9162c0b6 |
| outputs/formal/train_seed_43/seed_102/evaluation/metrics.json | 9c60122daa43d48e8c559f4352cc311479be96f2a5e4edf7c800b50909343abf |
| outputs/formal/train_seed_43/seed_103/evaluation/metrics.json | 6a0f2c6ca87cc8695bce187a8e0f17f31a81146852ef9a09dd5b58bca3de16dd |
| outputs/formal/train_seed_43/test_denoising/denoising_adapted.json | 84b457292f85272cbee2cbaaf0fc75e9e5d41f05e0377f64e505176eed45310d |
| outputs/formal/train_seed_44/seed_101/evaluation/metrics.json | e361adf416364b50f50cf5fc2b747a9668cfb2abfd7bd1fa080562ef2bd78b4a |
| outputs/formal/train_seed_44/seed_102/evaluation/metrics.json | c497e773e2768f79fd89938760815580dbf697b71e1910a70d231177faae2510 |
| outputs/formal/train_seed_44/seed_103/evaluation/metrics.json | ff913e5d992a7d87f9ea65661f69a219e9ef47507368f25f73cb47fc5608da50 |
| outputs/formal/train_seed_44/test_denoising/denoising_adapted.json | f06e94e580ba5a6538c5819c6c5a96ba895946192b1945c0c76ded2b262066bf |
| outputs/pilot/baseline/seed_101/evaluation/metrics.json | 74daa5d3b9d684685769d3f5a4ffba205a8cbb65dbff44bafa24e8e27fa354ec |
| outputs/pilot/baseline/test_denoising/denoising_baseline.json | 98aa45c5df933605c8b6d96a3c62aa089e1052495a7e71c8934ed3f03e849b79 |
| outputs/pilot/train_seed_42/seed_101/evaluation/metrics.json | 69493fd743a1b770467465de54ca705837cd3918c6d9e36599d48636cf901612 |
| outputs/pilot/train_seed_42/test_denoising/denoising_adapted.json | f149dc015e9d5f8a63aa8dcebde2633cb18dfcfd5b99e6f8b8efa07237274ee7 |
| runs/pf00194/pilot_seed_42/training_report.json | 646ae552aa524b300d849786f2b8e4690bbc0c2113cd4105eaa50b7857653d12 |
| runs/pf00194/seed_42/training_report.json | b2b0b2ffcc9532795b000cde0ebcf950b3d8a90b3d0f469878de59a873bc4208 |
| runs/pf00194/seed_43/training_report.json | 2de8cf50b184a7a75c0dbc55c6c04a19bd4932d9c7768f98f5acbd9df8bf03db |
| runs/pf00194/seed_44/training_report.json | 2f176163e710a03ad5074b052b4c0a9c64b595df5eca46d123e026763b668113 |
| runs/pf00194/throughput_mb4_seed_42/training_report.json | 649a7e374493c6b15c74dea2cb11deea42ae1e4bca8e62e6cbdd3eccae76e830 |
| runs/resume_validation/control_10/training_report.json | 208b62820fb407050f9fd9327cd23baf581d88ebd8912c245606e40f76e0734b |
| runs/resume_validation/resumed_10/training_report.json | ddb9d437cb3a252aaf2c1ac9e10a025ee2e2871968d8f3424528f451af9adf38 |
| runs/pf00194/pilot_seed_42/config.json | 314fa2c7f124831dd324701ac2bc6c185650ffa3098cf1608afaeb19ab5668d3 |
| runs/pf00194/pilot_seed_42/export_best/export_identity.json | 133a9507eb275b8b46101d79203c569df2ba4df03e884ed34edf5d8d12b7d79d |
| runs/pf00194/pilot_seed_42/export_best/roundtrip.json | f9699eb1e7b52ea3913d36e20ebc693adcca41fcfb5f6d20ca18b64acf2f938a |
| runs/pf00194/pilot_seed_42/export_best/family_lab_provenance.json | 9bcbe4cf3b17ff55db1c9b87c1b2602d518ef8cbcd438e5a21a596d8f8ecc151 |
| runs/pf00194/seed_42/config.json | dcec58c1f87121b3c9bb6231fcca17cb449d5a8329cfdd6582e9987e287294c9 |
| runs/pf00194/seed_42/export_best/export_identity.json | 71a884a53033bdc6190b4ead7d1b9c90740e9d717c4cf4bc5d9be4ac7011c994 |
| runs/pf00194/seed_42/export_best/roundtrip.json | 520e087592a5c3d14dff1c06fa7fc5b57cee386a544031961fcdf69f6b8a4b74 |
| runs/pf00194/seed_42/export_best/family_lab_provenance.json | 71f69d84aef51bb09b1cd348024786cb9da6a611c690982806ecfd98e7cf3cee |
| runs/pf00194/seed_43/config.json | 89426cec38dd675e66913ae712666e790767fc309a1eb604d9f8d540679366f4 |
| runs/pf00194/seed_43/export_best/export_identity.json | 106319ccf7fc59a35369a338c87c0cf4e98c31611e899318aa3799692e3e480e |
| runs/pf00194/seed_43/export_best/roundtrip.json | c384a5983252422ad761334add2bf0719e222ee31fdfaa791bea9c26d849c7f7 |
| runs/pf00194/seed_43/export_best/family_lab_provenance.json | 349a7c1d46ca926e4666b9e8ce4d4399a1ceea234673425cfbf4cdcb1b7329fa |
| runs/pf00194/seed_44/config.json | 2cec3ee43c2e5f692f4df0883d9f5c6bd057050be90c207d1d38df76092e8e6f |
| runs/pf00194/seed_44/export_best/export_identity.json | 14cdd3a0da009569d31ff79df3720d6ecc69d35884e1de28af4f104e86aec6ed |
| runs/pf00194/seed_44/export_best/roundtrip.json | 47d3da87d51865c5650339308d76b0bdc4580748886813f4f8b1eda6d6f35c8b |
| runs/pf00194/seed_44/export_best/family_lab_provenance.json | 5e09ea67c97ab945ac65ce9a012038183c405780cb2886be1a3174cf13b0f409 |
| runs/pf00194/throughput_mb4_seed_42/config.json | 7b903a03e02b4109b6f48d530738aed778826e2ee2c7bf81a010c2c2136c7462 |
| runs/resume_validation/control_10/config.json | 5059be82e80ca1979fbddad6aa358abefd9a54f703a5c90baa7945f01f62ca5e |
| runs/resume_validation/resumed_10/config.json | 0fe347bb595b5fe1aed8590dd7adcbd075c938491067266bc9bba98913738e0e |
| outputs/formal/baseline/seed_101/manifest.json | d665746f48179b327ee6f04c7afe0058338cf2b031dfccc8306a4aabd8c87e7d |
| outputs/length_schedules/seed_101_n256.json | d522d2ebbf0203970afde15d597c8e096a1849f52e3599bfbea3d614fe52c612 |
| outputs/formal/baseline/seed_102/manifest.json | d116a2d1c80dd50528a6a1342b5559f82388bc06f185d8f139513154b8517333 |
| outputs/length_schedules/seed_102_n256.json | beeb64ce92e6c19166011bf55ee7c4e005a3722cdfa911010863439e17a8d170 |
| outputs/formal/baseline/seed_103/manifest.json | 6017cf078285941e275a92c4b3a37a2d37d041bdc98ff2c080eff9013439f741 |
| outputs/length_schedules/seed_103_n256.json | fa7f959554f36c30c82c7ebed5b6513ee4f02d2ebba12c2e91a2d4a81699c9cf |
| outputs/formal/train_seed_42/seed_101/manifest.json | 2172f5dae57fc92dc4daae7a972b1dd53c7059def40c56ca356305231fb68b63 |
| outputs/formal/train_seed_42/seed_102/manifest.json | 65ff8c8454fcaf7ecce75b0e15d5223900792f2bd2664a5c9f80466fe1566343 |
| outputs/formal/train_seed_42/seed_103/manifest.json | fd384cbd3ac2062421ccd807d0c278f7c84a4bc41ccef535636774f386e369e9 |
| outputs/formal/train_seed_43/seed_101/manifest.json | a0ff168eafbac5e3270566e2c2c6d722d007ffb0a1bd9df221351932f7018f32 |
| outputs/formal/train_seed_43/seed_102/manifest.json | f4deb320e623ca832a40da12c4024e2867ee52e5ff8d1d13ca9621f9998fc05f |
| outputs/formal/train_seed_43/seed_103/manifest.json | 6787e9c2cc1576faf1a58cb627a3425d8d36e277f1f3a15fdf415d02f1f10225 |
| outputs/formal/train_seed_44/seed_101/manifest.json | a13f6a309b41766a5af25a3e3ea1b5a43492793d2b227c0120271b7059963b8b |
| outputs/formal/train_seed_44/seed_102/manifest.json | 75e580ad00319ac0e38525583c13a1f71423f48e90a16bea9ab12ca970c30dcc |
| outputs/formal/train_seed_44/seed_103/manifest.json | f3d2f73bf33061d4a0bd81fba25b799d95f428506c5c3100f6807782368eb9e2 |
| outputs/pilot/baseline/seed_101/manifest.json | e5aa3dd116db9d02c3e8a7ab3814f01acf7f0d90ca8f8e3d53d6e3d4b258b1ec |
| outputs/length_schedules/seed_101_n32.json | 80c226752707d6e5ebb5e8c4cce4f882d03fb4c4280e2caefec7798b5bfebce8 |
| outputs/pilot/train_seed_42/seed_101/manifest.json | 2362778384f1e6368b8965d36308b4e0f618114d437af46fd8ae3f64a1b8d2c5 |
| outputs/formal/experiment_status.json | 063f9ceb2bdee9cb3b51aa142e9e7b1e710aa85fd1cadc0dc2e5907b1e96f560 |
| data/processed/training_ref.json | e13423a8bef928de34df009f54989f65f7603af3d99f04b30188313a47fe4bb5 |
| data/processed/split_manifest.json | d0611fd93b331a12dab9407597cba246adad2b10115f86e1f26b314c32dbd345 |
| outputs/formal/baseline/seed_101/length_schedule.json | d522d2ebbf0203970afde15d597c8e096a1849f52e3599bfbea3d614fe52c612 |
| outputs/formal/baseline/seed_102/length_schedule.json | beeb64ce92e6c19166011bf55ee7c4e005a3722cdfa911010863439e17a8d170 |
| outputs/formal/baseline/seed_103/length_schedule.json | fa7f959554f36c30c82c7ebed5b6513ee4f02d2ebba12c2e91a2d4a81699c9cf |
| outputs/formal/train_seed_42/seed_101/length_schedule.json | d522d2ebbf0203970afde15d597c8e096a1849f52e3599bfbea3d614fe52c612 |
| outputs/formal/train_seed_42/seed_102/length_schedule.json | beeb64ce92e6c19166011bf55ee7c4e005a3722cdfa911010863439e17a8d170 |
| outputs/formal/train_seed_42/seed_103/length_schedule.json | fa7f959554f36c30c82c7ebed5b6513ee4f02d2ebba12c2e91a2d4a81699c9cf |
| outputs/formal/train_seed_43/seed_101/length_schedule.json | d522d2ebbf0203970afde15d597c8e096a1849f52e3599bfbea3d614fe52c612 |
| outputs/formal/train_seed_43/seed_102/length_schedule.json | beeb64ce92e6c19166011bf55ee7c4e005a3722cdfa911010863439e17a8d170 |
| outputs/formal/train_seed_43/seed_103/length_schedule.json | fa7f959554f36c30c82c7ebed5b6513ee4f02d2ebba12c2e91a2d4a81699c9cf |
| outputs/formal/train_seed_44/seed_101/length_schedule.json | d522d2ebbf0203970afde15d597c8e096a1849f52e3599bfbea3d614fe52c612 |
| outputs/formal/train_seed_44/seed_102/length_schedule.json | beeb64ce92e6c19166011bf55ee7c4e005a3722cdfa911010863439e17a8d170 |
| outputs/formal/train_seed_44/seed_103/length_schedule.json | fa7f959554f36c30c82c7ebed5b6513ee4f02d2ebba12c2e91a2d4a81699c9cf |

## 未读取或未验证项

- runs/pf00194/throughput_mb4_seed_42/training_report.json: adapted export identity is not verified: [Errno 2] No such file or directory: '/pyg-vepfs/public/jx/dplm-family-lab/runs/pf00194/throughput_mb4_seed_42/export_best/export_identity.json'
- runs/resume_validation/control_10/training_report.json: adapted export identity is not verified: [Errno 2] No such file or directory: '/pyg-vepfs/public/jx/dplm-family-lab/runs/resume_validation/control_10/export_best/export_identity.json'
- runs/resume_validation/resumed_10/training_report.json: adapted export identity is not verified: [Errno 2] No such file or directory: '/pyg-vepfs/public/jx/dplm-family-lab/runs/resume_validation/resumed_10/export_best/export_identity.json'
