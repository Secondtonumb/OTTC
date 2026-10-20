# 云端仍提示满额：复查结果

查询时间（UTC）：2026-10-19T05:45:14.723464+00:00

套餐：FREE_PLAN，storage_gigs=5，即标称 5 GB。

| 口径 | 字节 | 十进制 GB |
|---|---:|---:|
| Entity.storageBytes（cached=false；cached=true 相同） | 5667535280 | 5.667535 |
| 76 个 Project.storageBytes 之和 | 1450121616 | 1.450122 |
| 两者差额 | 4217413664 | 4.217414 |
| ottc-timit-reproduction 当前 Project.storageBytes | 259485900 | 0.259486 |
| ottc-english-multiphoneset 当前 Project.storageBytes | 422454943 | 0.422455 |

再次验收：73 个已批准删除的云端版本全部维持删除状态，188 个原有保留版本仍在，546 个本地文件再次通过 MD5。没有重复删除或扩大清理范围。

确认事实：账号汇总与逐项目容量之和不一致；账号汇总仍超过 5 GB，逐项目合计已低于 5 GB。

推断：现有证据更支持删除后的汇总统计、后台回收或配额计量尚未同步。不能仅靠这两个字段证明具体后台故障，也不能保证刷新期限。先前将账号总量与两个项目明细相减后直接归因给其他项目占用，混用了不同步的字段，应以此复查纠正。

官方文档：Forge 会保留删除后的 artifacts、run files 和 runs/history 7 天，可联系支持协助恢复。该文档没有承诺配额何时更新，因此不能把 7 天保留期当成准确的配额刷新时间。

https://docs.coreweave.com/products/wandb/platform/hosting/hosting-options/multi_tenant_cloud

当前不建议继续删除其他实验结果来消除这条警告。已经请求 cached=false 的只读用量查询，不能强制平台重算计量。

## 可提交给支持的说明（仅草稿，未发送）

Organization: kevingenghaopeng--org; entity: kevingenghaopeng-; free plan storage_gigs=5.
We deleted 73 evaluation artifact versions (4,137,553,755 logical bytes) and independently verified they remain deleted. All 188 protected versions remain present.
The account still reports entity.storageBytes(cached:false)=5667535280, but the sum of Project.storageBytes across all 76 projects is 1450121616. The difference is 4217413664 bytes. The storage-limit warning remains active.
Please verify whether deleted artifacts are still counted during retention/garbage collection, and refresh/reconcile the account storage and quota counters. Please confirm when the warning will clear. No further data deletion is requested.

Contact from the official Forge documentation: forge-support@coreweave.com.

Evidence: capacity_followup_evidence.json.

组织 teams 已核实：只有 kevingenghaopeng- 这一个 team，排除了同组织其他 team 占用的解释。
