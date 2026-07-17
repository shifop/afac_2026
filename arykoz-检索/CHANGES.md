# AFAC Retrieval V2 改动记录

## Phase 1：在线检索接入 V6.1

- 建立 `RetrievalPlan`、`EvidenceObligation`、`EvidenceSpanV2`、`RetrievalResultV2`。
- 以 V6.1 ClaimCompiler 为唯一正式 Claim 解析入口。
- 拆分文档路由和逐 Claim 证据检索。
- 新增索引感知主体校正，避免题干主体错误继承到选项。
- 新增多主体、跨文档、全文否定和表格证据义务。
- 新增有限轮次的缺口补检索。
- 新增证据预算，保留每个 Claim 和必需文档的证据。
- 保留旧 `retrieve()` 接口的兼容适配。

## Phase 2：Index V2

- 修复旧 `_merge_small_and_split_large()` 将正常章节成对合并的问题。
- 最大 Chunk 限制为2400字符。
- 生成持久化 BM25 postings，避免法规域首次查询在线重建倒排。
- 增加 TableAtom、structured_fields、source_hash 和 schema version。
- 直接从V2结构化字段、表格和 ValueRecord 构建 FactIndex。
- 法规重新抽取首页真实标题、处罚决定编号、文档角色和文档族。
- 规范正文优先于修订说明、MinerU重复件和附件解释。
- 研报使用独立 document family，避免通用标题导致误合并。
- 旧 V1/V2 scorer 和 annotation schema 不进入新索引。

## Phase 3：生产编排

- 接入 V6.1 Fact/Rule/LLM/Citation/AnswerAssembler。
- 正式模型路径只使用 Qwen。
- 增加 `run_v2_sharded.py`，按领域子进程隔离内存并合并输出。
- 增加 Token 聚合、运行日志和完整 retrieval trace。
- 增加纯模式和 baseline additive-only 守闸模式。

## 关键性能修复

- 修复旧分词函数在每个字符位置复制完整后缀导致的近二次复杂度。
- Chunk候选改为轻量打分后再创建 EvidenceSpan。
- 删除验证器每题强制 `gc.collect()` 造成的大堆扫描卡顿。
- FactIndex使用预建sidecar，避免每个运行分片重新构建40万事实。
