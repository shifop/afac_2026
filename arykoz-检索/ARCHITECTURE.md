# AFAC Retrieval V2 架构说明

## 1. 核心原则

系统以原子 Claim 为中心，而不是以“整道题的一段上下文”为中心：

```text
Question
└─ Option
   └─ Claim
      └─ EvidenceObligation
         └─ EvidenceSpan
            └─ ClaimDecision
```

检索层负责履行证据义务，推理层只在证据充分时裁决。

## 2. 统一数据契约

### RetrievalPlan

包含：

- V6.1 预编译 Claims；
- 显式文档引用；
- 主体组和年份组；
- 每个 Claim 的 EvidenceObligation；
- 最大文档和证据补检索轮数。

### EvidenceObligation

支持：

- `DOC_IDENTITY`
- `SUBJECT_FIELD`
- `FIELD_VALUE`
- `YEAR_VALUE_PAIR`
- `CROSS_DOC_PAIR`
- `UNIVERSAL_MULTI_DOC`
- `FULL_DOCUMENT_ABSENCE`
- `TABLE_HEADER_ROW_UNIT`
- `RULE_CONDITION_EXCEPTION`
- `FORMULA_COMPONENTS`

### EvidenceSpanV2

保留：

- doc_id / chunk_id；
- 原文位置；
- section_path；
- table_id、行名、列头、单位；
- 匹配项；
- source_hash；
- 全文扫描状态。

## 3. 文档路由

`DocumentRouter` 使用独立通道做可解释融合：

- 显式 doc_id；
- 文档标题和法规全称；
- 主体/产品身份；
- 字段；
- 年份；
- 问题 BM25；
- 每个原子 Claim 独立 BM25；
- FactIndex/结构化身份资产。

关键机制：

- 每个 Claim 保留文档配额；
- 每个主体组保留文档配额；
- 保险市场简称做词法身份归一化；
- 法规先按文档族合并重复提取件，再优先规范正文；
- 研报即使标题同为“证券研究报告”，仍保持独立文档族；
- Top-K 动态扩展，不固定所有题只看5篇。

## 4. 证据检索

`ClaimEvidenceRetriever` 对每个 Claim 独立执行：

1. 结构化字段；
2. TableAtom；
3. Chunk轻量打分；
4. 每目标文档配额；
5. 只对入选候选扩展正文窗口；
6. 全文否定命题执行完整扫描。

避免先创建数千个大 EvidenceSpan 再排序，降低批量运行的内存与GC压力。

## 5. 充分性闭环

`SufficiencyController` 不看 Top1 margin，而检查：

- 目标文档是否覆盖；
- 多文档两侧是否都有证据；
- 主体、字段、年份是否齐全；
- 表头、指标行和单位是否齐全；
- 全文扫描是否完成；
- 公式组件和例外上下文是否齐全。

有缺口时只针对缺失项生成第二轮检索请求；达到轮数上限后保持 UNKNOWN。

## 6. Index V2

Index V2 包含：

- DocMetaV2；
- 受控 ChunkV2；
- EntityEntryV2；
- ValueRecordV2；
- TableAtom；
- structured_fields；
- 持久化 BM25 postings；
- document_family_id；
- source_hash 和 schema version。

旧 V1/V2 scorer 不再保留。

## 7. 与 V6.1 的接口

- V6.1 `ClaimCompiler` 是唯一正式 Claim 编译器；
- `claim_atomizer.py` 仅作为其规则回退；
- 检索结果通过 `retrieval_result_v2` 直接传入 V6.1；
- 旧 ChunkHit 只由兼容适配器生成；
- FactIndex、保险规则、数值执行和Qwen局部裁决继续由 V6.1 处理；
- `evidence_ledger` 和旧 `compose_answer()` 不在生产链。

## 8. 生产运行

推荐按领域分片：

```text
主进程
├─ insurance子进程
├─ regulatory子进程
├─ financial_reports子进程
├─ financial_contracts子进程
└─ research子进程
```

每个子进程加载一次 Index 和 FactIndex，完成一个领域后退出，主进程合并 answer/evidence/retrieval/token 信息。
