# arykoz-检索 — AFAC 2026 赛题四 B榜 检索方案

> 独立检索模块，不含推理层（reasoning_engine.py / b_eval_pipeline_v2.py / llm_claim_verifier.py）。  
> 队友的 Whoosh 检索方案在项目根目录 `src/retriever.py` 等位置，本文件夹为 arykoz 的 Index V2 方案。

---

## 1. 索引体系（197MB）

| 索引 | 数据量 | 功能 |
|------|--------|------|
| BM25 × 5 域 | 327 文档 / 19,553 chunks | 全文检索，按 contracts/reports/insurance/regulatory/research 域隔离 |
| FactIndex | 401K 结构化事实 | 字段名→值匹配，含过度裁决门禁 |
| ValueIndex | 125K 条数值（6 vtype） | 单位归一化 + 数值范围查询（"营收 > 500亿"） |
| TableIndex | 18K 单元格 | 行列定位，表格数据提取 |
| EntityLexicon | 595 条（7 实体类型） | 实体名/别名→文档映射 |
| Clause KwIndex | 26 种法律关键词 | "不得/应当/豁免/违约/补偿"精准定位 |
| Structured Fields | 327 文档键值对 | 结构化属性匹配（issuer/year/subtype） |
| doc_identities | 327 文档元数据 | 文档标题/发行人/年份/文件路径 |

**没有的**：Embedding/向量检索、Relation 三元组索引（队友的 Whoosh 方案有）。

## 2. 检索流程

```
问题
  → ClaimCompiler：选项拆分为独立 Claim
    → ClaimAtomizer：Claim 拆分为可检索的原子查询
      → OptionQueryPlanner (B1-Lite)：每 Claim 生成 4 级 query variant
        → DocumentRouter：多通道 RRF 融合，确定相关文档
          → EvidenceRetriever：4 策略逐 Claim 检索证据 span
            → SufficiencyController：缺口检测 + 最多 2 轮补检索
              → Evidence Card：压缩去重
                → 交给推理层
```

### 2.1 查询扩展（B1-Lite OptionQueryPlanner）

每 Claim 生成 4 级 variant：
- **raw**：原文直接查询
- **identity**：提取实体名/文档身份 → EntityLexicon
- **value**：数值 + 单位提取 → ValueIndex 范围查询
- **domain**：领域关键词 + 条款模式 → Clause KwIndex

100 题实测：1,132 query variants routed，73 gap retry rounds。

### 2.2 多通道 RRF 融合（DocumentRouter）

7 通道 Weighted Reciprocal Rank Fusion，k=40：

| 通道 | Weight | 索引 |
|------|:---:|------|
| 精确匹配 | 1.8 | EntityLexicon + doc_identity |
| 结构化字段 | 1.5 | FactIndex structured fields |
| 数值 | 1.4 | ValueIndex 范围查询 |
| 表格 | 1.3 | TableIndex 行列定位 |
| 关键词 | 1.2 | Clause KwIndex |
| BM25 全文 | 1.0 | BM25 × 域 |
| 上下文扩展 | 0.8 | chunk 前后邻接 |

### 2.3 证据检索（EvidenceRetriever, 4 策略）

1. **精确匹配**：实体名称、文档身份逐字命中
2. **结构化字段匹配**：FactIndex 字段=值键值对
3. **表格原子定位**：TableIndex 行头+列头交叉定位
4. **Chunk 上下文扩展**：命中 chunk 的前后 chunk 扩展

### 2.4 文档绑定（DocumentReferenceResolver）

解决 B 榜无 doc_ids 时的"第一份/第二份文档"身份问题：
- **identity_probe**：通过发行体名称、债券简称等识别文档
- **deterministic_constraints**：年份、发行规模等约束匹配
- **candidate_pairs 评分**：多候选排序
- **ExplicitDocumentResolver**：题干显式文档ID解析

已知瓶颈：合同域 14 篇"向不特定对象发行可转换公司债券募集说明书"标题高度相似，identity_probe 无法区分。

### 2.5 证据压缩（Evidence Card）

- 去重：190 spans（100 题）
- 节省：1,140,712 chars
- 分域压缩率：insurance(25%) > contracts(20%) > reports(20%) > regulatory(9%) > research(5%)

## 3. 与队友 Whoosh 方案的差异

| 维度 | 本方案（Index V2） | 队友方案（Whoosh） |
|------|:---:|:---:|
| 全文检索 | BM25 × 5 域独立（手写） | Whoosh BM25F（jieba 分词） |
| 数值查询 | ValueIndex 125K（单位归一化） | ❌ 无 |
| 表格查询 | TableIndex 18K（行列定位） | ❌ 无 |
| 条款关键词 | 26 种法律关键词 | ❌ 无 |
| 实体匹配 | EntityLexicon 595（7 类型） | Entity Index（name/desc/entity_type） |
| 关系三元组 | ❌ 无 | Relation Index（3 倍增强：正向/反向/组合键） |
| 保险 MetaIndex | doc_identity 模糊匹配 | MetaIndex（insurer/contract_name → doc_id） |
| 向量/Embedding | ❌ | ❌（双方均无） |
| Chunk 链表 | chunk_index 映射获取 | prev_chunk_id / next_chunk_id（双向链表） |
| 查询扩展 | 4 级 variant（raw/identity/value/domain） | LLMExtractor（qwen2.5-7b → 结构化查询字段） |
| 多通道融合 | 7 通道 Weighted RRF | 4 通道 Min-Max 归一化 + 加权求和 |

## 4. 文件清单

```
arykoz-检索/
├── README_CN.md                       # 本文件
├── ARCHITECTURE.md                    # 架构说明
├── CHANGES.md                         # 变更日志
├── requirements.txt                   # pydantic
│
├── index_models.py                    # 索引数据模型（IndexV2, DocMetaV2, ChunkV2 等）
├── retrieval_models.py                # 检索数据模型
├── field_ontology.py                  # 字段本体（财务指标字段名映射和归一化）
├── text_utils.py                      # 文本处理工具（清洗/归一化/表格解析）
├── b_config.py                        # B榜配置文件（路径/阈值/预算）
├── document_registry.py               # 文档注册表（doc_id ↔ 文件名映射）
│
├── build_index_v2.py                  # ★ V2 索引构建主程序
├── chunker_v2.py                      # ★ V2 分块器（≤2400字符/边界修复）
├── fact_index_builder.py              # FactIndex 构建器
├── lightweight_fact_index.py          # 轻量事实索引（结构化字段快速匹配）
├── legacy_index.py                    # 旧版索引兼容层
├── migrate_legacy_index_v2.py         # 旧索引 → V2 迁移脚本
│
├── document_retriever.py              # ★ DocumentRetriever：文档级检索编排
├── document_router.py                 # ★ DocumentRouter：多通道 RRF 融合 + 文档路由
├── evidence_retriever.py              # ★ EvidenceRetriever：4 策略证据检索
├── claim_planner.py                   # ★ ClaimPlanner：解析问题选项，拆分为独立 Claim
├── claim_atomizer.py                  # ★ ClaimAtomizer：将 Claim 拆分为原子查询
├── plan_enricher.py                   # PlanEnricher：检索计划增强器
│
├── sufficiency_controller.py          # 证据充分性检查 + 缺口补检索（最多 2 轮）
├── budget_controller.py               # Token 预算控制（EvidenceBudgetController）
├── evidence_card.py                   # 证据压缩卡（去重 + 截断）
├── evidence_bundle_builder.py         # 证据打包器（Claim×文档轮转分配，Markdown 渲染）
├── option_query_planner.py            # B1-Lite Option 级检索查询分解
├── question_evidence_ledger.py        # 同题选项间证据共享
├── targeted_evidence_retry.py         # 定向证据补检索
│
├── document_reference_resolver.py     # ★ DocumentBinding：解决文档身份问题（44KB）
├── explicit_document_resolver.py      # 题干显式文档 ID 解析
│
├── insurance_rules.py                 # 保险域领域规则库（条款 ID/保单结构）
├── insurance_calculation_executor.py  # 保险公式执行器（退保/身故/医疗计算）
├── financial_metric_calculator.py     # 财务指标计算器（比率/同比/合计/单位换算）
│
└── scripts/
    ├── build_from_texts.sh            # 从全文本构建索引
    ├── build_from_legacy.sh           # 从旧索引构建 V2
    ├── run_blind_validation.sh        # 盲检索验证
    └── verify_release.sh              # 发布验证
```

## 5. 运行方法

### 构建索引
```bash
cd arykoz-检索
PYTHONPATH=".:." python3 build_index_v2.py \
  --texts-dir ../03_PyMuPDF提取原文/extracted_texts/ \
  --output b_index_v2.pkl
```

### 盲检索验证
```bash
bash scripts/run_blind_validation.sh
```

### 在代码中使用
```python
from document_retriever import DocumentRetriever
from claim_planner import ClaimPlanner

retriever = DocumentRetriever("b_index_v2.pkl")
compiler = ClaimPlanner()

claims = compiler.compile_options(question, options, domain, answer_format)
result = retriever.retrieve_v2(
    qid=qid, question=question, options=options,
    domain=domain, claims_by_option=claims,
)
# result.claim_evidence[claim_id] → List[EvidenceSpan]
# result.doc_ids → 排序后的相关文档 ID 列表
```
