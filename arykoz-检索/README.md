# AFAC Retrieval V2 + V6.1 Reasoning Integration

这是基于最新完整索引重构的 AFAC2026 金融长文本检索—证据—推理工程。

本版本已经完成三个阶段：

1. **Phase 1：完整旧索引接入 V6.1**  
   提供旧索引兼容适配、统一 Claim 编译、文档路由、逐 Claim 证据检索、全文扫描和证据充分性控制。
2. **Phase 2：Index V2 重建**  
   修复旧 Chunk 合并错误，重建受控 Chunk、表格原子、结构化字段、持久化 BM25 倒排和 FactIndex。
3. **Phase 3：生产编排**  
   接入 V6.1 三态推理、Qwen 局部裁决、引用校验、答案装配、Token 统计，以及按领域隔离的100题运行器。

旧 `v1v2_scorer.py` 未进入生产链。V1/V2 数值锚点和逻辑节点不再作为独立语义系统；其中仍有价值的能力已经重组进共享 Claim、结构化索引、字段本体和证据义务体系。

## 1. 推荐入口

生产运行推荐使用按领域隔离的编排器：

```bash
export DASHSCOPE_API_KEY="你的百炼API Key"
export QWEN_MODEL="qwen-plus"

python integration/run_v2_sharded.py questions.jsonl \
  --index b_index_v2.pkl \
  --output-dir output_pure
```

输出：

```text
output_pure/
├── answer.csv
├── evidence.json
├── retrieval_report.json
├── run_info.json
├── logs/
└── shards/
```

### baseline 守闸模式

```bash
python integration/run_v2_sharded.py questions.jsonl \
  --index b_index_v2.pkl \
  --output-dir output_guarded \
  --guarded \
  --baseline-csv baseline.csv
```

守闸模式遵循 additive-only：只允许可靠 SUPPORT 新增选项，不删除 baseline 选项。

## 2. 环境变量

```text
AFAC_PROJECT_ROOT      项目根目录
AFAC_INDEX_V2_PKL      V2索引路径
AFAC_FACT_INDEX_PKL    FactIndex路径；默认读取 <index>.fact_index.pkl
DASHSCOPE_API_KEY      Qwen API Key
QWEN_API_BASE          默认百炼兼容接口
QWEN_MODEL             默认 qwen-plus
INSURANCE_LLM_MODEL    保险域模型，默认与QWEN_MODEL相同
AFAC_TOKEN_BUDGET      单题证据上下文预算，默认10000
```

正式推理链只支持 Qwen。`openai` Python 包仅作为百炼 OpenAI-compatible API 的传输客户端。

## 3. 从最新完整旧索引迁移

```bash
python migrate_legacy_index_v2.py legacy_complete_index.pkl \
  --output b_index_v2.pkl \
  --export-dir index_v2_export

python fact_index_builder.py b_index_v2.pkl \
  --output b_index_v2.pkl.fact_index.pkl \
  --force
```

## 4. 从完整文本重新构建

```bash
python build_index_v2.py \
  --extracted-dir /path/to/extracted_texts \
  --extracted-dir /path/to/extracted_texts_v2 \
  --output b_index_v2.pkl \
  --export-dir index_v2_export
```

## 5. 兼容旧调用

旧代码仍可调用：

```python
from document_retriever import DocumentRetriever

retriever = DocumentRetriever("b_index_v2.pkl")
result = retriever.retrieve(
    question="...",
    options={"A": "...", "B": "...", "C": "...", "D": "..."},
    domain="financial_reports",
    answer_format="multi",
)
```

新代码应优先调用：

```python
result_v2 = retriever.retrieve_v2(
    qid="...",
    question="...",
    options=options,
    domain=domain,
    answer_format=answer_format,
    claims_by_option=precompiled_claims,
)
```

## 6. 关键设计

```text
V6.1 ClaimCompiler
→ IndexAwarePlanEnricher
→ RetrievalPlan / EvidenceObligation
→ DocumentRouter
→ ClaimEvidenceRetriever
→ SufficiencyController
→ EvidenceBudgetController
→ V6.1 FactExecutor / RuleExecutor / Qwen Judge
→ CitationValidator
→ AnswerAssembler
```

检索结束条件不是 Top1 分数，而是证据义务是否满足。

## 7. 测试

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python -m unittest -v integration.test_reasoning_engine
```

当前发布版：

- V2 检索/索引/集成测试：14项通过；
- V6.1 推理回归：18项通过；
- 合计：32项通过。

完整结果见 `VALIDATION.md` 和 `validation/BLIND_RETRIEVAL_100_REPORT.md`。

## 8. 重要边界

- 99.5% 是隐藏 A 榜 `doc_ids` 后的**目标文档召回率**，不是最终答题准确率。
- 当前环境没有配置 Qwen API，因此没有伪造100题端到端准确率。
- B榜真实结果仍取决于问题分布、Claim拆分、证据充分性、规则覆盖和Qwen局部裁决。
- 纯模式不会读取标准答案或 baseline；答案文件只能由独立评估流程使用。
