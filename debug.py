import logging
from src.retriever import Retriever
import os
import json
from openai import OpenAI
from pathlib import Path
import itertools

QUESTION_PROMPT = """
{question}

{options}
"""

ANSWER_PROMPT = {
    "res":"""
{doc}

你是一个应试专家，现在在做一个有一些争议的题目，请理解上述材料，解答下述问题({qtype})：

{question}

{options}


答题要求：
1. 对边界值的判断从宽处理，仅小数点的差别可以认为在范围内
2. 只要正负关系、限定实体、数值一致即可认为选项正确，不需要过度纠结选项中未提及，但文档中有的提及的范围
3. 选项与指标数值相关时，请将指标名称与指标的限定词分开理解，只要指标名称以及指标数值一致即可认为正确，忽略限定词的差异
4. 如果选择描述不清，对应原文一种情况正确，另一种情况错误，优先当作正确

请按以下格式输出，直接输出json格式数据，不需要添加额外内容，不需要用```包裹json数据
answer字段，直接输出正确选项
""",
    "fc":"""
{doc}

你是一个应试专家，现在在做一个有一些争议的题目（例如可能存在以偏概全，张冠李戴的问题），请理解上述材料，解答下述问题({qtype})：

{question}

{options}


答题要求：
1. 对边界值的判断从宽处理，仅小数点的差别可以认为在范围内
2. 只要正负关系、限定实体、数值一致即可认为选项正确，不需要过度纠结选项中未提及，但文档中有的提及的范围
3. 选项与指标数值相关时，请将指标名称与指标的限定词分开理解，只要指标名称以及指标数值一致即可认为正确，忽略限定词的差异
4. 请对观点提出人从宽处理，若选项中的观点与原文中观点一致，忽略主体的差异
5. 请从宽处理以偏概全，张冠李戴的问题
6. 有可能正确的选项优先判断为正确
7. 请特别注意合并前缀的财务业务指标，它是指把上市公司及其所控制的全部子公司，看作一个整体来反映财务状况和经营成果。

请按以下格式输出，直接输出json格式数据，不需要添加额外内容，不需要用```包裹json数据
answer字段，直接输出正确选项
""",
    "reg":"""
{doc}

请理解上述材料，解答下述问题({qtype})：

{question}

{options}


答题要求：
1. 对边界值的判断从宽处理，仅小数点的差别可以认为在范围内
1. 只要正负关系、限定实体、数值一致即可认为选项正确，不需要过度纠结选项中未提及，但文档中有的提及的范围

请按以下格式输出，直接输出json格式数据，不需要添加额外内容，不需要用```包裹json数据
answer字段，直接输出正确选项
""",
    "fin":"""
{doc}

你是一个应试专家，现在在做一个有一些争议的题目（例如可能存在以偏概全，张冠李戴的问题），请理解上述材料，解答下述问题({qtype})：

{question}

{options}


答题要求：
1. 对边界值的判断从宽处理，仅小数点的差别可以认为在范围内
2. 只要正负关系、限定实体、数值一致即可认为选项正确，不需要过度纠结选项中未提及，但文档中有的提及的范围
3. 如果原文中存在一种能与选项对上一致的情况，判定为正确
4. 如果选择描述不清，对应原文一种情况正确，另一种情况错误，优先当作正确

请按以下格式输出，直接输出json格式数据，不需要添加额外内容，不需要用```包裹json数据
answer字段，直接输出正确选项
""",
    "ins":"""
{doc}

请理解上述材料，解答下述问题({qtype})：

{question}

{options}


答题要求：
1. 请严格参照保险赔付范围判断选项

请按以下格式输出，直接输出json格式数据，不需要添加额外内容，不需要用```包裹json数据
answer字段，直接输出正确选项
""",
}


def select_chunk(total_chunks, ids2name):
    total_chunks.sort(key=lambda x:x.score, reverse=True)
    # 汇总并截断
    docs = {}
    used_chunks = []
    for chunk in total_chunks:
        if chunk.chunk.chunk_id not in used_chunks:
            used_chunks.append(chunk.chunk.chunk_id)
        else:
            continue
        if chunk.chunk.doc_id not in docs:
            docs[chunk.chunk.doc_id] = []
        docs[chunk.chunk.doc_id].append(chunk)

    for k,v in docs.items():
        v = sorted(v, key=lambda x:x.score, reverse=True)
        v_ = [_.chunk for _ in v if _.score>0.2]
        if len(v_)==0:
            v_ = [v[-1].chunk]
        docs[k] = v_

    # 构建可读性强文档
    used_titles = []
    md = []
    for doc_id, chunks in docs.items():
        md.append(f"# {ids2name[doc_id]}")
        chunks = chunks[:5]
        chunks.sort(key=lambda x: x.chunk_index)
        for chunk in chunks:
            section_path = chunk.section_path
            titles = section_path.split('/')
            if section_path not in used_titles:
                used_titles.append(section_path)
                md.append(f"## {section_path}")
            md.append(chunk.content)
    md = "\n\n".join(md)
    return md

def select_chunks_with_budget(total_chunks, ids2name, B, delta=0, min_score=0.0):
    """
    从候选片段中选择最优子集，保证每篇文档至少选一个片段，且总字符数 ≤ B + delta。
    
    total_chunks: 列表，每个元素应有属性 chunk (包含 chunk_id, doc_id, content, chunk_index, section_path) 和 score
    ids2name: 文档 id 到名称的映射
    B: 预算字符数
    delta: 允许的弹性量（总成本上限 = B + delta）
    min_score: 片段最低得分阈值，低于此分的片段会被过滤（但每篇文档至少保留一个）
    """
    max_cost = B + delta

    # 1. 去重并构造候选信息
    seen_ids = set()
    chunk_list = []
    for item in total_chunks:
        if item.chunk.chunk_id not in seen_ids:
            seen_ids.add(item.chunk.chunk_id)
            cost = len(item.chunk.content)  # 字符数近似 token 数
            chunk_list.append({
                'chunk': item.chunk,
                'score': item.score,
                'cost': cost
            })

    # 2. 按文档分组，应用最低分数阈值（并保证每篇至少一个候选）
    docs_candidates = {}   # doc_id -> list of candidates
    for cand in chunk_list:
        doc_id = cand['chunk'].doc_id
        if doc_id not in docs_candidates:
            docs_candidates[doc_id] = []
        docs_candidates[doc_id].append(cand)

    for doc_id in docs_candidates:
        candidates = docs_candidates[doc_id]
        # 过滤低分
        filtered = [c for c in candidates if c['score'] >= min_score]
        if not filtered:
            # 所有都不满足阈值，保留得分最高的一个
            best = max(candidates, key=lambda x: x['score'])
            filtered = [best]
        docs_candidates[doc_id] = filtered

    # 3. 如果某篇文档无任何候选（理论上不会，因为上一步保证了至少一个），抛出异常
    for doc_id in docs_candidates:
        if not docs_candidates[doc_id]:
            raise ValueError(f"文档 {doc_id} 无可用片段")

    # 准备基础候选：每篇文档取单位得分（score/cost）最高的 top-2 片段
    base_options_per_doc = []
    for doc_id in sorted(docs_candidates.keys()):  # 固定文档顺序，保证组合可复现
        cands = docs_candidates[doc_id]
        # 按单位得分降序
        cands_sorted = sorted(cands, key=lambda x: x['score'] / x['cost'], reverse=True)
        # 取前2个作为基础候选（如果只有一个，就只有一个）
        base_options_per_doc.append(cands_sorted[:2])

    # 穷举所有基础组合（文档数 ≈5，最多 2^5=32 种）
    best_base_combo = None
    best_base_score = -1
    best_base_cost = float('inf')

    for combo in itertools.product(*base_options_per_doc):
        total_cost = sum(c['cost'] for c in combo)
        total_score = sum(c['score'] for c in combo)
        if total_cost <= max_cost:
            if total_score > best_base_score or (total_score == best_base_score and total_cost < best_base_cost):
                best_base_score = total_score
                best_base_cost = total_cost
                best_base_combo = list(combo)

    # 最坏情况：所有基础组合都超预算，则强制用每篇文档成本最低的片段
    if best_base_combo is None:
        min_cost_base = []
        for doc_id in sorted(docs_candidates.keys()):
            cheapest = min(docs_candidates[doc_id], key=lambda x: x['cost'])
            min_cost_base.append(cheapest)
        total_cost = sum(c['cost'] for c in min_cost_base)
        if total_cost > max_cost:
            raise RuntimeError(f"即使每篇取最短片段总成本仍超出预算：{total_cost} > {max_cost}")
        best_base_combo = min_cost_base
        best_base_score = sum(c['score'] for c in min_cost_base)
        best_base_cost = total_cost

    selected = {c['chunk'].chunk_id for c in best_base_combo}
    current_cost = best_base_cost
    current_score = best_base_score

    # 4. 贪心扩充：利用剩余预算加入高收益片段
    remaining_budget = max_cost - current_cost
    pool = []
    for doc_id in docs_candidates:
        for cand in docs_candidates[doc_id]:
            if cand['chunk'].chunk_id not in selected:
                pool.append(cand)

    # 按得分降序，同分按成本升序
    pool.sort(key=lambda x: (-x['score'], x['cost']))

    for cand in pool:
        if cand['cost'] <= remaining_budget:
            selected.add(cand['chunk'].chunk_id)
            current_cost += cand['cost']
            current_score += cand['score']
            remaining_budget -= cand['cost']

    # 5. 后修正：如果预算远未用满（< B），尽量填满
    if current_cost < B:
        pool_remaining = [cand for cand in pool if cand['chunk'].chunk_id not in selected]
        pool_remaining.sort(key=lambda x: (-x['score'], x['cost']))
        for cand in pool_remaining:
            if current_cost + cand['cost'] <= max_cost:
                selected.add(cand['chunk'].chunk_id)
                current_cost += cand['cost']
                current_score += cand['score']

    # 如果仍超过上限（极少情况），剔除低效的非基础片段
    if current_cost > max_cost:
        non_base = [c for c in best_base_combo if c['chunk'].chunk_id in selected]
        # 收集所有非基础片段
        extra_ids = selected - {c['chunk'].chunk_id for c in best_base_combo}
        extra_cands = [c for c in pool if c['chunk'].chunk_id in extra_ids]
        # 按单位得分升序（低的先剔）
        extra_cands.sort(key=lambda x: x['score'] / x['cost'])
        while current_cost > max_cost and extra_cands:
            cand = extra_cands.pop(0)
            selected.remove(cand['chunk'].chunk_id)
            current_cost -= cand['cost']
            current_score -= cand['score']
        if current_cost > max_cost:
            raise RuntimeError("剔除后仍超预算，请检查配置")

    # 收集最终选中的 chunk 对象
    final_chunks = []
    for cand in chunk_list:  # 遍历原始去重后的所有 chunk
        if cand['chunk'].chunk_id in selected:
            final_chunks.append(cand)

    # 6. 按文档组织输出，保留原 MD 结构（按 chunk_index 排序，去重 section_path）
    docs = {}
    for cand in final_chunks:
        doc_id = cand['chunk'].doc_id
        if doc_id not in docs:
            docs[doc_id] = []
        docs[doc_id].append(cand)

    used_titles = []
    md_lines = []
    for doc_id, chunks in docs.items():
        md_lines.append(f"# {ids2name[doc_id]}")
        # 按 chunk_index 排序
        chunks.sort(key=lambda x: x['chunk'].chunk_index)
        for cand in chunks:
            chunk = cand['chunk']
            section_path = chunk.section_path
            if section_path not in used_titles:
                used_titles.append(section_path)
                md_lines.append(f"## {section_path}")
            md_lines.append(chunk.content)

    return "\n\n".join(md_lines)

def write_disk(data, path):
    Path(path).write_text(data, encoding='utf-8')

def main(data, client:OpenAI, retriever:Retriever, qid):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    save = []
    skip = 0
    for collection, name2ids in data:
        ids2name = {v:k for k,v in name2ids.items()}

        for record in collection:
            if record['qid']!=qid:
                continue
            if skip>0:
                skip-=1
                continue
            question_str = record['question']
            options = record['options']
            save.append({
                "qid":record['qid'],
                "question":record,
                "answer":[]
            })
            # 先检索资料
            doc_ids = {f"第{i+1}篇文档：{doc_name}":name2ids[doc_name] for i, doc_name in enumerate(record['doc_ids'])}
            doc_ids = json.dumps(doc_ids, ensure_ascii=False, indent=2)

            total_chunks = []
            if 'tf'==record['answer_format']:
                question = QUESTION_PROMPT.format(
                    question=question_str,
                    options=options)
                results = retriever.retrieve(question, top_k=5, category_name="保险合同检索")
                for result in results:
                    total_chunks.extend(result['results'])
            else:
                question = QUESTION_PROMPT.format(
                    question=question_str,
                    options="\n".join([f"{k}. {v}" for k,v in options.items()]))
                results = retriever.retrieve(question, top_k=5, category_name="保险合同检索")
                for result in results:
                    total_chunks.extend(result['results'])

            md = select_chunks_with_budget(total_chunks, ids2name, 6000, 2000, 0.2)

            answer_prompt = ANSWER_PROMPT[record['qid'].split('_')[0]].format(
                qtype="多选题" if 'multi'==record['answer_format'] else "单选题",
                doc=md,
                question=question_str,
                options=options
            )+'\n{"think":"判断理由...","answer":"..."}'

            try:
                response = client.chat.completions.create(
                    model="qwen3.7-plus",
                    messages=[
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.0,
                    extra_body = {
                        "enable_thinking": False,
                        "thinking":{"type": "disabled"}
                    }
                )
                answer = json.loads(response.choices[0].message.content)
                # save[-1]['answer'].append(answer)
                question_fm = ANSWER_PROMPT[record['qid'].split('_')[0]].format(
                    qtype="多选题" if 'multi'==record['answer_format'] else "单选题",
                    doc=md,
                    question=question_str,
                    options=json.dumps(options, ensure_ascii=False, indent=4)
                )
                write_disk(question_fm,'./question_fm.md')
                write_disk(answer['think'],'./think.md')
                print(doc_ids)
                print(f"{answer['answer']}({response.usage.total_tokens})")
            except Exception as e:
                # raise RuntimeError(f"调用大模型失败: {e}")
                print(f"调用大模型失败: {e}")
                save[-1]['answer'].append(None)

            # # with open('./save.jsonl', 'a', encoding='utf-8') as f:
            # #     f.write(json.dumps(save[-1], ensure_ascii=False) + '\n')
            # print('')

def load_data():
    data = []
    for path in Path('./data/question').glob("*.json"):
        load_path = Path('./data/output/') / path.stem.replace('_questions','') / '.governance_status'
        name2id = {}
        for status_file in load_path.glob('*.json'):
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            name2id[Path(status['file_path']).stem] = status['doc_id']
        data.append([
            json.loads(path.read_text(encoding='utf-8')),
            name2id
        ])
    return data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    data = load_data()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    retriever = Retriever(
        config_dir="config",
        documents_dir="./data/documents.all/insurance",
        index_dir="./data/group/insurance",
        force_rebuild=False,
        llm_api_base=os.environ['OPENAI_BASE_URL'],
        llm_api_key=os.environ['OPENAI_API_KEY'],
        llm_model="qwen3.7-plus",
        index_manager_type="instuance"
    )

    print("已加载数据，输入题目编号开始处理，输入 q 退出")
    while True:
        qid = input("\n题目编号: ").strip()
        if qid.lower() == 'q':
            break
        if not qid:
            continue
        main(data, client, retriever, qid)
