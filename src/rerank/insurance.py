import itertools

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
