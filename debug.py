import logging
from src.retriever import Retriever
from src.rerank.report import batch_rerank_and_clip
from src.rerank.insurance import select_chunks_with_budget
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
                results, extracted_list = retriever.retrieve(question, top_k=10, category_name="募集说明书",  filter_by_llm=not record['qid'].startswith("res"))
                for result in results:
                    total_chunks.extend(result['results'])
            else:
                question = QUESTION_PROMPT.format(
                    question=question_str,
                    options="\n".join([f"{k}. {v}" for k,v in options.items()]))
                results, extracted_list = retriever.retrieve(question, top_k=10, category_name="募集说明书", filter_by_llm=not record['qid'].startswith("res"))
                for result in results:
                    total_chunks.extend(result['results'])

            if record['qid'].startswith("ins"):
                # md = select_chunks_with_budget(total_chunks, ids2name, 6000, 2000, 0.2)
                indicators = []
                for x,_ in extracted_list:
                    indicators.append([])
                    indicators[-1].append(x['contract_identity'])
                    indicators[-1].append(x['contract_party'])
                    indicators[-1].append(x['parameter'])
                    indicators[-1].append(x['coverage_and_exclusion'])
                    indicators[-1].append(x['business_rules'])
                    indicators[-1] = [_ for _ in indicators[-1] if _]
                md = batch_rerank_and_clip(total_chunks, ids2name, indicators, 6000, 2000, 15)
            elif record['qid'].startswith("fin"):
                indicators = []
                for x,_ in extracted_list:
                    indicators.append([])
                    indicators[-1].append(x['time'])
                    # indicators[-1].append(x['main'])
                    indicators[-1].append(x['metric_name'])
                    indicators[-1].append(x['metric_value'])
                    indicators[-1].append(x['metric_value_operator'])
                    indicators[-1] = [_ for _ in indicators[-1] if _]
                md = batch_rerank_and_clip(total_chunks, ids2name, indicators, 6000, 2000, 15)
            elif record['qid'].startswith("res"):
                indicators = []
                for x,_ in extracted_list:
                    indicators.append([])
                    indicators[-1].append(x['organization'])
                    indicators[-1].append(x['time_expression'])
                    indicators[-1].append(x['metric'])
                    indicators[-1].append(x['value'])
                    indicators[-1].append(x['statement'])
                    indicators[-1] = [_ for _ in indicators[-1] if _]
                md = batch_rerank_and_clip(total_chunks, ids2name, indicators, 6000, 2000, 15)
            elif record['qid'].startswith("reg"):
                indicators = []
                for x,_ in extracted_list:
                    indicators.append([])
                    indicators[-1].append(x['regulatory_subject'])
                    indicators[-1].append(x['obligation_clause'])
                    indicators[-1].append(x['timeframe'])
                    indicators[-1].append(x['threshold'])
                    indicators[-1].append(x['legal_basis'])
                    indicators[-1] = [_ for _ in indicators[-1] if _]
                md = batch_rerank_and_clip(total_chunks, ids2name, indicators, 6000, 2000, 15)
            elif record['qid'].startswith("fc"):
                indicators = []
                for x,_ in extracted_list:
                    indicators.append([])
                    indicators[-1].append(x['document'])
                    indicators[-1].append(x['issuer'])
                    indicators[-1].append(x['issue_information'])
                    indicators[-1].append(x['clause_detail'])
                    indicators[-1].append(x['financial_metric'])
                    indicators[-1] = [_ for _ in indicators[-1] if _]
                md = batch_rerank_and_clip(total_chunks, ids2name, indicators, 6000, 2000, 15)

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
        documents_dir="./data/documents.all/financial_contracts",
        index_dir="./data/group/financial_contracts",
        force_rebuild=False,
        llm_api_base=os.environ['OPENAI_BASE_URL'],
        llm_api_key=os.environ['OPENAI_API_KEY'],
        llm_model="qwen3.7-plus",
        index_manager_type="default"
    )

    print("已加载数据，输入题目编号开始处理，输入 q 退出")
    while True:
        qid = input("\n题目编号: ").strip()
        if qid.lower() == 'q':
            break
        if not qid:
            continue
        main(data, client, retriever, qid)
