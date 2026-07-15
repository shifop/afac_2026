import logging
from src.retriever import Retriever
import os
import json
from openai import OpenAI
from pathlib import Path
from loguru import logger

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

def main(data, client:OpenAI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
    save = []
    skip = 9
    for collection, name2ids in data:
        ids2name = {v:k for k,v in name2ids.items()}

        for record in collection:
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
            doc_ids_obj = {f"第{i+1}篇文档：{doc_name}":name2ids[doc_name] for i, doc_name in enumerate(record['doc_ids'])}
            doc_ids = json.dumps(doc_ids_obj, ensure_ascii=False, indent=2)

            chunks = []
            if 'tf'==record['answer_format']:
                question = QUESTION_PROMPT.format(
                    question=question_str,
                    options=options)
                results = retriever.retrieve(question, top_k=5, category_name="保险合同检索")
                for result in results:
                        chunks.extend(result['results'])
            else:
                for k,v in options.items():
                    question = QUESTION_PROMPT.format(
                        question=question_str,
                        options=f"{k}. {v}")
                    results = retriever.retrieve(question, top_k=5, category_name="保险合同检索")
                    for result in results:
                        chunks.extend(result['results'])

            chunks.sort(key=lambda x:x.score, reverse=True)
            # 汇总并截断
            docs = {}
            used_chunks = []
            for chunk in chunks:
                if chunk.chunk.chunk_id not in used_chunks:
                    used_chunks.append(chunk.chunk.chunk_id)
                else:
                    continue
                if chunk.chunk.doc_id not in docs:
                    docs[chunk.chunk.doc_id] = []
                docs[chunk.chunk.doc_id].append(chunk)

            for k,v in docs.items():
                v = sorted(v, key=lambda x:x.score)
                v_ = [_.chunk for _ in v if _.score>0.2]
                if len(v_)==0:
                    v_ = [v[-1].chunk]
                docs[k] = v_

            # 构建可读性强文档
            used_titles = []
            md = []
            for doc_id, chunks in docs.items():
                md.append(f"# {ids2name[doc_id]}")
                chunks.sort(key=lambda x: x.chunk_index)
                for chunk in chunks:
                    section_path = chunk.section_path
                    titles = section_path.split('/')
                    if section_path not in used_titles:
                        used_titles.append(section_path)
                        md.append(f"## {section_path}")
                    md.append(chunk.content)
            md = "\n\n".join(md)

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
                answer['usage'] = response.usage.total_tokens 
                save[-1]['answer'].append(answer)
            except Exception as e:
                # raise RuntimeError(f"调用大模型失败: {e}")
                print(f"调用大模型失败: {e}")
                save[-1]['answer'].append(None)

            with open('./save.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(save[-1], ensure_ascii=False) + '\n')

            # for k,v in doc_ids_obj.items():
            #     if v not in docs:
            #         logger.info(f"全量查询。无法命中文档：{k}({v})")
            # print('')

if __name__ == "__main__":
    from pathlib import Path
    import json
    data = []
    for path in Path('./data/question').glob("*.json"):
        if path.stem!='insurance_questions':
            continue
        # 获取文档名称与 id 的映射
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

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    main(data, client)
