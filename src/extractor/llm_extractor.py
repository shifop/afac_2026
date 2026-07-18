import json
import re
from loguru import logger
from typing import Dict, Optional, List

import openai

from src.models.schema import CategorySchema



class ExtractionError(Exception):
    pass


class LLMExtractor:
    def __init__(self, api_base: str, api_key: str, model: str,
                 temperature: float = 0.0, timeout: int = 30, max_retries: int = 2):
        self.client = openai.OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def extract(self, question: str, schema: CategorySchema) -> Dict[str, Optional[str]]:
        prompt = self._build_prompt(question, schema)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body = {
                        "enable_thinking": False,
                        "thinking":{"type": "disabled"}
                    }
                )
                raw_text = response.choices[0].message.content.strip()
                result = self._parse_response(raw_text, schema)
                # logger.info(f"LLM提取成功: {result}")
                logger.info(f'usage:{response.usage.total_tokens }')
                return result
            except openai.APIError as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"LLM提取异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
        raise ExtractionError("LLM调用失败，已达到最大重试次数")
    
    def extract_filter(self, question: str, schema: CategorySchema) -> Dict[str, Optional[str]]:
        prompt = self._build_filter_prompt(question, schema)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body = {
                        "enable_thinking": False,
                        "thinking":{"type": "disabled"}
                    }
                )
                raw_text = response.choices[0].message.content.strip()
                result = self._parse_response(raw_text, schema)
                # logger.info(f"LLM提取成功: {result}")
                logger.info(f'usage:{response.usage.total_tokens }')
                return result
            except openai.APIError as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"LLM提取异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
        raise ExtractionError("LLM调用失败，已达到最大重试次数")
    
    def filter(self, question: str, doc_ids: List[Dict]) -> Dict[str, Optional[str]]:
        prompt = """
请仔细阅读以下内容，找出与题目选项相关的所有文档ID。

问题内容：

{question}

文档信息：
{docs}

关联规则（必须严格遵守）：
1. 逐一检查选项A、B、C、D以及题干中提到的产品名称（包括保险公司和产品类型）。
2. 对于每个选项，在文档列表中寻找同时满足以下两个条件的文档：
   - 保险公司名称完全一致（如“平安”对应“平安”，“众安”对应“众安”）；
   - 文档名称中包含该选项产品名称的核心关键词（允许合理简称，例如“预防接种意外险”可匹配“预防接种意外伤害保险”，“食责险”可匹配“食品安全责任保险”，“重疾险”可匹配“重大疾病保险”）。
3. 如果存在符合上述条件的文档，则该文档即为关联文档，必须被选出；如果找不到，可以选择最有可能的。
4. 注意区分不同保险公司
5. 不管题目信息是否充足，只要产品在文档中出现，就必须列出所有关联的文档ID，不能遗漏。

请输出一个JSON数组，包含所有关联的文档序号，格式如：["序号1","序号1",...]
只输出JSON，不要添加任何其他文字或代码块标记。
"""
        docs_info = [[v['desc'], k] for k,v in doc_ids.items()]
        sn2ids = {i+1:_[1] for i,_ in enumerate(docs_info)}
        docs_info = [f"{i+1}. {_[0]}"for i,_ in enumerate(docs_info)]
        prompt = prompt.format(question=question, docs="\n".join(docs_info))
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    extra_body = {
                        "enable_thinking": False,
                        "thinking":{"type": "disabled"}
                    }
                )
                raw_text = response.choices[0].message.content.strip()
                result = json.loads(raw_text)
                # logger.info(f"LLM提取成功: {result}")
                logger.info(f'usage:{response.usage.total_tokens }')
                result = [sn2ids[int(_)] for _ in result if int(_) in sn2ids]
                return result
            except openai.APIError as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"LLM提取异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
        raise ExtractionError("LLM调用失败，已达到最大重试次数")

    def _build_prompt(self, question: str, schema: CategorySchema) -> str:
        fields_desc = []
        example = {}
        for f in schema.fields:
            required_mark = "必填" if f.required else "选填"
            fields_desc.append(f"- {f.field_name} ({required_mark}): {f.description}")
            example[f.field_name] = None

        fields_description = "\n".join(fields_desc)
        example = example
        example_json = json.dumps(example, ensure_ascii=False, indent=2)
        example_json = f"[\n{example_json},\n...]"

        prompt = (
            "你是一个信息检索专家。根据问题和选项提取要查询的信息，从问题中抽取对应的结构化信息。\n"
            "- 严格输出 JSON，不要包含任何其他文字或 markdown 标记。\n"
            "- 如果某个字段无法从问题中提取，该字段值设为 null。\n"
            "- 需要多次查询，例如不同文档查询条件不一样时，输出多组查询\n"
            "- 问题中存在数值，需要重点关注，存在数值需要在合适位置查询，不可丢失\n"
            "- 尽可能填充字段，例如问题或选项存在关键实体，应该在实体名称，关系首位实体上都进行查询\n"
            "- 对每个选项从多个角度生成2条查询，也就是至少生成选项数*2条查询\n"
            "- 不要编造信息。\n\n"
            f"问题类别：{schema.category_name}\n"
            f"提取字段说明：\n{fields_description}\n\n"
            f"问题：{question}\n\n"
            f"输出示例：\n{example_json}"
        )
        return prompt
    
    def _build_filter_prompt(self, question: str, schema: CategorySchema) -> str:
        fields_desc = []
        example = {}
        for f in schema.fields:
            required_mark = "必填" if f.required else "选填"
            fields_desc.append(f"- {f.field_name} ({required_mark}): {f.description}")
            example[f.field_name] = None

        fields_description = "\n".join(fields_desc)
        example = example
        example_json = json.dumps(example, ensure_ascii=False, indent=2)
        example_json = f"[\n{example_json},\n...]"

        prompt = (
            "你是一个信息检索专家。根据问题和选项提取要查询的信息，从问题中抽取对应的结构化信息。\n"
            "- 严格输出 JSON，不要包含任何其他文字或 markdown 标记。\n"
            "- 如果某个字段无法从问题中提取，该字段值设为 null。\n"
            "- 需要多次查询，例如不同文档查询条件不一样时，输出多组查询\n"
            "- 请仔细阅读问题和选项内容，找出相关的信息\n"
            "- 尽可能填充字段，确保查询充分\n"
            "- 对每个选项从多个角度生成2-3条查询，所以生成的查询数量不得少于选项数*2\n"
            "- 不要编造信息。\n\n"
            f"问题类别：{schema.category_name}\n"
            f"提取字段说明：\n{fields_description}\n\n"
            f"问题：{question}\n\n"
            f"输出示例：\n{example_json}"
        )
        return prompt

    def _parse_response(self, raw_text: str, schema: CategorySchema) -> Dict[str, Optional[str]]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ExtractionError(f"无法解析LLM返回的JSON: {raw_text[:200]}")

        if isinstance(data, dict):
            data = [data]
        
        collections = []
        for record in data:
            result = {}
            doc_ids = record.get("filter",{}).get('doc_ids',[])
            query = record

            for f in schema.fields:
                val = query.get(f.field_name)
                result[f.field_name] = str(val) if val is not None else None
            collections.append([
                result,
                doc_ids
            ])
        return collections