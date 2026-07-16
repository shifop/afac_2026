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
                return result
            except openai.APIError as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"LLM提取异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
        raise ExtractionError("LLM调用失败，已达到最大重试次数")
    
    def filter(self, question: str, doc_ids: List[Dict]) -> Dict[str, Optional[str]]:
        prompt = """
问题内容：

{question}

文档信息：
{docs}

请仔细理解上述文本已经候选文档信息，选择所以需要阅读的文档，直接输出文档id即可

输出格式：

["...",...]

请直接输出json数据，不需要添加其他内容，不需要使用```包裹
"""
        prompt = prompt.format(question=question, docs=json.dumps(doc_ids,ensure_ascii=False, indent=2))
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
                result = json.loads(raw_text)
                # logger.info(f"LLM提取成功: {result}")
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
        example = {
            "query":example
        }
        example_json = json.dumps(example, ensure_ascii=False, indent=2)
        example_json = f"[\n{example_json},\n...]"

        prompt = (
            "你是一个信息检索专家。根据问题和选项提取要查询的信息，从问题中抽取对应的结构化信息。\n"
            "- 严格输出 JSON，不要包含任何其他文字或 markdown 标记。\n"
            "- 如果某个字段无法从问题中提取，该字段值设为 null。\n"
            "- 需要多次查询，例如不同文档查询条件不一样时，输出多组查询\n"
            "- 问题中存在数值，需要重点关注，存在数值需要在合适位置查询，不可丢失\n"
            "- 尽可能填充字段，例如问题或选项存在关键实体，应该在实体名称，关系首位实体上都进行查询\n"
            "- 对每个选项从多个角度生成2-3条查询\n"
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
        example = {
            "query":example
        }
        example_json = json.dumps(example, ensure_ascii=False, indent=2)
        example_json = f"[\n{example_json},\n...]"

        prompt = (
            "你是一个信息检索专家。根据问题和选项提取要查询的信息，从问题中抽取对应的结构化信息。\n"
            "- 严格输出 JSON，不要包含任何其他文字或 markdown 标记。\n"
            "- 如果某个字段无法从问题中提取，该字段值设为 null。\n"
            "- 需要多次查询，例如不同文档查询条件不一样时，输出多组查询\n"
            "- 请仔细阅读问题和选项内容，找出相关的信息\n"
            "- 尽可能填充字段，确保查询充分\n"
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
            query = record['query']

            for f in schema.fields:
                val = query.get(f.field_name)
                result[f.field_name] = str(val) if val is not None else None
            collections.append([
                result,
                doc_ids
            ])
        return collections