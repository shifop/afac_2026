import os
import json
import logging
from typing import Dict, List, Tuple, Optional

import yaml
import jieba

from src.models.schema import (
    CategoryConfig,
    CategorySchema,
    CategoryConfigKeywordGroup,
)

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.categories: List[CategoryConfig] = []
        self.schema_registry: Dict[str, CategorySchema] = {}

    def load_all(self):
        self._load_categories()
        self._load_schemas()
        logger.info(f"配置加载完成: {len(self.categories)} 个类别, {len(self.schema_registry)} 个Schema")

    def _load_categories(self):
        categories_path = os.path.join(self.config_dir, "categories.yaml")
        if not os.path.exists(categories_path):
            logger.error(f"类别配置文件不存在: {categories_path}")
            return
        with open(categories_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in data.get("categories", []):
            try:
                cat = CategoryConfig(**item)
                self.categories.append(cat)
            except Exception as e:
                logger.error(f"解析类别配置失败: {e}")

    def _load_schemas(self):
        schemas_dir = os.path.join(self.config_dir, "schemas")
        if not os.path.exists(schemas_dir):
            logger.error(f"Schema目录不存在: {schemas_dir}")
            return
        for cat in self.categories:
            schema_path = os.path.join(self.config_dir, cat.schema_file)
            if not os.path.exists(schema_path):
                logger.warning(f"Schema文件不存在: {schema_path}")
                continue
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                schema = CategorySchema(**data)
                self.schema_registry[cat.id] = schema
            except Exception as e:
                logger.error(f"加载Schema失败 {schema_path}: {e}")

    def get_schema(self, category_id: str) -> Optional[CategorySchema]:
        return self.schema_registry.get(category_id)

    def get_default_schema(self) -> Optional[CategorySchema]:
        return self.schema_registry.get("default")