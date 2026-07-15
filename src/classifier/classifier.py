import logging
from typing import Tuple, Optional

import jieba

from src.models.schema import CategoryConfig, CategorySchema
from src.classifier.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def get_category_by_name(self, name: str) -> Tuple[CategoryConfig, CategorySchema]:
        for cat in self.config_manager.categories:
            if cat.name == name:
                schema = self.config_manager.get_schema(cat.id)
                return cat, schema
        return None, None

    def classify(self, question: str) -> Tuple[CategoryConfig, CategorySchema]:
        best_category: Optional[CategoryConfig] = None
        best_score: float = 0.0

        for cat in self.config_manager.categories:
            if cat.classification.method == "default":
                continue
            if cat.classification.method == "keyword":
                score = self._keyword_score(question, cat)
                threshold = cat.classification.threshold or 0.0
                if score >= threshold and score > best_score:
                    best_score = score
                    best_category = cat
            elif cat.classification.method == "llm":
                pass

        if best_category is None:
            best_category = self._get_default_category()

        schema = self.config_manager.get_schema(best_category.id)
        if schema is None:
            schema = self.config_manager.get_default_schema()
            best_category = self._get_default_category()

        logger.info(f"问题分类: '{question[:30]}...' -> {best_category.id} (得分: {best_score:.3f})")
        return best_category, schema

    def _keyword_score(self, question: str, cat: CategoryConfig) -> float:
        if not cat.classification.keywords:
            return 0.0
        question_tokens = set(jieba.lcut(question))
        question_chars = set(question)
        total_score = 0.0
        for group in cat.classification.keywords:
            hit_count = 0
            for word in group.words:
                if word in question_chars or word in question_tokens:
                    hit_count += 1
            if len(group.words) > 0:
                group_score = group.weight * (hit_count / len(group.words))
                total_score += group_score
        return total_score

    def _get_default_category(self) -> CategoryConfig:
        for cat in self.config_manager.categories:
            if cat.classification.method == "default":
                return cat
        return self.config_manager.categories[0]