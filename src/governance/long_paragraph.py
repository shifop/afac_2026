"""长段落处理模块 - 智能切分、上下文保留"""
import re
from typing import List, Tuple
import logging; logger = logging.getLogger(__name__)


class LongParagraphHandler:
    """长段落处理器"""

    def __init__(self, max_chars: int = 1000) -> None:
        self.max_chars = max_chars

    def needs_split(self, text: str) -> bool:
        """判断是否需要切分"""
        return len(text) > self.max_chars

    def split_with_context(self, text: str) -> List[str]:
        """切分长段落并保留上下文前缀"""
        if not self.needs_split(text):
            return [text]

        first_sentence = self._extract_first_sentence(text)
        sub_segments = self._split_by_threshold(text)
        logger.debug(f"长段落切分: {len(text)} 字符 → {len(sub_segments)} 段")

        results: List[str] = []
        prev_last = ""
        for i, seg in enumerate(sub_segments):
            context_parts: list[str] = []
            if first_sentence and not seg.strip().startswith(first_sentence[:10]):
                context_parts.append(f"【段首上下文】{first_sentence}")
            if prev_last and i > 0:
                context_parts.append(f"【前文】{prev_last}")
            context_parts.append(seg)
            results.append("\n".join(context_parts))
            prev_last = self._extract_last_sentence(seg)

        return results

    def _split_by_threshold(self, text: str) -> List[str]:
        """按阈值切分，优先在强终止符处切分"""
        segments: list[str] = []
        remaining = text

        while len(remaining) > self.max_chars:
            # 找最接近阈值上限的强终止符
            chunk = remaining[: self.max_chars]
            best_pos = -1
            for pattern in ["。", "；", "！", "？"]:
                pos = chunk.rfind(pattern)
                if pos > best_pos:
                    best_pos = pos

            if best_pos > self.max_chars * 0.3:  # 至少保留30%
                cut_pos = best_pos + 1
            else:
                # 找不到合适位置，硬切在 max_chars 处
                # 尽可能在最后一个非中文字符处切
                cut_pos = self.max_chars
                # 向后搜索最近的终止符
                search_end = min(len(remaining), cut_pos + 200)
                for pattern in ["。", "；", "！", "？"]:
                    pos = remaining.find(pattern, cut_pos, search_end)
                    if pos != -1:
                        cut_pos = pos + 1
                        break

            segments.append(remaining[:cut_pos].strip())
            remaining = remaining[cut_pos:].strip()

        if remaining:
            segments.append(remaining)

        return segments

    def _extract_first_sentence(self, text: str) -> str:
        """提取首句"""
        for sep in ["。", "；", "！", "？"]:
            idx = text.find(sep)
            if 0 < idx < 200:
                return text[: idx + 1]
        return text[:200]

    def _extract_last_sentence(self, text: str) -> str:
        """提取末句"""
        for sep in ["。", "；", "！", "？"]:
            idx = text.rfind(sep)
            if idx > len(text) - 200:
                prev_idx = text.rfind(sep, 0, idx)
                start = (prev_idx + 1) if prev_idx > 0 else 0
                return text[start : idx + 1]
        return text[-200:]
