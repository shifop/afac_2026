"""结构化拆分模块 - 按标题拆分章节、段落句子切分、定位标识"""
import re
from typing import List, Tuple
import logging; logger = logging.getLogger(__name__)

from .models import Section, Paragraph, Sentence, Location, new_id


# 中文句子终止符模式: 匹配 。！？； 但不匹配小数点、百分号等
_SENT_END_PATTERN = re.compile(
    r"(?<![0-9.]\.)(?<!\d)%?(?:[。！？；]|(?<=\d)[。！？；])"
)

# Markdown 标题正则
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# 空行分隔段落
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class StructuralSplitter:
    """结构化拆分器"""

    def split(self, doc_id: str, content: str) -> List[Section]:
        """
        拆分文档为章节列表，每个章节包含段落和句子。
        """
        sections = self._split_sections(doc_id, content)
        for sec in sections:
            self._split_paragraphs(sec)
            for para in sec.paragraphs:
                self._split_sentences(para)
        logger.info(
            f"结构化拆分完成: doc_id={doc_id}, "
            f"sections={len(sections)}, "
            f"paragraphs={sum(len(s.paragraphs) for s in sections)}, "
            f"sentences={sum(sum(len(p.sentences) for p in s.paragraphs) for s in sections)}"
        )
        return sections

    def _split_sections(self, doc_id: str, content: str) -> List[Section]:
        """按标题拆分章节"""
        lines = content.split("\n")
        # 找到所有标题位置
        heading_positions: List[Tuple[int, int, str]] = []  # (line_idx, level, title)
        for i, line in enumerate(lines):
            m = _HEADING_PATTERN.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                heading_positions.append((i, level, title))

        if not heading_positions:
            # 无标题，整个文档作为一个章节
            sec_id = f"{doc_id}#"
            section = Section(
                section_id=sec_id,
                doc_id=doc_id,
                section_path="",
                title="全文",
                level=0,
                para_start=0,
                para_end=0,
                raw_text=content,
            )
            return [section]

        # 构建章节
        sections: List[Section] = []
        path_stack: List[str] = []  # 标题路径栈

        for idx, (pos, level, title) in enumerate(heading_positions):
            # 确定此标题在路径层级中的位置
            while path_stack and level <= len(path_stack):
                path_stack.pop()
            path_stack = path_stack[: level - 1] + [title]
            section_path = " > ".join(path_stack)
            sec_id = f"{doc_id}#{section_path}"

            # 获取章节文本范围
            start_line = pos + 1  # 标题下一行
            end_line = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
            sec_text = "\n".join(lines[start_line:end_line]).strip()

            section = Section(
                section_id=sec_id,
                doc_id=doc_id,
                section_path=section_path,
                title=title,
                level=level,
                para_start=0,
                para_end=0,
                raw_text=sec_text,
            )
            sections.append(section)

        # 记录段落索引范围（将在 _split_paragraphs 中更新）
        return sections

    def _split_paragraphs(self, section: Section) -> None:
        """在章节内按空行拆分段落"""
        text = section.raw_text
        if not text.strip():
            return

        parts = _PARAGRAPH_SPLIT.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return

        for idx, para_text in enumerate(parts):
            para_id = f"{section.section_id}_p{idx}"
            para = Paragraph(
                paragraph_id=para_id,
                section_id=section.section_id,
                para_index=idx,
                content=para_text,
            )
            section.paragraphs.append(para)

        section.para_start = 0
        section.para_end = len(section.paragraphs) - 1

    def _split_sentences(self, paragraph: Paragraph) -> None:
        """在段落内切分句子"""
        text = paragraph.content
        if not text:
            return

        # 按终止符切分
        parts = _SENT_END_PATTERN.split(text)
        # 保留终止符
        sentences_text: List[str] = []
        pos = 0
        for m in _SENT_END_PATTERN.finditer(text):
            end = m.end()
            sentences_text.append(text[pos:end])
            pos = end
        if pos < len(text):
            remaining = text[pos:].strip()
            if remaining:
                sentences_text.append(remaining)

        # 合并空句子
        sentences_text = [s.strip() for s in sentences_text if s.strip()]

        for idx, sent_text in enumerate(sentences_text):
            sent_id = f"{paragraph.paragraph_id}_s{idx}"
            loc = Location(
                doc_id=paragraph.paragraph_id.split("#")[0],
                section_path="#".join(paragraph.paragraph_id.split("#")[1:]).rsplit("_p", 1)[0] if "#" in paragraph.paragraph_id else "",
                paragraph_index=paragraph.para_index,
                sentence_index=idx,
            )
            sentence = Sentence(
                sentence_id=sent_id,
                text=sent_text,
                location=loc,
            )
            paragraph.sentences.append(sentence)
