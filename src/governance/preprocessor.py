"""文档预处理模块 - 格式统一、文档类型识别、ID生成"""
import re
import os
from datetime import datetime, timezone
from typing import Tuple
import logging; logger = logging.getLogger(__name__)

from .models import DocType, DocumentMeta, new_id

# 文件夹名 → 文档类型
_FOLDER_TYPE_MAP: dict[str, DocType] = {
    "financial_contracts": DocType.PROSPECTUS,
    "financial_reports": DocType.ANNUAL_REPORT,
    "insurance": DocType.INSURANCE_CONTRACT,
    "regulatory": DocType.FINANCIAL_REGULATION,
    "research": DocType.RESEARCH_REPORT,
}


class DocumentPreprocessor:
    """文档预处理器"""

    def process(self, file_path: str, raw_content: str) -> Tuple[str, DocumentMeta]:
        """预处理文档，返回标准化内容和元数据"""
        content = self._normalize(raw_content)
        doc_type = self._detect_type(file_path)
        title = self._extract_title(content)
        doc_id = new_id("doc")
        meta = DocumentMeta(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            file_path=file_path,
            upload_time=datetime.now(timezone.utc).isoformat(),
            status="processing",
            version=1,
        )
        logger.info(f"文档预处理完成: id={doc_id}, type={doc_type.value}, title={title}")
        return content, meta

    def _normalize(self, content: str) -> str:
        """统一换行符、移除不可见字符"""
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        if content.startswith("\ufeff"):
            content = content[1:]
        content = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = self._inline_footnotes(content)
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[图片：\1]", content)
        content = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", content)
        return content.strip()

    def _inline_footnotes(self, content: str) -> str:
        """将脚注定义内联到引用位置"""
        footnote_defs: dict[str, str] = {}
        def_lines: list[str] = []
        for line in content.split("\n"):
            m = re.match(r"^\[\^(\d+)\]:\s*(.+)", line)
            if m:
                footnote_defs[m.group(1)] = m.group(2).strip()
            else:
                def_lines.append(line)
        if not footnote_defs:
            return content
        result_lines: list[str] = []
        for line in def_lines:
            for fn_id, fn_text in footnote_defs.items():
                line = line.replace(f"[^{fn_id}]", f"（脚注：{fn_text}）")
            result_lines.append(line)
        return "\n".join(result_lines)

    def _detect_type(self, file_path: str) -> DocType:
        """通过文件所在文件夹名判断文档类型"""
        # 沿路径向上查找匹配的文件夹名
        path = os.path.normpath(file_path)
        parts = path.split(os.sep)
        for part in reversed(parts):
            if part in _FOLDER_TYPE_MAP:
                return _FOLDER_TYPE_MAP[part]
        logger.warning(f"无法通过路径判断文档类型，标记为 unknown: {file_path}")
        return DocType.UNKNOWN

    def _extract_title(self, content: str) -> str:
        """提取文档标题（跳过 HTML 标签行，优先取 # 标题）"""
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("<"):
                continue
            m = re.match(r"^#+\s+(.+)", line)
            if m:
                return m.group(1).strip()
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("<"):
                return line[:100]
        return "Untitled"
