"""数据持久化模块 - PostgreSQL/SQLite 数据库操作"""
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import logging; logger = logging.getLogger(__name__)

from .models import (
    DocumentMeta, Section, Paragraph, Sentence,
    ExtractedEntity, ExtractedRelation, ExtractionResult,
)


class Database:
    """数据库访问层，支持 PostgreSQL 和 SQLite"""

    def __init__(self, db_url: str = "sqlite:///data/governance.db") -> None:
        """
        db_url 格式:
            - "sqlite:///path/to/db.sqlite"
            - "postgresql://user:pass@host:5432/dbname"
        """
        self.db_url = db_url
        self._engine = None
        self._conn = None
        self._dialect = "sqlite" if db_url.startswith("sqlite") else "postgresql"
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        if self._dialect == "sqlite":
            path = self.db_url.replace("sqlite:///", "")
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self.db_url)
            except ImportError:
                logger.warning("psycopg2 未安装，回退到 SQLite")
                self._dialect = "sqlite"
                self._init_db()
                return
        self._create_tables()

    def _create_tables(self) -> None:
        """创建所有数据表"""
        if self._dialect == "sqlite":
            self._create_tables_sqlite()
        else:
            self._create_tables_pg()

    def _create_tables_sqlite(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 1,
            is_latest INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL DEFAULT '',
            doc_type TEXT NOT NULL DEFAULT 'unknown',
            file_path TEXT DEFAULT '',
            upload_time TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'processing',
            metadata TEXT DEFAULT '{}',
            UNIQUE(doc_id, version)
        );

        CREATE TABLE IF NOT EXISTS sections (
            section_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            doc_version INTEGER NOT NULL DEFAULT 1,
            section_path TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            level INTEGER NOT NULL DEFAULT 0,
            para_start INTEGER NOT NULL DEFAULT 0,
            para_end INTEGER NOT NULL DEFAULT 0,
            reading_guide TEXT DEFAULT '{}',
            raw_text TEXT DEFAULT '',
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS paragraphs (
            paragraph_id TEXT PRIMARY KEY,
            section_id TEXT NOT NULL,
            para_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (section_id) REFERENCES sections(section_id)
        );

        CREATE TABLE IF NOT EXISTS sentences (
            sentence_id TEXT PRIMARY KEY,
            paragraph_id TEXT NOT NULL,
            sent_index INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL DEFAULT '',
            is_cell INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (paragraph_id) REFERENCES paragraphs(paragraph_id)
        );

        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            doc_version INTEGER NOT NULL DEFAULT 1,
            entity_type TEXT NOT NULL DEFAULT '',
            mention TEXT NOT NULL DEFAULT '',
            canonical_name TEXT NOT NULL DEFAULT '',
            attributes TEXT DEFAULT '{}',
            sentence_id TEXT DEFAULT '',
            source TEXT DEFAULT 'auto',
            confidence REAL NOT NULL DEFAULT 1.0,
            is_valid INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT '',
            PRIMARY KEY (entity_id, doc_id)
        );

        CREATE TABLE IF NOT EXISTS entity_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL,
            sentence_id TEXT NOT NULL,
            mention_text TEXT DEFAULT '',
            location TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS relations (
            relation_id TEXT PRIMARY KEY,
            relation_type TEXT NOT NULL DEFAULT '',
            head_entity_id TEXT NOT NULL DEFAULT '',
            tail_entity_id TEXT NOT NULL DEFAULT '',
            properties TEXT DEFAULT '{}',
            sentence_id TEXT DEFAULT '',
            doc_id TEXT NOT NULL DEFAULT '',
            doc_version INTEGER NOT NULL DEFAULT 1,
            source TEXT DEFAULT 'auto',
            confidence REAL NOT NULL DEFAULT 1.0,
            is_valid INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS entity_links (
            local_entity_id TEXT NOT NULL,
            global_entity_id TEXT NOT NULL,
            cluster_score REAL DEFAULT 0.0,
            PRIMARY KEY (local_entity_id)
        );

        CREATE TABLE IF NOT EXISTS rejected_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT DEFAULT '',
            stage TEXT DEFAULT '',
            reason TEXT DEFAULT '',
            raw_data TEXT DEFAULT '{}',
            created_at TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, canonical_name);
        CREATE INDEX IF NOT EXISTS idx_entities_sentence ON entities(sentence_id);
        CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations(doc_id, doc_version);
        CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
        """)
        self._conn.commit()

    def _create_tables_pg(self) -> None:
        # PostgreSQL 版本的 DDL（使用 TEXT 替代 JSONB 简化实现）
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR(64) PRIMARY KEY,
            version INT NOT NULL DEFAULT 1,
            is_latest BOOLEAN NOT NULL DEFAULT TRUE,
            title TEXT NOT NULL DEFAULT '',
            doc_type VARCHAR(32) NOT NULL DEFAULT 'unknown',
            file_path TEXT DEFAULT '',
            upload_time TIMESTAMPTZ DEFAULT NOW(),
            status VARCHAR(16) NOT NULL DEFAULT 'processing',
            metadata JSONB DEFAULT '{}',
            UNIQUE(doc_id, version)
        );
        -- ... (其他表类似)
        """)
        self._conn.commit()

    # ---------- 写入方法 ----------

    def insert_document(self, meta: DocumentMeta) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO documents
                (doc_id, version, is_latest, title, doc_type, file_path, upload_time, status, metadata)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)""",
                (meta.doc_id, meta.version, meta.title, meta.doc_type.value,
                 meta.file_path, meta.upload_time, meta.status, json.dumps(meta.metadata, ensure_ascii=False)),
            )
        self._conn.commit()

    def insert_section(self, section: Section, doc_version: int = 1) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO sections
                (section_id, doc_id, doc_version, section_path, title, level, para_start, para_end, reading_guide, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (section.section_id, section.doc_id, doc_version, section.section_path,
                 section.title, section.level, section.para_start, section.para_end,
                 json.dumps(section.reading_guide or {}, ensure_ascii=False),
                 section.raw_text),
            )
        self._conn.commit()

    def insert_paragraph(self, para: Paragraph) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO paragraphs
                (paragraph_id, section_id, para_index, content)
                VALUES (?, ?, ?, ?)""",
                (para.paragraph_id, para.section_id, para.para_index, para.content),
            )
        self._conn.commit()

    def insert_sentence(self, sent: Sentence, is_cell: bool = False) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO sentences
                (sentence_id, paragraph_id, sent_index, text, is_cell)
                VALUES (?, ?, ?, ?, ?)""",
                (sent.sentence_id, sent.sentence_id.rsplit("_s", 1)[0]
                 if "_s" in sent.sentence_id else "",
                 sent.location.sentence_index, sent.text, 1 if is_cell else 0),
            )
        self._conn.commit()

    def insert_entity(self, entity: ExtractedEntity, doc_id: str, doc_version: int = 1) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO entities
                (entity_id, doc_id, doc_version, entity_type, mention, canonical_name,
                 attributes, sentence_id, source, confidence, is_valid, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (entity.entity_id, doc_id, doc_version, entity.entity_type,
                 entity.mention, entity.canonical_name,
                 json.dumps(entity.attributes, ensure_ascii=False),
                 entity.sentence_id, entity.source, entity.confidence, now),
            )
        self._conn.commit()

    def insert_entity_mention(self, entity_id: str, sentence_id: str,
                               mention_text: str, location: dict) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT INTO entity_mentions
                (entity_id, sentence_id, mention_text, location)
                VALUES (?, ?, ?, ?)""",
                (entity_id, sentence_id, mention_text, json.dumps(location, ensure_ascii=False)),
            )
        self._conn.commit()

    def insert_relation(self, relation: ExtractedRelation, doc_id: str, doc_version: int = 1) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT OR REPLACE INTO relations
                (relation_id, relation_type, head_entity_id, tail_entity_id,
                 properties, sentence_id, doc_id, doc_version, source, confidence, is_valid, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (relation.relation_id, relation.relation_type,
                 relation.head_entity_id, relation.tail_entity_id,
                 json.dumps(relation.properties, ensure_ascii=False),
                 relation.evidence.get("sentence_id", ""),
                 doc_id, doc_version, relation.source, relation.confidence, now),
            )
        self._conn.commit()

    def insert_rejected(self, doc_id: str, stage: str, reason: str, raw_data: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                """INSERT INTO rejected_records (doc_id, stage, reason, raw_data, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (doc_id, stage, reason, json.dumps(raw_data, ensure_ascii=False), now),
            )
        self._conn.commit()

    def update_document_status(self, doc_id: str, status: str) -> None:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("UPDATE documents SET status=? WHERE doc_id=?", (status, doc_id))
        self._conn.commit()

    # ---------- 查询方法 ----------

    def get_document(self, doc_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM documents WHERE doc_id=? AND is_latest=1", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        return None

    def search_entities(
        self, entity_type: Optional[str] = None,
        canonical_name: Optional[str] = None, limit: int = 100,
    ) -> List[dict]:
        cur = self._conn.cursor()
        conditions = ["is_valid=1"]
        params: list = []
        if entity_type:
            conditions.append("entity_type=?")
            params.append(entity_type)
        if canonical_name:
            conditions.append("canonical_name LIKE ?")
            params.append(f"%{canonical_name}%")
        where = " AND ".join(conditions)
        if self._dialect == "sqlite":
            cur.execute(
                f"SELECT * FROM entities WHERE {where} LIMIT ?",
                params + [limit],
            )
            return [dict(r) for r in cur.fetchall()]
        return []

    def search_relations(
        self, relation_type: Optional[str] = None,
        head_entity_id: Optional[str] = None,
        tail_entity_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        cur = self._conn.cursor()
        conditions = ["is_valid=1"]
        params: list = []
        if relation_type:
            conditions.append("relation_type=?")
            params.append(relation_type)
        if head_entity_id:
            conditions.append("head_entity_id=?")
            params.append(head_entity_id)
        if tail_entity_id:
            conditions.append("tail_entity_id=?")
            params.append(tail_entity_id)
        where = " AND ".join(conditions)
        if self._dialect == "sqlite":
            cur.execute(
                f"SELECT * FROM relations WHERE {where} LIMIT ?",
                params + [limit],
            )
            return [dict(r) for r in cur.fetchall()]
        return []

    def get_entity_by_id(self, entity_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM entities WHERE entity_id=? AND is_valid=1", (entity_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        return None

    def get_section_guide(self, section_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT reading_guide FROM sections WHERE section_id=?", (section_id,))
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    return {}
        return None

    def get_section_by_path(self, doc_id: str, path_pattern: str) -> List[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute(
                "SELECT * FROM sections WHERE doc_id=? AND section_path LIKE ?",
                (doc_id, f"%{path_pattern}%"),
            )
            return [dict(r) for r in cur.fetchall()]
        return []

    def get_document_sections(self, doc_id: str) -> List[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM sections WHERE doc_id=? ORDER BY level, para_start", (doc_id,))
            return [dict(r) for r in cur.fetchall()]
        return []

    def get_sentence_by_id(self, sentence_id: str) -> Optional[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM sentences WHERE sentence_id=?", (sentence_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        return None

    def search_sentences_fulltext(self, query: str, doc_id: Optional[str] = None,
                                    limit: int = 20) -> List[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            if doc_id:
                cur.execute(
                    """SELECT s.* FROM sentences s
                    JOIN paragraphs p ON s.paragraph_id = p.paragraph_id
                    JOIN sections sec ON p.section_id = sec.section_id
                    WHERE s.text LIKE ? AND sec.doc_id = ? LIMIT ?""",
                    (f"%{query}%", doc_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM sentences WHERE text LIKE ? LIMIT ?",
                    (f"%{query}%", limit),
                )
            return [dict(r) for r in cur.fetchall()]
        return []

    def get_all_entities_for_doc(self, doc_id: str) -> List[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM entities WHERE doc_id=? AND is_valid=1", (doc_id,))
            return [dict(r) for r in cur.fetchall()]
        return []

    def get_all_relations_for_doc(self, doc_id: str) -> List[dict]:
        cur = self._conn.cursor()
        if self._dialect == "sqlite":
            cur.execute("SELECT * FROM relations WHERE doc_id=? AND is_valid=1", (doc_id,))
            return [dict(r) for r in cur.fetchall()]
        return []

    def close(self) -> None:
        if self._conn:
            self._conn.close()
