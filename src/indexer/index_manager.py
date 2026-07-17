from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os
import logging
from pathlib import Path
from whoosh import index as whoosh_index
from whoosh import fields, qparser
from whoosh.analysis import Tokenizer, Token, LowercaseFilter, StopFilter
from whoosh import query as wquery
from whoosh.query import Query
import jieba
import json

from src.models.document import Document, Chunk, Entity, Relation   # 假设原有模型路径

logger = logging.getLogger(__name__)

# ---------- 检索结果模型 ----------
class EntitySearchResult(BaseModel):
    entity: Entity
    chunk_id: str
    doc_id: str

class RelationSearchResult(BaseModel):
    relation: Relation
    chunk_id: str
    doc_id: str

# ---------- 分词器 ----------
class JiebaTokenizer(Tokenizer):
    def __call__(self, value, positions=False, chars=False,
                 keeporiginal=False, removestops=True,
                 start_pos=0, start_char=0, mode='', **kwargs):
        tokens = jieba.lcut(value)
        for pos, token in enumerate(tokens):
            t = Token()
            t.text = token.lower()
            t.pos = pos + start_pos
            t.boost = 1.0
            t.startchar = 0
            t.endchar = 0
            yield t

custom_analyzer = JiebaTokenizer() | LowercaseFilter() | StopFilter()

# ---------- 索引 Schema 定义 ----------
CHUNK_SCHEMA = fields.Schema(
    chunk_id=fields.ID(stored=True, unique=True),
    doc_id=fields.ID(stored=True),
    section_title=fields.TEXT(stored=True, analyzer=custom_analyzer),
    section_path=fields.TEXT(stored=True, analyzer=custom_analyzer),
    content=fields.TEXT(stored=True, analyzer=custom_analyzer),
    chunk_index=fields.NUMERIC(stored=True, numtype=int),
    prev_chunk_id=fields.ID(stored=True),
    next_chunk_id=fields.ID(stored=True),
)

ENTITY_SCHEMA = fields.Schema(
    entity_id=fields.ID(stored=True, unique=True),
    name=fields.TEXT(stored=True, analyzer=custom_analyzer),
    desc=fields.TEXT(stored=True, analyzer=custom_analyzer),
    entity_type=fields.ID(stored=True),
    chunk_id=fields.ID(stored=True),
    doc_id=fields.ID(stored=True),
)

RELATION_SCHEMA = fields.Schema(
    relation_id=fields.ID(stored=True, unique=True),
    subject=fields.TEXT(stored=True, analyzer=custom_analyzer),
    predicate=fields.TEXT(stored=True, analyzer=custom_analyzer),
    object=fields.TEXT(stored=True, analyzer=custom_analyzer),
    chunk_id=fields.ID(stored=True),
    doc_id=fields.ID(stored=True),
)

META_SCHEMA = fields.Schema(
    desc=fields.TEXT(stored=True, analyzer=custom_analyzer),
    doc_id=fields.ID(stored=True),
)


class DocumentLoader:
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir

    def load_all(self) -> List[Document]:
        documents = []
        if not os.path.exists(self.documents_dir):
            logger.warning(f"文档目录不存在: {self.documents_dir}")
            return documents
        for filename in Path(self.documents_dir).rglob("*.json"):
            if filename.parent.stem=='versions' or filename.stem=='latest':
                continue
            if not filename.suffix=='.json':
                continue
            filepath = filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data['doc_tree'] = [data.get('chunks',[{}])[0].get('doc_tree',[])]
                data['meta'] = data['structured_data']
                doc = Document(**data)
                documents.append(doc)
                logger.info(f"加载文档: {doc.doc_id} ({len(doc.chunks)} 个块)")
            except Exception as e:
                logger.error(f"加载文档失败 {filepath}: {e}")
        return documents


class IndexManager:
    def __init__(self, index_dir: str, documents_dir: str, force_rebuild: bool = False):
        self.index_dir = index_dir
        self.documents_dir = documents_dir
        self.force_rebuild = force_rebuild

        # 三个独立索引的路径
        self.meta_index_dir = os.path.join(index_dir, "metas")
        self.chunk_index_dir = os.path.join(index_dir, "chunks")
        self.entity_index_dir = os.path.join(index_dir, "entities")
        self.relation_index_dir = os.path.join(index_dir, "relations")

        # 内存映射
        self.chunk_dict: Dict[str, Chunk] = {}
        self.doc_dict: Dict[str, Document] = {}
        self.chunk_to_doc: Dict[str, str] = {}

        # 索引对象（chunk 索引保留为 self.index 以保持向后兼容）
        self.meta_index: whoosh_index.Index = None
        self.index: whoosh_index.Index = None          # chunk 索引
        self.entity_index: whoosh_index.Index = None
        self.relation_index: whoosh_index.Index = None

    def initialize(self):
        loader = DocumentLoader(self.documents_dir)
        documents = loader.load_all()
        self._fill_memory_maps(documents)

        # 检查并加载/构建 meta 索引
        if whoosh_index.exists_in(self.meta_index_dir) and not self.force_rebuild:
            self.meta_index = whoosh_index.open_dir(self.meta_index_dir)
        else:
            self._build_meta_index(documents)

        # 检查并加载/构建 chunk 索引
        if whoosh_index.exists_in(self.chunk_index_dir) and not self.force_rebuild:
            self.index = whoosh_index.open_dir(self.chunk_index_dir)
        else:
            self._build_chunk_index(documents)

        # 检查并加载/构建 entity 索引
        if whoosh_index.exists_in(self.entity_index_dir) and not self.force_rebuild:
            self.entity_index = whoosh_index.open_dir(self.entity_index_dir)
        else:
            self._build_entity_index(documents)

        # 检查并加载/构建 relation 索引
        if whoosh_index.exists_in(self.relation_index_dir) and not self.force_rebuild:
            self.relation_index = whoosh_index.open_dir(self.relation_index_dir)
        else:
            self._build_relation_index(documents)

        logger.info(f"索引初始化完成，meta: {self.meta_index.doc_count()}, "
                    f"chunk: {self.index.doc_count()}, "
                    f"entity: {self.entity_index.doc_count()}, "
                    f"relation: {self.relation_index.doc_count()}, "
                    )

    def _fill_memory_maps(self, documents: List[Document]):
        for doc in documents:
            self.doc_dict[doc.doc_id] = doc
            for chunk in doc.chunks:
                self.chunk_dict[chunk.chunk_id] = chunk
                self.chunk_to_doc[chunk.chunk_id] = doc.doc_id

    # ----- 分索引构建方法 -----
    def _build_meta_index(self, documents: List[Document]):
        os.makedirs(self.meta_index_dir, exist_ok=True)
        self.meta_index = whoosh_index.create_in(self.meta_index_dir, META_SCHEMA)
        writer = self.meta_index.writer()
        try:
            for doc in documents:
                if doc.doc_type=='CONTRACT':
                    writer.add_document(
                        doc_id=doc.doc_id,
                        desc= doc.meta['contract_name']+" "+doc.meta['insurer']
                    )
                elif doc.doc_type=='ANNUAL':
                    writer.add_document(
                        doc_id=doc.doc_id,
                        desc= doc.meta.get('stock_code')+" "+doc.meta['company_name']+" "+doc.meta['fiscal_year_end']
                    )
                else:
                    print('')
            writer.commit(optimize=True)
            logger.info("Meta 索引构建完成")
        except Exception as e:
            writer.cancel()
            logger.error(f"Meta 索引构建失败: {e}")
            raise

    def _build_chunk_index(self, documents: List[Document]):
        os.makedirs(self.chunk_index_dir, exist_ok=True)
        self.index = whoosh_index.create_in(self.chunk_index_dir, CHUNK_SCHEMA)
        writer = self.index.writer()
        try:
            for doc in documents:
                for chunk in doc.chunks:
                    writer.add_document(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        section_title=chunk.section_title,
                        section_path=chunk.section_path,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        prev_chunk_id=chunk.prev_chunk_id or "",
                        next_chunk_id=chunk.next_chunk_id or "",
                    )
            writer.commit(optimize=True)
            logger.info("Chunk 索引构建完成")
        except Exception as e:
            writer.cancel()
            logger.error(f"Chunk 索引构建失败: {e}")
            raise

    def _build_entity_index(self, documents: List[Document]):
        os.makedirs(self.entity_index_dir, exist_ok=True)
        self.entity_index = whoosh_index.create_in(self.entity_index_dir, ENTITY_SCHEMA)
        writer = self.entity_index.writer()
        try:
            for doc in documents:
                for chunk in doc.chunks:
                    for i, entity in enumerate(chunk.entities):
                        entity_id = f"{chunk.chunk_id}_entity_{i}"
                        writer.add_document(
                            entity_id=entity_id,
                            name=entity.name,
                            desc=entity.desc or "",
                            entity_type=entity.entity_type or "",
                            chunk_id=chunk.chunk_id,
                            doc_id=doc.doc_id,
                        )
            writer.commit(optimize=True)
            logger.info("Entity 索引构建完成")
        except Exception as e:
            writer.cancel()
            logger.error(f"Entity 索引构建失败: {e}")
            raise

    def _build_relation_index(self, documents: List[Document]):
        os.makedirs(self.relation_index_dir, exist_ok=True)
        self.relation_index = whoosh_index.create_in(self.relation_index_dir, RELATION_SCHEMA)
        writer = self.relation_index.writer()
        try:
            for doc in documents:
                for chunk in doc.chunks:
                    for j, relation in enumerate(chunk.relations):
                        relation_id = f"{chunk.chunk_id}_relation_{j}"
                        writer.add_document(
                            relation_id=relation_id,
                            subject=relation.subject or "",
                            predicate=relation.predicate,
                            object=relation.object,
                            chunk_id=chunk.chunk_id,
                            doc_id=doc.doc_id,
                        )
                        writer.add_document(
                            relation_id=relation_id,
                            subject=relation.object,
                            predicate=relation.predicate,
                            object=relation.subject or "",
                            chunk_id=chunk.chunk_id,
                            doc_id=doc.doc_id,
                        )
                        writer.add_document(
                            relation_id=relation_id,
                            subject=relation.object + " " + relation.predicate,
                            predicate=relation.predicate,
                            object=relation.subject or "" + " "+ relation.predicate,
                            chunk_id=chunk.chunk_id,
                            doc_id=doc.doc_id,
                        )
            writer.commit(optimize=True)
            logger.info("Relation 索引构建完成")
        except Exception as e:
            writer.cancel()
            logger.error(f"Relation 索引构建失败: {e}")
            raise

    # ----- 统一重建所有索引 -----
    def rebuild_index(self):
        loader = DocumentLoader(self.documents_dir)
        documents = loader.load_all()
        self.chunk_dict.clear()
        self.doc_dict.clear()
        self.chunk_to_doc.clear()
        self._fill_memory_maps(documents)

        self._build_chunk_index(documents)
        self._build_entity_index(documents)
        self._build_relation_index(documents)

    # ================= 三个检索函数 =================
    def search_chunks(self, query_str: str, limit: int = 10, doc_filter=None) -> List[Chunk]:
        """
        检索 Chunk，返回 Chunk 对象列表。
        默认搜索 content, section_title, section_path 字段。
        """
        if self.index is None:
            logger.error("Chunk 索引未初始化")
            return []

        qp = qparser.MultifieldParser(
            ["content", "section_title", "section_path"],
            schema=self.index.schema
        )
        q = qp.parse(query_str)
        with self.index.searcher() as searcher:
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])

            results = searcher.search(q, limit=limit)
            chunks = []
            for hit in results:
                chunk_id = hit["chunk_id"]
                if chunk_id in self.chunk_dict:
                    chunks.append(self.chunk_dict[chunk_id])
            return chunks

    def search_entities(self, query_str: str, limit: int = 10, doc_filter=None) -> List[EntitySearchResult]:
        """
        检索实体，返回实体对象及其来源信息。
        默认搜索 name, desc 字段（也可加入 entity_type）。
        """
        if self.entity_index is None:
            logger.error("实体索引未初始化")
            return []

        qp = qparser.MultifieldParser(
            ["name", "desc", "entity_type"],
            schema=self.entity_index.schema
        )
        q = qp.parse(query_str)
        with self.entity_index.searcher() as searcher:
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])
            results = searcher.search(q, limit=limit)
            output = []
            for hit in results:
                entity = Entity(
                    name=hit["name"],
                    desc=hit.get("desc", ""),
                    entity_type=hit.get("entity_type", "")
                )
                output.append(EntitySearchResult(
                    entity=entity,
                    chunk_id=hit["chunk_id"],
                    doc_id=hit["doc_id"]
                ))
            return output

    def search_relations(self, query_str: str, limit: int = 10, doc_filter=None) -> List[RelationSearchResult]:
        """
        检索关系，返回关系对象及其来源信息。
        默认搜索 subject, predicate, object 字段。
        """
        if self.relation_index is None:
            logger.error("关系索引未初始化")
            return []

        qp = qparser.MultifieldParser(
            ["subject", "predicate", "object"],
            schema=self.relation_index.schema
        )
        q = qp.parse(query_str)
        with self.relation_index.searcher() as searcher:
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])

            results = searcher.search(q, limit=limit)
            output = []
            for hit in results:
                relation = Relation(
                    subject=hit.get("subject", ""),
                    predicate=hit["predicate"],
                    object=hit["object"]
                )
                output.append(RelationSearchResult(
                    relation=relation,
                    chunk_id=hit["chunk_id"],
                    doc_id=hit["doc_id"]
                ))
            return output
    
    def get_entity_schema(self):
        """获取实体索引的 Schema，若未初始化返回 None"""
        if self.entity_index is None:
            return None
        return self.entity_index.schema

    def get_relation_schema(self):
        """获取关系索引的 Schema，若未初始化返回 None"""
        if self.relation_index is None:
            return None
        return self.relation_index.schema

    def search_metas_for_scoring(
        self,
        query: Query,
        limit: int = 10000,
        doc_filter: Optional[Query] = None,
    ) -> List[Tuple[str, float]]:
        """使用自定义 Query 搜索关系索引，返回 (chunk_id, score) 列表"""
        if self.meta_index_dir is None:
            logger.error("关系索引未初始化")
            return []
        with self.meta_index.searcher() as searcher:
            q = query
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])
            results = searcher.search(q, limit=limit)
            return [({
                "doc_id":hit["doc_id"],
                "desc":hit['desc']
            }, hit.score) for hit in results]

    def search_entities_for_scoring(
        self,
        query: Query,
        limit: int = 10000,
        doc_filter: Optional[Query] = None,
    ) -> List[Tuple[str, float]]:
        """使用自定义 Query 搜索实体索引，返回 (chunk_id, score) 列表"""
        if self.entity_index is None:
            logger.error("实体索引未初始化")
            return []
        with self.entity_index.searcher() as searcher:
            q = query
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])
            results = searcher.search(q, limit=limit)
            return [(hit["chunk_id"], hit.score) for hit in results]

    def search_relations_for_scoring(
        self,
        query: Query,
        limit: int = 10000,
        doc_filter: Optional[Query] = None,
    ) -> List[Tuple[str, float]]:
        """使用自定义 Query 搜索关系索引，返回 (chunk_id, score) 列表"""
        if self.relation_index is None:
            logger.error("关系索引未初始化")
            return []
        with self.relation_index.searcher() as searcher:
            q = query
            if doc_filter is not None:
                q = wquery.And([doc_filter, q])
            results = searcher.search(q, limit=limit)
            return [(hit["chunk_id"], hit.score) for hit in results]
