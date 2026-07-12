"""向量存储与医学知识检索"""

import os
from pathlib import Path

from langchain_core.documents import Document

from app.config import EMBEDDING_MODEL, RETRIEVAL_TOP_K, VECTOR_STORE_PATH
from app.retriever.knowledge_loader import load_medical_documents, split_documents
from app.retriever.keyword_retriever import build_keyword_index, get_keyword_retriever

# 检索模式: auto | vector | keyword
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "auto")

_vector_store = None
_use_keyword_fallback = False


def _keyword_index_path() -> Path:
    return Path(VECTOR_STORE_PATH) / "tfidf_index.pkl"


def _keyword_index_exists() -> bool:
    return _keyword_index_path().exists()


def _try_get_embeddings():
    """尝试加载 HuggingFace 嵌入模型"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    persist_directory: str | None = None,
    force_rebuild: bool = False,
) -> bool:
    """
    构建检索索引。优先向量库，失败时自动降级为关键词检索。

    Returns:
        bool: True 表示使用向量检索，False 表示使用关键词检索
    """
    global _vector_store, _use_keyword_fallback

    if RETRIEVAL_MODE == "keyword":
        build_keyword_index(force_rebuild=force_rebuild)
        _use_keyword_fallback = True
        print("使用 TF-IDF 关键词检索模式")
        return False

    # auto 模式下优先使用已有关键词索引，避免下载 HuggingFace 模型卡住
    if RETRIEVAL_MODE == "auto" and _keyword_index_exists() and not force_rebuild:
        build_keyword_index(force_rebuild=False)
        _use_keyword_fallback = True
        print("使用 TF-IDF 关键词检索模式（检测到已有索引，快速启动）")
        return False

    persist_path = persist_directory or VECTOR_STORE_PATH
    store_path = Path(persist_path)

    if RETRIEVAL_MODE != "vector" and store_path.exists() and not force_rebuild:
        try:
            from langchain_chroma import Chroma

            embeddings = _try_get_embeddings()
            _vector_store = Chroma(
                persist_directory=persist_path,
                embedding_function=embeddings,
            )
            _use_keyword_fallback = False
            return True
        except Exception:
            pass

    if RETRIEVAL_MODE == "vector":
        return _build_chroma_store(persist_path, force_rebuild)

    try:
        return _build_chroma_store(persist_path, force_rebuild)
    except Exception as e:
        print(f"向量库构建失败 ({e})，自动切换为关键词检索模式")
        build_keyword_index(force_rebuild=force_rebuild)
        _use_keyword_fallback = True
        return False


def _build_chroma_store(persist_path: str, force_rebuild: bool) -> bool:
    global _vector_store, _use_keyword_fallback

    from langchain_chroma import Chroma

    embeddings = _try_get_embeddings()
    store_path = Path(persist_path)

    if store_path.exists() and not force_rebuild:
        _vector_store = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings,
        )
        _use_keyword_fallback = False
        return True

    documents = load_medical_documents()
    chunks = split_documents(documents)
    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
    )
    _use_keyword_fallback = False
    print("向量库构建完成（语义检索模式）")
    return True


def get_retriever(top_k: int | None = None):
    """获取检索器（向量或关键词）"""
    if _use_keyword_fallback:
        return get_keyword_retriever(top_k=top_k)

    if _vector_store is None:
        build_vector_store()

    if _use_keyword_fallback:
        return get_keyword_retriever(top_k=top_k)

    return _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k or RETRIEVAL_TOP_K},
    )


def retrieve_context(query: str, top_k: int | None = None) -> tuple[str, list[Document]]:
    """检索相关医学知识并格式化为上下文字符串"""
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(query)

    if not docs:
        return "未检索到相关医学知识，请基于通用医学常识谨慎回答。", []

    context_parts = []
    for i, doc in enumerate(docs, 1):
        disease = doc.metadata.get("disease", "未知")
        context_parts.append(f"[来源{i}: {disease}]\n{doc.page_content}")

    return "\n\n".join(context_parts), docs
