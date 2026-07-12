"""基于 TF-IDF 的关键词检索（无需下载嵌入模型，适合离线环境）"""

import pickle
from pathlib import Path

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import KNOWLEDGE_DIR, RETRIEVAL_TOP_K, VECTOR_STORE_PATH
from app.retriever.knowledge_loader import load_medical_documents, split_documents

INDEX_FILE = "tfidf_index.pkl"


class KeywordRetriever:
    """轻量级关键词检索器，作为向量检索的备用方案"""

    def __init__(self, chunks: list[Document], top_k: int = RETRIEVAL_TOP_K):
        self.chunks = chunks
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [doc.page_content for doc in chunks]
        )

    def invoke(self, query: str) -> list[Document]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][: self.top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]


_retriever_cache: KeywordRetriever | None = None


def _get_index_path() -> Path:
    return Path(VECTOR_STORE_PATH) / INDEX_FILE


def build_keyword_index(force_rebuild: bool = False) -> KeywordRetriever:
    """构建或加载 TF-IDF 索引"""
    global _retriever_cache
    index_path = _get_index_path()

    if not force_rebuild and index_path.exists():
        with open(index_path, "rb") as f:
            _retriever_cache = pickle.load(f)
        return _retriever_cache

    documents = load_medical_documents()
    chunks = split_documents(documents)
    _retriever_cache = KeywordRetriever(chunks)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(_retriever_cache, f)

    return _retriever_cache


def get_keyword_retriever(top_k: int | None = None) -> KeywordRetriever:
    retriever = build_keyword_index()
    if top_k:
        retriever.top_k = top_k
    return retriever
