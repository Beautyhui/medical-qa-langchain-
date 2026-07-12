"""医学知识库加载与文档切分"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR


def load_medical_documents(knowledge_dir: Path | None = None) -> list[Document]:
    """从知识库目录加载医学文档"""
    target_dir = knowledge_dir or KNOWLEDGE_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在: {target_dir}")

    loader = DirectoryLoader(
        str(target_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    for doc in documents:
        source_path = Path(doc.metadata.get("source", ""))
        filename = source_path.stem

        if "generated" in source_path.parts:
            if filename.startswith("pneumonia_qa"):
                doc.metadata["disease"] = "医患问答"
                doc.metadata["category"] = "qa"
            elif filename.startswith("drug_library"):
                doc.metadata["disease"] = "药品参考"
                doc.metadata["category"] = "drug"
            else:
                doc.metadata["disease"] = filename
                doc.metadata["category"] = "generated"
        else:
            doc.metadata["disease"] = filename
            doc.metadata["category"] = "medical_knowledge"

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档切分为适合检索的文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "，", " "],
    )
    return splitter.split_documents(documents)
