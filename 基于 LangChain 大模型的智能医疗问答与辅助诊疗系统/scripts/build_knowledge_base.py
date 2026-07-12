"""初始化医学知识库向量索引"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever.vector_store import build_vector_store


def main():
    parser = argparse.ArgumentParser(description="构建医学知识库向量索引")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建向量库（忽略已有索引）",
    )
    args = parser.parse_args()

    print("正在加载医学知识文档...")
    build_vector_store(force_rebuild=args.force)
    print("向量库构建完成！")


if __name__ == "__main__":
    main()
