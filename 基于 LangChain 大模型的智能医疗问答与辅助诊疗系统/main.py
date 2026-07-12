"""医疗智能问答系统 - 命令行入口"""

import sys

from app.chains.rag_chain import MedicalRAGChain
from app.retriever.vector_store import build_vector_store


def print_banner():
    print("=" * 60)
    print("  智能医疗问答与辅助诊疗系统")
    print("  基于 LangChain + RAG 技术")
    print("=" * 60)
    print("  输入症状或健康问题开始咨询，输入 quit 退出")
    print("  输入 reset 开始新的问诊会话")
    print("  ⚠️  本系统仅供参考，不能替代专业医生诊断")
    print("=" * 60)


def print_response(result: dict):
    print("\n" + "-" * 60)
    print("【辅助诊疗建议】")
    print(result["answer"])

    if result["sources"]:
        print("\n【检索来源】")
        for i, src in enumerate(result["sources"], 1):
            cat = src.get("category", "")
            label = f"{src['disease']}" + (f" ({cat})" if cat else "")
            print(f"  {i}. {label}: {src['content_preview']}")

    if result["symptom_summary"]:
        print(f"\n【症状摘要】{result['symptom_summary']}")

    print(f"\n（第 {result['turn_count']} 轮对话）")
    print("-" * 60)


def main():
    print_banner()

    print("\n正在初始化知识库...")
    try:
        build_vector_store()
        print("知识库加载完成！\n")
    except Exception as e:
        print(f"知识库初始化失败: {e}")
        sys.exit(1)

    try:
        chain = MedicalRAGChain()
    except ValueError as e:
        print(f"\n错误: {e}")
        print("请复制 .env.example 为 .env 并配置 API Key")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n您: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "reset":
            chain.reset_conversation()
            print("已开始新的问诊会话。")
            continue

        try:
            result = chain.chat(user_input)
            print_response(result)
        except Exception as e:
            print(f"\n处理出错: {e}")
            print("请检查 API 配置和网络连接。")


if __name__ == "__main__":
    main()
