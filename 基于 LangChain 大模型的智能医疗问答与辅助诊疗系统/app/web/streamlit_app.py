"""智能医疗问答系统 - Streamlit Web 界面"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，解决 streamlit run 找不到 app 模块的问题
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.chains.rag_chain import MedicalRAGChain
from app.retriever.vector_store import build_vector_store

st.set_page_config(
    page_title="智能医疗问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a73e8;
        text-align: center;
        padding: 1rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1a73e8;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .symptom-summary {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def init_vector_store():
    return build_vector_store()


@st.cache_resource
def init_chain():
    return MedicalRAGChain()


def render_sidebar():
    with st.sidebar:
        st.header("系统信息")
        st.markdown("""
        **技术架构**
        - LangChain 应用框架
        - RAG 检索增强生成
        - 多轮对话记忆

        **功能模块**
        1. 症状输入与分析
        2. 医学知识检索
        3. 疾病可能性分析
        4. 检查与用药建议
        """)

        st.divider()
        st.header("操作")
        if st.button("🔄 开始新问诊", use_container_width=True):
            if "chain" in st.session_state:
                st.session_state.chain.reset_conversation()
            st.session_state.messages = []
            st.rerun()

        if "chain" in st.session_state:
            turn = st.session_state.chain.memory.turn_count
            st.metric("对话轮次", turn)
            if st.session_state.chain.memory.symptom_summary:
                st.subheader("症状摘要")
                st.info(st.session_state.chain.memory.symptom_summary)


def render_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 检索来源"):
                    for src in msg["sources"]:
                        cat = src.get("category", "")
                        label = src["disease"] + (f" · {cat}" if cat else "")
                        st.markdown(
                            f'<div class="source-card"><b>{label}</b><br>'
                            f'{src["content_preview"]}</div>',
                            unsafe_allow_html=True,
                        )


def main():
    st.markdown('<div class="main-header">🏥 智能医疗问答与辅助诊疗系统</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠️ <b>免责声明</b>：本系统基于大语言模型和医学知识库提供'
        "健康信息参考，<b>不能替代专业医生的诊断和治疗</b>。"
        "如有不适请及时就医。</div>",
        unsafe_allow_html=True,
    )

    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("正在加载医学知识库（首次约需 10~20 秒，请稍候）..."):
        try:
            init_vector_store()
            st.session_state.kb_ready = True
        except Exception as e:
            st.error(f"知识库初始化失败: {e}")
            st.stop()

    if "chain" not in st.session_state:
        try:
            st.session_state.chain = init_chain()
        except ValueError as e:
            st.error(str(e))
            st.info("请复制 `.env.example` 为 `.env`，配置 `OPENAI_API_KEY` 后重启应用。")
            st.stop()

    render_chat()

    if prompt := st.chat_input("请描述您的症状或健康问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索医学知识并分析..."):
                try:
                    result = st.session_state.chain.chat(prompt)
                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("📚 检索来源"):
                            for src in result["sources"]:
                                cat = src.get("category", "")
                                label = src["disease"] + (f" · {cat}" if cat else "")
                                st.markdown(
                                    f'<div class="source-card"><b>{label}</b><br>'
                                    f'{src["content_preview"]}</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.error(f"处理失败: {e}")


if __name__ == "__main__":
    main()
