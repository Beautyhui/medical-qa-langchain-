"""LangChain RAG 诊疗链路"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import LLM_MODEL, OPENAI_API_BASE, OPENAI_API_KEY
from app.memory.conversation_memory import MedicalConversationMemory
from app.prompts.medical_prompts import RAG_PROMPT_TEMPLATE, SUMMARY_PROMPT, SYSTEM_PROMPT
from app.retriever.vector_store import retrieve_context


def create_llm(temperature: float = 0.3) -> ChatOpenAI:
    """创建大语言模型实例"""
    if not OPENAI_API_KEY:
        raise ValueError(
            "未配置 OPENAI_API_KEY，请在 .env 文件中设置 API Key"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=temperature,
    )


class MedicalRAGChain:
    """
    医疗 RAG 问答链路

    完整流程：症状输入 → 知识检索 → 上下文构建 → 模型生成
    """

    def __init__(self):
        self.llm = create_llm()
        self.memory = MedicalConversationMemory()
        self._setup_chains()

    def _setup_chains(self) -> None:
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE),
        ])
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

        summary_prompt = ChatPromptTemplate.from_messages([
            ("human", SUMMARY_PROMPT),
        ])
        self.summary_chain = summary_prompt | self.llm | StrOutputParser()

    def chat(self, user_input: str) -> dict:
        """
        处理用户输入，返回完整诊疗辅助回答

        Returns:
            dict: 包含回答文本、检索来源、对话轮次等信息
        """
        retrieval_query = self._build_retrieval_query(user_input)
        context, source_docs = retrieve_context(retrieval_query)

        chat_history = self.memory.get_full_context()

        answer = self.rag_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": user_input,
        })

        self.memory.add_exchange(user_input, answer)

        if self.memory.turn_count >= 2:
            self._update_symptom_summary()

        sources = [
            {
                "disease": doc.metadata.get("disease", "未知"),
                "category": doc.metadata.get("category", ""),
                "content_preview": doc.page_content[:150] + "...",
            }
            for doc in source_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "turn_count": self.memory.turn_count,
            "symptom_summary": self.memory.symptom_summary,
        }

    def _build_retrieval_query(self, user_input: str) -> str:
        """结合症状摘要增强检索查询"""
        if self.memory.symptom_summary:
            return f"{self.memory.symptom_summary}\n{user_input}"
        return user_input

    def _update_symptom_summary(self) -> None:
        """多轮对话后更新症状摘要"""
        conversation = self.memory.get_chat_history_text()
        summary = self.summary_chain.invoke({"conversation": conversation})
        self.memory.update_symptom_summary(summary)

    def reset_conversation(self) -> None:
        """重置对话，开始新的问诊会话"""
        self.memory.clear()
