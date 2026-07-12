"""多轮对话记忆管理"""

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage

from app.config import MEMORY_WINDOW


class MedicalConversationMemory:
    """医疗对话记忆，支持多轮症状信息的连续追踪"""

    def __init__(self, window_size: int | None = None):
        self.window_size = window_size or MEMORY_WINDOW
        self.memory = ConversationBufferWindowMemory(
            k=self.window_size,
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )
        self.symptom_summary: str = ""

    def add_exchange(self, user_input: str, assistant_response: str) -> None:
        """记录一轮对话"""
        self.memory.save_context(
            {"question": user_input},
            {"answer": assistant_response},
        )

    def get_chat_history_text(self) -> str:
        """将对话历史格式化为文本"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return "（暂无对话历史）"

        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"助手: {msg.content[:300]}...")
        return "\n".join(lines)

    def update_symptom_summary(self, summary: str) -> None:
        """更新症状摘要（用于跨轮综合分析）"""
        self.symptom_summary = summary

    def get_full_context(self) -> str:
        """获取包含症状摘要和对话历史的完整上下文"""
        parts = []
        if self.symptom_summary:
            parts.append(f"【症状摘要】\n{self.symptom_summary}")
        history = self.get_chat_history_text()
        if history != "（暂无对话历史）":
            parts.append(f"【对话历史】\n{history}")
        return "\n\n".join(parts) if parts else "（首次对话）"

    def clear(self) -> None:
        """清空对话记忆"""
        self.memory.clear()
        self.symptom_summary = ""

    @property
    def turn_count(self) -> int:
        return len(self.memory.chat_memory.messages) // 2
