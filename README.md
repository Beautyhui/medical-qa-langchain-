# 智能医疗问答与辅助诊疗系统

基于 **LangChain** 框架和 **RAG（检索增强生成）** 技术的医疗智能问答原型系统，实现从症状输入、医学知识检索到辅助诊疗建议生成的完整流程。

## 系统架构

```
用户症状输入
    ↓
多轮对话记忆（症状摘要 + 历史上下文）
    ↓
RAG 知识检索（Chroma 向量库 + 语义搜索）
    ↓
上下文构建（检索结果 + 对话历史 + 提示词模板）
    ↓
大语言模型生成（疾病分析 / 检查建议 / 用药参考）
    ↓
结构化回答输出（附检索来源，可解释）
```

## 功能特性

| 模块 | 说明 |
|------|------|
| LangChain 应用链路 | 症状输入 → 检索 → 上下文构建 → 模型生成的完整 Pipeline |
| RAG 检索增强 | 外部医学知识库 + 向量语义检索，降低模型幻觉 |
| 多轮对话 | 结合历史症状连续分析，自动生成症状摘要 |
| 辅助诊疗输出 | 疾病可能性、检查建议、用药参考、就医建议 |
| 可解释性 | 展示检索来源，说明回答依据 |

## 项目结构

```
项目四 医疗/
├── app/
│   ├── chains/rag_chain.py        # LangChain RAG 诊疗主链路
│   ├── memory/conversation_memory.py  # 多轮对话记忆
│   ├── retriever/
│   │   ├── knowledge_loader.py    # 知识库加载与切分
│   │   └── vector_store.py        # 向量存储与检索
│   ├── prompts/medical_prompts.py # 提示词模板
│   ├── web/streamlit_app.py       # Web 界面
│   └── config.py                  # 配置管理
├── data/medical_knowledge/        # 医学知识库
│   ├── *.md                       # 8 份基础疾病文档
│   └── generated/                 # CSV 预处理生成的问答/药品知识
├── scripts/
│   ├── preprocess_csv_data.py     # CSV 清洗与知识库生成
│   └── build_knowledge_base.py    # 检索索引构建
├── main.py                        # 命令行入口
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 第一步：安装 Python 环境

确保已安装 **Python 3.10+**，然后在项目目录下执行：

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> 首次安装会下载嵌入模型（约 400MB），需要网络连接，请耐心等待。

### 第二步：配置 API Key

```bash
# 复制环境变量模板
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

编辑 `.env` 文件，填入你的大模型 API 配置：

```env
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

**支持的 API 服务（OpenAI 兼容格式）：**

| 服务商 | OPENAI_API_BASE | 推荐模型 |
|--------|-----------------|----------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
| 智谱 AI | https://open.bigmodel.cn/api/paas/v4 | glm-4-flash |

### 第三步：构建知识库

```bash
# 1. 预处理 CSV 数据（首次或 CSV 更新后执行）
python scripts/preprocess_csv_data.py

# 2. 构建检索索引
python scripts/build_knowledge_base.py --force
```

> 若 HuggingFace 模型下载失败，使用关键词检索：
> ```powershell
> $env:RETRIEVAL_MODE="keyword"
> python scripts/build_knowledge_base.py --force
> ```

### 第四步：启动系统

**方式一：Web 界面（推荐）**

```bash
streamlit run app/web/streamlit_app.py
```

浏览器访问 `http://localhost:8501`

**方式二：命令行**

```bash
python main.py
```

## 使用示例

### 单轮问诊

```
您: 我最近总是头痛，两侧太阳穴位置，按压会加重，工作压力大时更明显

系统输出:
  ## 症状分析
  根据描述，头痛位于双侧太阳穴，压迫加重，与压力相关...
  ## 疾病可能性分析
  1. 紧张型头痛（可能性高）...
  ## 建议检查
  - 血压测量...
  ## 用药参考
  - 对乙酰氨基酚、布洛芬...
```

### 多轮对话

```
您: 我咳嗽已经两周了，干咳为主
助手: [分析咳嗽可能原因...]

您: 晚上躺下咳得更厉害，还有反酸的感觉
助手: [结合上一轮信息，综合分析胃食管反流相关咳嗽...]
      [症状摘要自动更新: 干咳2周，夜间平卧加重，伴反酸]
```

## 扩展知识库

**方式一：添加 Markdown 文档**

在 `data/medical_knowledge/` 目录下添加 Markdown 格式的医学文档，然后重建索引。

**方式二：使用 CSV 数据（已内置）**

项目支持从 `肺炎(1).csv` 和 `药品库(1).csv` 自动导入：

```bash
python scripts/preprocess_csv_data.py          # 清洗 CSV → 生成 Markdown
python scripts/build_knowledge_base.py --force  # 重建索引
```

可选参数：
- `--max-qa 3000`：最大问答条数（默认 3000）
- `--max-drugs 5000`：最大药品条数（默认 5000）

## 技术说明

### RAG 方案设计

1. **文档切分**：按医学章节（`##` 标题）优先切分，保持语义完整性
2. **嵌入模型**：`paraphrase-multilingual-MiniLM-L12-v2`，支持中英文语义检索
3. **向量存储**：ChromaDB 本地持久化，无需额外数据库
4. **检索策略**：Top-K 相似度检索，结合症状摘要增强查询
5. **降级方案**：网络不佳时自动切换 TF-IDF 关键词检索（设置 `RETRIEVAL_MODE=keyword`）

### 多轮对话设计

1. **滑动窗口记忆**：保留最近 10 轮对话
2. **症状摘要**：第 2 轮起自动提取关键症状信息
3. **增强检索**：检索时将症状摘要与当前输入合并，提高召回相关性

## 注意事项

- 本系统为**课程/原型项目**，知识库为示例数据，不应用于实际医疗决策
- 所有回答均附带免责声明，不能替代专业医生诊断
- 首次运行需下载嵌入模型，建议使用稳定网络
- API 调用会产生费用，请合理选择模型

## 常见问题

**Q: 提示未配置 API Key？**
A: 确保已创建 `.env` 文件并正确填写 `OPENAI_API_KEY`。

**Q: 嵌入模型下载慢或失败？**
A: 设置 HuggingFace 镜像：`set HF_ENDPOINT=https://hf-mirror.com`（Windows），或直接使用关键词检索：`set RETRIEVAL_MODE=keyword` 后重新构建知识库。

**Q: 如何添加更多疾病知识？**
A: 在 `data/medical_knowledge/` 添加 `.md` 文件后运行 `python scripts/build_knowledge_base.py --force`。

**Q: 回答不够专业？**
A: 可扩充知识库内容，或换用更强的模型（如 gpt-4o、deepseek-chat）。
