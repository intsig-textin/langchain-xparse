"""
examples/document_rag_qa.py

演示如何结合 langchain-xparse + LangChain 框架，
实现一个最简单的“基于本地文档的问答（RAG）”示例。
使用国内可用的通义千问（Qwen）模型，避免 OpenAI 在国内的连接问题。

一键运行方式：

    python examples/document_rag_qa.py

前提条件（强烈建议按顺序检查）：

1. 安装依赖：
   - 核心：
     pip install langchain-xparse langchain-core langchain-community
   - 向量库（本示例默认使用 FAISS，本地运行最简单）：
     pip install faiss-cpu
   - 通义千问（阿里云 DashScope）：
     pip install dashscope
   - 可选：自动读取 .env
     pip install python-dotenv

2. 在项目根目录配置环境变量（推荐使用 .env）：
   - XPARSE_APP_ID       # 从 TextIn 控制台获取
   - XPARSE_SECRET_CODE  # 从 TextIn 控制台获取
   - DASHSCOPE_API_KEY   # 从阿里云百炼/灵积控制台获取（通义千问 API Key）

   示例 .env 内容（位于项目根目录）：
       XPARSE_APP_ID=your-app-id
       XPARSE_SECRET_CODE=your-secret-code
       DASHSCOPE_API_KEY=sk-xxx

3. 示例 PDF 文件存在：
   - example_docs/layout-parser-paper.pdf

脚本做的事情：
1. 使用 XParseLoader 调用 xParse Pipeline API，对 PDF 做解析 + 按标题切分 chunk。
2. 使用通义千问 Embeddings（DashScope）+ FAISS 向量库，将 chunk 构建成检索索引。
3. 使用通义千问大模型（ChatTongyi）+ LCEL，对用户问题进行基于文档的回答。
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_xparse import XParseLoader

# 可选：优先尝试从 .env 加载 XPARSE_APP_ID / XPARSE_SECRET_CODE
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    # 如果没有安装 python-dotenv 或没有 .env，也没关系，只要环境变量已设置即可
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PDF = PROJECT_ROOT / "example_docs" / "layout-parser-paper.pdf"


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_qa_chain() -> RunnableSerializable | None:
    """
    使用 XParseLoader + LangChain 构建一个最简单的基于文档的问答链。

    流程说明：
    1. XParseLoader 调用 xParse，将 PDF 解析为多个 chunk（LangChain Document）。
    2. 使用通义千问 Embeddings（DashScope）对每个 chunk 做向量化。
    3. 使用 FAISS 构建向量索引。
    4. 使用通义千问大模型（ChatTongyi）+ LCEL，在检索到的 chunk 上做问答。
    """

    print_separator("步骤 1：加载并切分 PDF（使用 XParseLoader）")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return None

    # 这里使用“按标题”分段策略，适合论文 / 报告类文档。
    loader = XParseLoader(
        file_path=str(EXAMPLE_PDF),
        parse_provider="textin",
        chunk_strategy="by_title",
        chunk_max_characters=800,
        chunk_overlap=100,
    )
    docs = loader.load()
    print(f"✅ 解析并切分完成，共得到 {len(docs)} 个 chunk（Document）")

    if not docs:
        print("文档列表为空，请检查 xParse 配置或示例文件。")
        return None

    print_separator("步骤 2：构建向量索引（通义千问 Embeddings + FAISS）")

    if not os.getenv("DASHSCOPE_API_KEY"):
        print(
            "未检测到 DASHSCOPE_API_KEY 环境变量，无法创建 Embeddings / LLM。\n"
            "请在 .env 或系统环境中设置 DASHSCOPE_API_KEY（阿里云百炼/灵积 API Key）后重试。"
        )
        return None

    # 通义千问文本向量模型，国内可直接访问。可选：text-embedding-v3, text-embedding-v2 等
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    print("✅ 向量索引构建完成。")

    print_separator("步骤 3：构建问答链（通义千问 ChatTongyi + LCEL）")

    # 通义千问大模型，国内可直接访问。qwen-plus 效果与成本平衡；可选 qwen-turbo（更快）、qwen-max（更强）
    llm = ChatTongyi(model="qwen-plus", temperature=0.1)

    # 使用 LangChain Expression Language 构建问答链：
    # 1) 从输入中取出 query，用 retriever 检索文档，格式化为 context 字符串
    # 2) prompt 把 context + question 拼成提示词，llm 生成答案，StrOutputParser 输出字符串
    prompt = ChatPromptTemplate.from_template(
        (
            "你是一个擅长阅读技术论文并用中文回答问题的助手。\n\n"
            "已知内容：\n{context}\n\n"
            "问题：{question}\n\n"
            "请用中文基于已知内容进行回答，如果无法从文档中找到答案，就明确说明。"
        )
    )

    document_chain = prompt | llm | StrOutputParser()

    def get_query(x: dict | str):
        if isinstance(x, dict):
            return x.get("query") or x.get("question") or ""
        return x

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    qa_chain: RunnableSerializable = (
        {
            "context": RunnableLambda(get_query) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(get_query),
        }
        | document_chain
    )

    print("✅ 问答链构建完成。")
    return qa_chain


def demo_single_qa(qa: RunnableSerializable) -> None:
    """
    运行一个简单的单轮问答 Demo。

    你可以根据自己的文档内容，修改下面的示例问题。
    当前示例假设 example_docs/layout-parser-paper.pdf 是 layout-parser 论文。
    """

    print_separator("示例问答：基于文档的 AI 问答 Demo")

    question = "这篇 layout-parser 论文的主要目标和贡献是什么？请用中文总结。"
    print(f"问题：{question}\n")

    result = qa.invoke({"query": question})
    # 链输出为字符串（StrOutputParser）
    answer = result if isinstance(result, str) else (result.get("result") or result.get("answer") or "")
    print("回答：")
    print(answer)


def main() -> None:
    print("当前示例 PDF 路径：", EXAMPLE_PDF)

    qa = build_qa_chain()
    if qa is None:
        print("\n❌ QA 链构建失败，请根据上面的提示检查配置。")
        return

    demo_single_qa(qa)

    print("\n示例运行结束。如需交互式多轮问答，可在本文件中自行扩展 input 循环。")


if __name__ == "__main__":
    main()