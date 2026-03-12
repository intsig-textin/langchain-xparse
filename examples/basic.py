"""
examples/basic.py

langchain-xparse 的基础用法示例脚本，可直接运行：

    python examples/basic.py

前提条件：
1. 已安装依赖：
   - pip install langchain-xparse python-dotenv
2. 在项目根目录配置 .env（或直接设置环境变量）：
   - XPARSE_APP_ID
   - XPARSE_SECRET_CODE
3. 示例文件存在：
   - example_docs/layout-parser-paper.pdf
"""

from __future__ import annotations

from pathlib import Path

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


def basic_parse_only() -> None:
    """最基础：只做 parse，把文档转成 LangChain 的 Document 列表。"""
    print_separator("示例 1：Basic (parse only)")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(file_path=str(EXAMPLE_PDF))
    docs = loader.load()

    print(f"共解析出 {len(docs)} 个 Document")
    if not docs:
        return

    first = docs[0]
    print("\n第 1 个 Document 文本前 300 字：")
    print(first.page_content[:300])
    print("\n第 1 个 Document 元数据：")
    for k, v in first.metadata.items():
        print(f"  {k}: {v}")


def parse_with_chunk() -> None:
    """在 parse 的基础上开启 chunk，把长文档切成小段。"""
    print_separator("示例 2：Parse + Chunk（按标题切分）")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(
        file_path=str(EXAMPLE_PDF),
        parse_provider="textin",      # 解析使用 textin
        chunk_strategy="by_title",    # 按标题分段
        chunk_max_characters=500,     # 每段字符上限
        chunk_overlap=50,             # 段之间的重叠字符数
    )
    docs = loader.load()

    print(f"共解析并切分出 {len(docs)} 个 chunk（Document）")
    if not docs:
        return

    first = docs[0]
    print("\n第 1 个 chunk 文本前 300 字：")
    print(first.page_content[:300])
    print("\n第 1 个 chunk 元数据：")
    for k, v in first.metadata.items():
        print(f"  {k}: {v}")


def parse_chunk_embed() -> None:
    """演示 parse + chunk + embed（需要在 xParse 控制台开启对应 embed 能力）。"""
    print_separator("示例 3：Parse + Chunk + Embed（如果已在控制台开通 embedding）")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(
        file_path=str(EXAMPLE_PDF),
        parse_provider="textin",
        chunk_strategy="basic",
        chunk_max_characters=800,
        # 这里的 provider / model_name 需要与你在 xParse 控制台中配置的一致
        embed_provider="qwen",
        embed_model_name="text-embedding-v4",
    )
    docs = loader.load()

    print(f"共解析 / 切分 / 向量化出 {len(docs)} 个 Document")
    if not docs:
        return

    first = docs[0]
    print("\n第 1 个 Document 文本前 200 字：")
    print(first.page_content[:200])
    print("\n第 1 个 Document 元数据中的 embeddings 信息（如果存在）：")
    if "embeddings" in first.metadata:
        emb = first.metadata["embeddings"]
        print(f"  向量维度：{len(emb)}，前 5 维：{emb[:5]}")
    else:
        print("  未返回 embeddings（请检查是否在控制台启用了 embedding 能力）")


def main() -> None:
    print("当前示例 PDF 路径：", EXAMPLE_PDF)
    basic_parse_only()
    parse_with_chunk()
    parse_chunk_embed()
    print("\n示例运行结束。")


if __name__ == "__main__":
    main()