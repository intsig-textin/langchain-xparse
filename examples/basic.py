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


def basic_parse() -> None:
    """最基础：使用默认配置解析 PDF。"""
    print_separator("示例 1：基础解析（使用默认配置）")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(file_path=str(EXAMPLE_PDF))
    docs = loader.load()

    print(f"共解析出 {len(docs)} 个 Document（element）")
    if not docs:
        return

    first = docs[0]
    print("\n第 1 个 Document 文本前 300 字：")
    print(first.page_content[:300])
    print("\n第 1 个 Document 元数据：")
    for k, v in first.metadata.items():
        print(f"  {k}: {v}")


def parse_with_hierarchy() -> None:
    """开启层级关系和表格结构解析。"""
    print_separator("示例 2：开启层级关系和表格结构")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(
        file_path=str(EXAMPLE_PDF),
        config={
            "capabilities": {
                "include_hierarchy": True,         # 包含父子关系
                "include_table_structure": True,   # 详细表格结构
                "title_tree": True,                # 文档目录
            }
        },
    )
    docs = loader.load()

    print(f"共解析出 {len(docs)} 个 Document（element）")
    if not docs:
        return

    # 查找包含 parent_id 的文档
    for doc in docs[:5]:
        if "parent_id" in doc.metadata:
            print(f"\n元素 {doc.metadata.get('element_id')} 的父元素: {doc.metadata['parent_id']}")
            print(f"文本: {doc.page_content[:100]}...")
            break


def parse_with_advanced_features() -> None:
    """演示高级特性：内嵌对象、字符详情、图片数据。"""
    print_separator("示例 3：高级特性（内嵌对象、字符详情、图片数据）")

    if not EXAMPLE_PDF.exists():
        print(f"示例 PDF 不存在，请检查路径：{EXAMPLE_PDF}")
        return

    loader = XParseLoader(
        file_path=str(EXAMPLE_PDF),
        config={
            "capabilities": {
                "include_hierarchy": True,
                "include_inline_objects": True,    # 提取公式、手写等内嵌对象
                "include_char_details": False,     # 字符级详情（可选）
                "include_image_data": True,        # 图片 URL 和数据
                "pages": True,                     # 页面元信息
            },
            "scope": {
                "page_range": "1-3"  # 只处理前 3 页
            },
        },
    )
    docs = loader.load()

    print(f"共解析出 {len(docs)} 个 Document（element）")
    if not docs:
        return

    # 查找包含内嵌对象的文档
    for doc in docs:
        if "has_inline_objects" in doc.metadata and doc.metadata["has_inline_objects"]:
            print(f"\n找到包含内嵌对象的元素:")
            print(f"  类型: {doc.metadata.get('category')}")
            print(f"  内嵌对象类型: {doc.metadata.get('inline_object_types')}")
            print(f"  文本: {doc.page_content[:200]}...")
            break


def main() -> None:
    print("当前示例 PDF 路径：", EXAMPLE_PDF)
    basic_parse()
    parse_with_hierarchy()
    parse_with_advanced_features()
    print("\n示例运行结束。")


if __name__ == "__main__":
    main()