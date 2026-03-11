"""
【案例】RecursiveCharacterTextSplitter 分割纯文本并验证重叠与完整性（V2）

对应教程章节：第 19 章 - RAG 检索增强生成 → 2、RAG 文本处理核心知识

知识点速览：
- split_text() 得到字符串列表后，可手动用 [Document(page_content=text) for text in texts] 转成 Document，避免 create_documents 可能对块再做切分导致内容变化。
- chunk_overlap 会使相邻块有重复片段；若按「剔除重叠长度」再拼接，可验证是否覆盖原文且无丢失（本示例用固定 30 字符剔除演示）。
- 入门时以 RecursiveTextSplitter.py 的 split_text + create_documents 为主即可；本脚本侧重理解重叠与完整性。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

content = (
    "大模型RAG（检索增强生成）是一种结合生成模型与外部知识检索的技术，通过从大规模文档或数据库中检索相关信息，"
    "辅助生成模型以提升回答的准确性和相关性。其核心流程包括用户输入查询、系统检索相关知识、"
    "生成模型基于检索结果生成内容，并输出最终答案。RAG的优势在于能够弥补生成模型的知识盲区，"
    "提供更准确、实时和可解释的输出，广泛应用于问答系统、内容生成、客服、教育和企业领域。"
    "然而，其也面临依赖高质量知识库、可能的响应延迟、较高的维护成本以及数据隐私等挑战。"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30,
    length_function=len
)

# 先 split_text，再手动转 Document，避免 create_documents 二次切分
splitter_texts = text_splitter.split_text(content)
splitter_documents = [Document(page_content=text) for text in splitter_texts]

# 剔除重叠部分后拼接，用于验证与原文一致
full_content = ""
for text in splitter_texts:
    if full_content:
        full_content += text[30:]
    else:
        full_content += text

print(f"原始文本大小：{len(content)}，原始内容：\n{content}\n")
print(f"分割文档数量：{len(splitter_documents)}\n")
for idx, splitter_document in enumerate(splitter_documents, 1):
    print(f"第{idx}个文档 - 大小：{len(splitter_document.page_content)}, 内容：{splitter_document.page_content}\n")

print(f"拼接后文本大小：{len(full_content)}")
print(f"是否与原始文本完全一致：{full_content == content}")
print(f"拼接后完整内容：\n{full_content}")
