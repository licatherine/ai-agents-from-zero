"""
【案例】LangChain DashScope 封装：单条与批量文本向量化

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4、Embedding 文本向量化

知识点速览：
- LangChain 的 DashScopeEmbeddings 对百炼嵌入做了统一封装，便于在链、检索器等组件中复用。
- embed_query(text)：对「单条」查询文本向量化，常用于把用户问题转成向量做检索。
- embed_documents(texts)：对「多条」文本批量向量化，常用于建索引时把文档片段转成向量。
- 返回为浮点数列表（单条）或列表的列表（多条）；len(result) 即为向量维度。

模型文档链接：https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654
"""

# pip install langchain-community dashscope
import os
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 使用项目统一的 aliQwen-api；DashScopeEmbeddings 默认只读 DASHSCOPE_API_KEY，故显式传入
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("aliQwen-api"),
)

text = "This is a test document."

# 单条文本 → 一个向量（列表）
query_result = embeddings.embed_query(text)
# sep=""：print 多个参数时用空字符串连接，默认是空格；这里让「文本向量长度：」和数字紧挨着输出，中间不留空
print("文本向量长度：", len(query_result), sep="")

# 多条文本 → 多个向量（列表的列表）
doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)
print(doc_results)
# sep=""：多个参数之间不加空格，输出如「文本向量数量：5，文本向量长度：1024」
print(
    "文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep=""
)
