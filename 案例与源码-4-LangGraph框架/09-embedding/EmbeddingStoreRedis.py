"""
【案例】将 Document 列表向量化并写入 Redis（langchain_community）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 6、Embedding 存入向量数据库（Redis）

知识点速览：
- 向量数据库用于存储各类数据（文本、图像、视频等）经嵌入模型得到的向量，查询时按相似度返回最接近的几条；Redis（RedisStack）可通过 RediSearch 支持向量检索。
- LangChain 的 Redis.from_documents()：传入 Document 列表 + embedding 实例，会自动对每个 document 的 page_content 做向量化并写入 Redis。
- 写入后可调用 as_retriever() 得到检索器，用 invoke(查询文本) 做相似检索，返回与查询最相关的 Document 列表。
- redis_url 需与本地 Redis 端口一致（如 26379）；索引名 index_name 用于区分不同业务的数据。

Redis.from_documents 常用参数与基础用法：
- 固定写法：Redis.from_documents(documents=..., embedding=..., redis_url=..., index_name=...)，前两个必填。
- documents：Document 列表，每个 Document 需有 page_content，可选 metadata。
  能否用图片/音频取决于 embedding：本处用文本嵌入模型，故 page_content 只能是文本。
  若用多模态嵌入或先算好向量，也可把图像/音频的向量写入同一 Redis；纯图片/音频通常需用 add_embeddings 等接口（from_documents 默认只对文本调 embed_documents）。
- embedding：嵌入模型实例（如 DashScopeEmbeddings），用于把 page_content 转成向量；文本模型只处理文本，多模态模型可处理图文等。
- redis_url：Redis 连接地址，如 "redis://localhost:26379"（RedisStack 常用 26379）。
- index_name：索引名，同一 Redis 可建多个索引区分业务；检索时须用同一 index_name。
- 可选：key_prefix（键前缀）、distance_metric（相似度度量，如 COSINE）等，见 langchain_community.vectorstores.redis 文档。
"""

# pip install langchain-community dashscope redis redisvl
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)

# 2. 构造 Document 列表：page_content 为正文，metadata 可存来源等
# 本脚本用文本嵌入模型，故此处仅为文本；存图像/音频向量需多模态嵌入或 add_embeddings
texts = [
    "通义千问是阿里巴巴研发的大语言模型。",
    "Redis 是一个高性能的键值存储系统，支持向量检索。",
    "LangChain 可以轻松集成各种大模型和向量数据库。",
]
documents = [
    Document(page_content=text, metadata={"source": "manual"}) for text in texts
]

# 3. 一次性写入 Redis：内部会对每个 document 调用 embeddings 并建索引
# 写法固定：documents + embedding 必填，redis_url、index_name 按环境与业务填写
vector_store = Redis.from_documents(
    documents=documents,
    embedding=embeddings,
    redis_url="redis://localhost:26379",
    index_name="my_index11",
)

# 4. 得到检索器，按相似度取前 k 条（此处 k=2）
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("LangChain 和 Redis 怎么结合？")
for res in results:
    print(res.page_content)
