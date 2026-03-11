"""
【案例】在 Redis 向量库中做相似性检索（similarity_search_with_score）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 6、Embedding 存入向量数据库（Redis）

知识点速览：
- 相似性检索：将查询文本向量化后，在向量库中找与查询向量「最接近」的若干条，返回对应文档及相似度（或距离）。
- similarity_search_with_score(query, k)：返回 (Document, score) 的列表，score 一般为距离（越小越相似），可转为相似度如 1 - score。
- 使用前需确保 Redis 中已有数据（如先运行同目录下 RedisVectorStore.py 写入）；index_name、redis_url 须与写入端一致。
"""

from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 嵌入模型（与写入时一致，保证向量空间一致）；需在 .env 中配置 aliQwen-api
embeddingsModel = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)

# 2. 连接已有索引（与 RedisVectorStore.py 中 index_name、redis_url 一致）
vector_store = RedisVectorStore(
    embeddingsModel,
    config=RedisConfig(index_name="newsgroups", redis_url="redis://localhost:26379"),
)

# 3. 查询文本 → 向量化后在库中做相似度检索，取前 k 条
query = "我喜欢用什么手机"
results = vector_store.similarity_search_with_score(query, k=3)

print("=== 查询结果 ===")
for i, (doc, score) in enumerate(results, 1):
    # score 为距离，可转为相似度（如 1 - score，视具体实现而定）
    similarity = 1 - score
    print(f"结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print(f"相似度: {similarity:.4f}")
