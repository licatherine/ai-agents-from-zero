"""
【案例】使用 langchain_redis 将文本写入 Redis 向量库（add_texts）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 6、Embedding 存入向量数据库（Redis）

知识点速览：
- langchain_redis 的 RedisVectorStore：需传入 Embedding 实例和 RedisConfig（index_name、redis_url），用于向量存储与检索。
- add_texts(texts, metadata)：将字符串列表与可选元数据写入向量库；内部会调用 embedding.embed_documents(texts) 得到向量再写入。
- add_texts 内部会调用 embed_documents，因此只需传 texts 与 metadata，无需为写入而额外先执行 embed_documents；若需在写入前对向量或元数据做自定义处理，可先自行 embed 再使用 add_embeddings 等接口。
- 返回的 ids 可用于后续按 ID 删除或更新；index_name 需与检索端一致（如与 RedisVectorStore_SimilaritySearch 同用 newsgroups）。
"""

from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化嵌入模型
embeddingsModel = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)

# 2. 待写入的文本及（可选）元数据
texts = [
    "我喜欢吃苹果",
    "苹果是我最喜欢吃的水果",
    "我喜欢用苹果手机",
]


# 批量转成向量（若仅做存储可交给 add_texts 内部调用，此处演示可先取向量查看）
embeddings = embeddingsModel.embed_documents(texts)
for i, vec in enumerate(embeddings, 1):
    print(f"文本 {i}: {texts[i-1]}")
    print(f"向量长度: {len(vec)}")
    print(f"前5个向量值: {vec[:10]}\n")

# 定义每条文本对应的元数据信息
metadata = [{"segment_id": "1"}, {"segment_id": "2"}, {"segment_id": "3"}]

# 3. Redis 连接与索引名（需与检索案例一致）
config = RedisConfig(
    index_name="newsgroups",
    redis_url="redis://localhost:26379",
)

# 创建Redis向量存储实例
vector_store = RedisVectorStore(embeddingsModel, config=config)

# 4. 将文本与元数据写入向量库（add_texts 内部会调 embed_documents，无需先算向量）
ids = vector_store.add_texts(texts, metadata)

# 打印前5个存储记录的ID
print(ids[0:5])

# 【输出示例】
# 文本 1: 我喜欢吃苹果
# 向量长度: 1024
# 前5个向量值: [-0.04062262922525406, 0.03663524612784386, -0.07420649379491806, 0.003861021716147661, -0.06338627636432648, -0.02864176034927368, -0.027855515480041504, 0.03684116527438164, -0.023493731394410133, -0.027892956510186195]

# 文本 2: 苹果是我最喜欢吃的水果
# 向量长度: 1024
# 前5个向量值: [-0.03398064523935318, 0.04141449183225632, -0.06892527639865875, 0.005737593863159418, -0.06951850652694702, -0.04560413956642151, -0.04171110317111015, 0.04508506879210472, -0.04549290984869003, -0.017945043742656708]

# 文本 3: 我喜欢用苹果手机
# 向量长度: 1024
# 前5个向量值: [-0.052530914545059204, 0.006213586777448654, -0.11318981647491455, -0.023480866104364395, -0.036481890827417374, -0.04383847862482071, 0.005418661516159773, 0.02874900959432125, 0.0019732017535716295, 0.01118539646267891]

# ['newsgroups:01KKDZ5MRGBDPWJHDZZWH4W2Q6', 'newsgroups:01KKDZ5MRGBDPWJHDZZWH4W2Q7', 'newsgroups:01KKDZ5MRGBDPWJHDZZWH4W2Q8']
