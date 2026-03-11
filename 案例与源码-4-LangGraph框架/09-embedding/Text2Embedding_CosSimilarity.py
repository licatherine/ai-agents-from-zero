"""
【案例】通过向量计算语义相似度：余弦相似度

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 5、通过向量计算语义相似度

知识点速览：
- 把文本转换成向量有什么用呢？答：最核心的作用是可以通过向量之间的计算，来分析文本与文本之间的相似性。计算的方法有很多种，其中用得最多的是向量余弦相似度。
- 文本转成向量后，可用「余弦相似度」衡量两段文本的语义是否接近：值在 [-1, 1]，越接近 1 表示方向越一致、语义越相似。
- 公式：cos(θ) = (A·B) / (|A||B|)，即两向量点积除以各自模长；numpy 中可用 np.dot 与 np.linalg.norm 实现。
- 先对每条文本调用嵌入接口得到向量，再两两计算相似度，即可比较「谁和谁更像」；常用于检索排序、聚类、去重等。
"""

import dashscope
import os
from http import HTTPStatus
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 准备多句文本，用于比较它们之间的语义相似度
texts = ["我喜欢吃苹果", "苹果是我最喜欢吃的水果", "我喜欢用苹果手机"]

embeddings = []
# 假如要处理图片，请参考https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587
for text in texts:
    input_data = [{"text": text}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        api_key=os.getenv("aliQwen-api"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input=input_data,
    )
    if resp.status_code == HTTPStatus.OK:
        embedding = resp.output["embeddings"][0]["embedding"]
        embeddings.append(embedding)


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度：点积 / (模长之积)，越接近 1 越相似"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


print("文本相似度比较结果:")
print("=" * 60)

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        print(f"文本{i+1} vs 文本{j+1}:")
        print(f"  文本{i+1}: {texts[i]}")
        print(f"  文本{j+1}: {texts[j]}")
        print(f"  余弦相似度: {similarity:.4f}")
        print("-" * 40)

# 【输出示例】
# 文本相似度比较结果:
# ============================================================
# 文本1 vs 文本2:
#   文本1: 我喜欢吃苹果
#   文本2: 苹果是我最喜欢吃的水果
#   余弦相似度: 0.9064
# ----------------------------------------
# 文本1 vs 文本3:
#   文本1: 我喜欢吃苹果
#   文本3: 我喜欢用苹果手机
#   余弦相似度: 0.7656
# ----------------------------------------
# 文本2 vs 文本3:
#   文本2: 苹果是我最喜欢吃的水果
#   文本3: 我喜欢用苹果手机
#   余弦相似度: 0.7421
# ----------------------------------------
