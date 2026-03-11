"""
【案例】DashScope 原生调用：单句文本向量化（Hello 级）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4、Embedding 文本向量化

知识点速览：
- Embedding（嵌入）是把文本变成一串数字（向量）的过程，便于用相似度比较语义。
- 阿里云百炼提供文本嵌入 API，可直接用 dashscope.TextEmbedding.call() 传入模型名和文本，返回向量结果。
- 返回的 resp 中，resp.output 含嵌入结果；若需单独取出向量数组，可从 output.embeddings[0].embedding 获取。

模型文档链接：https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587
"""

import os
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("aliQwen-api")

# 待向量化的单句文本
input_text = "衣服的质量杠杠的"

# 调用百炼文本嵌入接口：model 指定模型，input 为文本（或文本列表）
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=input_text,
)

if resp.status_code == HTTPStatus.OK:
    print(resp)

# 【输出示例】
# {"status_code": 200, "request_id": "0a76a5db-f4af-4e5a-b0c4-1689d81ba154", "code": "", "message": "", "output": {"embeddings": [{"embedding": [0.02258586511015892, -0.08700370043516159, -0.013521800749003887, -0.05904024466872215, 0.027100207284092903, -0.03104848973453045, 0.01432843878865242, -0.0008265386568382382,……], "text_index": 0}]}, "usage": {"total_tokens": 6}}
