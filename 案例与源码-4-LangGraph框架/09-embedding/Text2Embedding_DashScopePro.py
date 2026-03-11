"""
【案例】DashScope 多模态 Embedding：文本/图像向量化（进阶）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4、Embedding 文本向量化

知识点速览：
- 多模态嵌入模型（如 tongyi-embedding-vision-plus）可同时处理文本和图像，input 为列表，每项可为 {"text": "..."} 或 {"image": "url"}。
- 返回结构与单模态类似，output.embeddings 为列表，每项含 embedding 向量；多模态时可用于图文匹配、以图搜图等。
- 本示例仅演示文本输入；图像输入需传图片 URL 或 base64，详见百炼多模态嵌入文档。

模型文档链接：https://bailian.console.aliyun.com/?productCode=p_efm&tab=model#/model-market/all?capabilities=ME
"""

import dashscope
import json
import os
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()
# 多模态 call 内部用 get_default_api_key()，必须提前设置 dashscope.api_key，否则报 No api key provided
dashscope.api_key = os.getenv("aliQwen-api")

# 调用多模态 embedding 接口：支持文本或图像输入，input 为列表
resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=[{"text": "尚硅谷AI"}],
)

result = ""

if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage,
    }
    # ensure_ascii=False：中文等非 ASCII 按原样输出，不转成 \uxxxx；indent=4：每层缩进 4 格，便于阅读
    print(json.dumps(result, ensure_ascii=False, indent=4))

print("=================================")
print()

# 从完整结果中取出第一条的 embedding 向量（若需用于相似度计算可单独保存）
embedding_values = result["output"]["embeddings"][0]["embedding"]
print(json.dumps(embedding_values, ensure_ascii=False))
