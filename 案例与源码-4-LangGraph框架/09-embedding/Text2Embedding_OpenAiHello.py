"""
【案例】OpenAI 兼容接口调用阿里百炼 Embedding（Hello 级）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4、Embedding 文本向量化

知识点速览：
- 阿里百炼提供「OpenAI 兼容模式」：用 OpenAI 官方 SDK，只需配置 base_url 和 api_key 即可调百炼的嵌入接口。
- 这样写的好处是同一套代码可切换不同厂商（换 base_url/api_key 即可），便于可移植性。
- client.embeddings.create() 的 input 可以是单字符串或字符串列表；返回的 completion 中含各条文本的 embedding。
- 新加坡与北京地域的 base_url 不同，API Key 也需在对应地域创建。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

input_text = "衣服的质量杠杠的"

# 使用 OpenAI 兼容接口连接阿里百炼：api_key、base_url 指向百炼
client = OpenAI(
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 与 OpenAI 调用方式一致：model 为百炼模型名，input 为文本
completion = client.embeddings.create(model="text-embedding-v4", input=input_text)

print(completion.model_dump_json())

# 【输出示例】
# 注：embedding共1024维度，即len(completion.data[0].embedding) == 1024
# {"data":[{"embedding":[0.02258586511015892,-0.08700370043516159,-0.013521800749003887,-0.05904024466872215,0.027100207284092903,-0.03104848973453045,0.01432843878865242,0.01706676371395588,'.....'],"index":0,"object":"embedding"}],"model":"text-embedding-v4","object":"list","usage":{"prompt_tokens":6,"total_tokens":6},"id":"37989997-27b1-9416-98af-091ae0b5c118"}
