"""
【案例】使用 OpenAI 官方 SDK 直接调用大模型（不经过 LangChain）

对应教程章节：第 11 章 - Model I/O 与模型接入 → 3、接入大模型

知识点速览：
适合「仅需简单 HTTP 调用」、不打算用 LangChain 链式组件的场景。依赖 openai 包，
运行前在 .env 中配置对应平台的 API Key（如 deepseek-api）。
"""

# ========== 1. 导入与环境 ==========
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 初始化客户端（底层 API，直接请求厂商接口） ==========
client = OpenAI(
    api_key=os.getenv("deepseek-api"),   # 从环境变量读取，此处以 DeepSeek 为例
    base_url="https://api.deepseek.com"   # 可改为其他 OpenAI 兼容地址（如阿里百炼）
)

# ========== 3. 发起对话并打印回复 ==========
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello，你是谁？"},
    ],
    stream=False
)

print(response.choices[0].message.content)
