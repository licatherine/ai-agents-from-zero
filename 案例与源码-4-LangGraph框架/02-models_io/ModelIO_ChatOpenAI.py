"""
【案例】使用 LangChain ChatOpenAI 调用大模型（OpenAI 兼容接口）

对应教程章节：第 11 章 - Model I/O 与模型接入 → 3、接入大模型

知识点速览：
通过 langchain_openai 的 ChatOpenAI + base_url 接入通义/阿里百炼等兼容接口，
可与 Prompt、Chain、Agent、Memory 等无缝配合。依赖 langchain-openai，运行前在 .env 中配置 API Key。
"""

# ========== 1. 导入与环境 ==========
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 初始化聊天模型（OpenAI 兼容接口） ==========
chat_llm = ChatOpenAI(
    model="qwen-plus",                    # 可按需更换，模型列表见阿里云文档
    api_key=os.getenv("aliQwen-api"),     # 或 os.getenv("QWEN_API_KEY")
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========== 3. 调用模型并打印回复 ==========
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
]

response = chat_llm.invoke(messages)
print(response.content)
