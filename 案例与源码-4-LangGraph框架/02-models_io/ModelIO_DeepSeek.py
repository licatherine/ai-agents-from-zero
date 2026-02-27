"""
【案例】使用 langchain-deepseek 原生集成调用 DeepSeek

对应教程章节：第 11 章 - Model I/O 与模型接入 → 3、接入大模型

知识点速览：
使用厂商官方集成包时无需手动写 base_url，SDK 内已封装。依赖 pip install langchain-deepseek，
运行前在 .env 中配置 deepseek-api。
"""

# ========== 1. 导入与环境 ==========
import os
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 初始化 DeepSeek 聊天模型 ==========
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("deepseek-api"),
)

# ========== 3. 调用并打印回复 ==========
print(model.invoke("什么是 LangChain？100 字以内回答，简洁。").content)
