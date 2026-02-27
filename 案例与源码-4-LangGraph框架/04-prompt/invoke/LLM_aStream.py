"""
【案例】模型调用：异步 astream（异步流式输出）

对应教程章节：第 13 章 - 提示词与消息模板 → 2、模型调用方法

知识点速览：
- astream：异步版 stream，流式返回且不阻塞主线程，适合高并发下的「打字机」体验。
- 用法：astream 返回异步生成器，须用 async for 遍历（不能用普通 for）；在 async 函数内调用，入口 asyncio.run()。
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------- 1. 实例化模型 ----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---------- 2. 构建多角色消息 ----------
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]


# ---------- 3. 异步流式调用（在 async 函数中）----------
async def async_stream_call():
    # astream(messages) 返回的是「异步生成器」，不是 await 一个整体结果
    response = model.astream(messages)
    print(f"响应类型：{type(response)}")  # <class 'async_generator'>

    # 必须用 async for 遍历异步生成器，不能用普通 for
    async for chunk in response:
        print(chunk.content, end="", flush=True)
    print("\n")


# ---------- 4. 运行异步函数 ----------
if __name__ == "__main__":
    asyncio.run(async_stream_call())

# 【输出示例】
# 响应类型：<class 'async_generator'>
# 你好呀！我是小问，一个乐于助人的AI人工助手～😊  
# 我擅长解答问题、帮你理清思路、写文案、做学习规划、整理资料，甚至陪你聊聊天、出出主意。不管是学习上的难题、工作中的困惑，还是生活里的小烦恼，我都很乐意倾听和帮忙！

# 你今天有什么想了解的，或者需要我帮什么忙吗？✨