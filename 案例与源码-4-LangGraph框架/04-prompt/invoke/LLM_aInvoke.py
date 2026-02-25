"""
【第 11 章 - 模型调用：异步 ainvoke】
对应笔记：11-提示词与输出解析.md → 1.4 模型调用方法 → 1.4.1 普通调用（异步）

知识点速览：
- ainvoke：异步版的 invoke。在等待模型响应的过程中不会阻塞主线程，可同时发起多个请求。
- 适用场景：高并发 Web 服务（如 FastAPI）、大批量请求、需要并发的脚本。
- Python 用法：ainvoke 是异步方法，必须在 async 函数里用 await 调用，并通过 asyncio.run() 驱动运行。
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------- 1. 实例化模型（与同步版本相同）----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


async def main():
    """异步主函数：必须用 async def，内部用 await 调用 ainvoke。"""
    # ---------- 2. 异步调用一条请求 ----------
    # 这里传入的是「纯字符串」也可以（部分模型支持）；多角色时仍可传 messages 列表。
    # await：等待模型返回时不阻塞事件循环，其他协程可同时执行。
    response = await model.ainvoke("解释一下LangChain是什么，简洁回答100字以内")
    print(f"响应类型：{type(response)}")
    print(response.content_blocks)


# ---------- 3. 运行异步程序 ----------
# asyncio.run(main()) 会启动事件循环、执行 main()，直到 main() 结束。初学者只需记住：异步入口这样写。
if __name__ == "__main__":
    asyncio.run(main())

# 【输出示例】
# 响应类型：<class 'langchain_core.messages.ai.AIMessage'>
# [{'type': 'text', 'text': 'LangChain是一个开源框架，用于构建基于大语言模型（LLM）的应用程序。它提供模块化组件（如链、代理、记忆、工具集成等），支持提示工程、数据检索增强（RAG）、多步推理和外部工具调用，简化LLM应用的开发、编排与部署。'}]