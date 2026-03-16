"""
【案例】模型调用：同步 stream（流式输出）

对应教程章节：第 13 章 - 提示词与消息模板 → 2、模型调用方法

知识点速览：
- stream：流式响应。模型「生成一点就返回一点」，不必等整段话写完再一次性返回。
- 适用场景：聊天界面「打字机」效果、长文生成，用户能边等边看到内容，体验更好。
- 与 invoke 区别：invoke 等全部生成完再返回；stream 返回一个可迭代对象，用 for 循环逐块取内容。
"""

from langchain_core.messages.ai import AIMessageChunk


import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------- 1. 实例化模型 ----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ---------- 2. 构建多角色消息（同 invoke）----------
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁"),
]

# ---------- 3. 流式调用：model.stream(messages) ----------
# stream 返回的是一个「生成器」（generator），不会等模型全部生成完才返回；
# 边生成边 yield，每次 for 取到的一小块类型是 AIMessageChunk。所以这里 type(response) 是 generator。
# 对比：invoke/ainvoke 等全部生成完一次性返回 → 类型是 AIMessage（一条完整消息）。
response = model.stream(messages)
print(f"响应类型：{type(response)}")

# 逐块打印：chunk.content 是当前这一小段的文本；end="" 表示不换行，flush=True 表示立即输出到屏幕
for chunk in response:
    print(chunk.content, end="", flush=True)
print("\n")

"""
【输出示例】
响应类型：<class 'generator'>
你好呀！我是小问，一个乐于助人的AI人工助手～😊
我擅长解答问题、帮你理清思路、写文案、做学习规划、整理资料，甚至陪你聊聊天、出出主意。不管是学习上的难题、工作中的困惑，还是生活里的小烦恼，我都很乐意倾听和帮忙！
"""

# 你今天有什么想了解的，或者需要我帮什么忙吗？✨

