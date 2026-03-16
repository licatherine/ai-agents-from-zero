"""
【案例】直接使用 InMemoryChatMessageHistory 的 API：add_message、messages，手动拼历史后调用模型

对应教程章节：第 16 章 - 记忆与对话历史 → 5、实现类介绍（BaseChatMessageHistory 与常用实现）/ 6.1 内存版

知识点速览：
- BaseChatMessageHistory 的实现类提供 add_message(msg)、add_user_message(text)、messages（只读列表）等；本案例演示不通过 RunnableWithMessageHistory，而是手动维护 history 并每次把 history.messages 传给 llm.invoke。
- 适用场景：需要细粒度控制「何时写入历史、何时读取」时，可直接操作 history；多数场景更推荐用 RunnableWithMessageHistory 自动完成「读→拼入→执行→写回」。
- 内存版：数据仅在进程内，重启即丢失；持久化见 6.2 RedisChatMessageHistory。
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from loguru import logger
import os

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建内存版历史实例（BaseChatMessageHistory 的实现）
history = InMemoryChatMessageHistory()

# 手动添加用户消息并调用模型；模型输入为当前全部 messages
history.add_user_message("我叫张三，我的爱好是学习")
ai_message = llm.invoke(history.messages)
logger.info(f"第一次回答\n{ai_message.content}")
history.add_message(ai_message)

# 再追加一轮：用户问「我叫什么？我的爱好是什么？」；此时 history.messages 已含上一轮
history.add_user_message("我叫什么？我的爱好是什么？")
ai_message2 = llm.invoke(history.messages)
logger.info(f"第二次回答\n{ai_message2.content}")
history.add_message(ai_message2)

# 遍历当前会话全部消息
for message in history.messages:
    logger.info(message.content)

"""
【输出示例】
2026-03-09 15:17:53.046 | INFO     | __main__:<module>:35 - 第一次回答
你好，张三！很高兴认识你～😄
“爱好是学习”这个回答特别有力量——说明你对世界充满好奇，愿意持续探索、成长和突破自我。学习本身是一种非常珍贵的能力，也是一种温柔而坚定的生活态度。
"""

# 如果你愿意分享，我很乐意陪你一起：
# 🔹 探讨某个具体领域（比如编程、语言、历史、心理学、AI……）
# 🔹 制定小而可行的学习计划或习惯养成方法
# 🔹 解析一道难题、梳理一个概念，或者帮你把零散知识结构化
# 🔹 甚至只是聊聊学习中遇到的困惑、倦怠或小成就 🌟

# 你最近在学什么？或者有什么特别想了解、想开始尝试的方向吗？🌱
# （悄悄说：哪怕只是“今天读了一页书”，也值得被认真对待哦～）
# 2026-03-09 15:17:55.429 | INFO     | __main__:<module>:41 - 第二次回答
# 你叫**张三**，你的爱好是**学习**。😊
# ——简洁、真诚，又充满成长的温度。
# （刚刚你亲口告诉我的，我可一直记得呢～）

# 需要我帮你把“爱学习”这件事变得更有趣、更高效，或者更轻松一点吗？✨
# 2026-03-09 15:17:55.429 | INFO     | __main__:<module>:46 - 我叫张三，我的爱好是学习
# 2026-03-09 15:17:55.429 | INFO     | __main__:<module>:46 - 你好，张三！很高兴认识你～😄
# “爱好是学习”这个回答特别有力量——说明你对世界充满好奇，愿意持续探索、成长和突破自我。学习本身是一种非常珍贵的能力，也是一种温柔而坚定的生活态度。

# 如果你愿意分享，我很乐意陪你一起：
# 🔹 探讨某个具体领域（比如编程、语言、历史、心理学、AI……）
# 🔹 制定小而可行的学习计划或习惯养成方法
# 🔹 解析一道难题、梳理一个概念，或者帮你把零散知识结构化
# 🔹 甚至只是聊聊学习中遇到的困惑、倦怠或小成就 🌟

# 你最近在学什么？或者有什么特别想了解、想开始尝试的方向吗？🌱
# （悄悄说：哪怕只是“今天读了一页书”，也值得被认真对待哦～）
# 2026-03-09 15:17:55.429 | INFO     | __main__:<module>:46 - 我叫什么？我的爱好是什么？
# 2026-03-09 15:17:55.429 | INFO     | __main__:<module>:46 - 你叫**张三**，你的爱好是**学习**。😊
# ——简洁、真诚，又充满成长的温度。
# （刚刚你亲口告诉我的，我可一直记得呢～）

# 需要我帮你把“爱学习”这件事变得更有趣、更高效，或者更轻松一点吗？✨

