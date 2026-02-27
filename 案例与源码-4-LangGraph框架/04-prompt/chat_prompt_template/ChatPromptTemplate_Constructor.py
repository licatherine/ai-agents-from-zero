"""
【案例】使用构造函数创建 ChatPromptTemplate

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：

一、ChatPromptTemplate 是什么？
  - 大模型（如 GPT、通义）是「多轮对话」形式：有系统设定、用户提问、AI 回复等不同「角色」。
  - ChatPromptTemplate 是 LangChain 里专门用来组装「多角色、多条消息」的提示模板，
    比纯文本的 PromptTemplate 更贴合聊天模型。

二、两种常见创建方式：
  - 方式一：构造函数 ChatPromptTemplate(messages)  ← 本文件演示
  - 方式二：类方法 ChatPromptTemplate.from_messages(messages)  ← 教程里更常用

三、构造函数里 messages 参数支持三种写法（任选一种，混用也可以）：
  - tuple 列表：[(role, content), ...]，例如 ("system", "你是{name}")；
    role 常用 "system" / "human" / "ai"，content 里可写占位符 {变量名}。
  - dict 列表：[{"role": "system", "content": "..."}, ...]
  - Message 类列表：[SystemMessage(...), HumanMessage(...), ...]

四、占位符：在 content 里写 {name}、{user_input} 等，在 format_messages(...) 或 invoke({...}) 时传入具体值。
"""

from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chat_models import init_chat_model

# 用「元组列表」定义对话结构：system 设定角色，human 表示用户，ai 表示助手回复（可带占位符）
chatPromptTemplate = ChatPromptTemplate(
    [
        ("system", "你是一个AI开发工程师，你的名字是{name}。"),
        ("human", "你能帮我做什么?"),
        ("ai", "我能开发很多{thing}。"),
        ("human", "{user_input}"),
    ]
)

# format_messages：把模板里的占位符替换成实际值，得到「消息列表」，可直接交给 model.invoke(prompt)
prompt = chatPromptTemplate.format_messages(
    name="小谷AI", thing="AI", user_input="7 + 5等于多少")
print(prompt)

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print()
print("======================")

# invoke(prompt)：把组装好的消息列表发给模型，返回的 result 是 AIMessage，.content 即模型生成的文本
result = llm.invoke(prompt)
print(result)
print(result.content)
