"""
【案例】隐式使用 MessagesPlaceholder：("placeholder", "{变量名}") 简写

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、隐式 vs 显式
  - 显式：MessagesPlaceholder("memory")  ← 见 ChatPromptTemplate_ExplicitPlaceholder.py
  - 隐式：("placeholder", "{memory}") 是 MessagesPlaceholder("memory") 的简写，效果相同。
  - 隐式写法少写一个类名，列表里全是「元组」形式，风格统一；初学者两种任选一种即可。

二、调用方式一样
  - 都是 invoke({"memory": [消息列表], "question": "当前问题"})。
  - 变量名要和占位符一致：这里占位符是 {memory}，所以传入的键为 "memory"。

三、小结
  - 需要「在提示词里插入一段动态消息（如聊天历史）」时，用 MessagesPlaceholder 或 ("placeholder", "{变量名}")。
  - 显式更直观，隐式更简洁；本文件演示隐式写法。
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 隐式：("placeholder", "{memory}") 等价于 MessagesPlaceholder("memory")
# 表示这里留一个「坑」，invoke 时用 "memory" 对应的消息列表填充
prompt = ChatPromptTemplate.from_messages([
    ("placeholder", "{memory}"),
    ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
    ("human", "{question}")
])

# 传入 memory（历史消息列表）和 question（当前问题）
prompt_value = prompt.invoke({
    "memory": [
        HumanMessage("我的名字叫亮仔，是一名程序员"),
        AIMessage("好的，亮仔你好")
    ],
    "question": "请问我的名字叫什么？"
})
print(prompt_value.to_string())

# 【输出示例】
# Human: 我的名字叫亮仔，是一名程序员
# AI: 好的，亮仔你好
# System: 你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题
# Human: 请问我的名字叫什么？