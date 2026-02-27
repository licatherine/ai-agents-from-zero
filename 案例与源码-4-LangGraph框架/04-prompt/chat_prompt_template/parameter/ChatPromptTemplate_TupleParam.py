"""
【案例】用「元组 (role, content)」定义 ChatPromptTemplate 的消息

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、为什么用元组？
  - ChatPromptTemplate 的消息列表里，每一项可以是：字符串、字典、元组、或 Message 类。
  - 元组写法最简洁：("角色", "内容")，角色常用 "system"、"human"、"ai"。
  - 内容里可以写占位符，例如 {name}、{user_input}，在 format_messages 或 invoke 时再传入具体值。

二、和 dict、Message 的对比：
  - 元组：("system", "你是{name}")  ← 本文件
  - 字典：{"role": "system", "content": "你是{name}"}  ← 见 ChatPromptTemplate_DictParam.py
  - Message：SystemMessage(content="你是{name}")  ← 见 ChatPromptTemplate_MessageParam.py
  三种写法效果等价，按习惯选一种即可。
"""

from langchain_core.prompts import ChatPromptTemplate

# 用「(role, content) 元组」的列表定义对话：system、human、ai、再一条 human（带占位符）
chatPromptTemplate = ChatPromptTemplate(
    [
        ("system", "你是一个AI开发工程师，你的名字是{name}。"),
        ("human", "你能帮我做什么?"),
        ("ai", "我能开发很多{thing}。"),
        ("human", "{user_input}"),
    ]
)

# 传入占位符变量，得到消息列表
prompt = chatPromptTemplate.format_messages(name="小谷AI", thing="AI", user_input="7 + 5等于多少")
print(prompt)

# 【输出示例】
# [SystemMessage(content='你是一个AI开发工程师，你的名字是小谷AI。', additional_kwargs={}, response_metadata={}), HumanMessage(content='你能帮我做什么?', additional_kwargs={}, response_metadata={}), AIMessage(content='我能开发很多AI。', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='7 + 5等于多少', additional_kwargs={}, response_metadata={})]