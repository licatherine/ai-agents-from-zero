"""
【第 11 章 - PromptTemplate 的 invoke() 方法】
对应笔记：11-提示词与输出解析.md → 1.6.1 PromptTemplate 的用法（常用方法）

知识点速览：
- invoke(input)：和 format 类似，传入变量（字典或关键字），但返回的是 **PromptValue** 对象，不是字符串。
- PromptValue 可 .to_string() 得到字符串，或 .to_messages() 转成消息列表，便于和 LangChain 的链（LCEL）衔接。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建模板 ----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. 用 invoke 传入变量（字典），得到 PromptValue 对象 ----------
prompt = template.invoke({"role": "python开发", "question": "冒泡排序怎么写？"})
print(prompt)
print(type(prompt))
print()

# ---------- 3. 从 PromptValue 取内容：to_string() 得到整段字符串 ----------
print(prompt.to_string())
print(type(prompt.to_string()))
print()

# ---------- 4. to_messages()：转成「消息列表」，可接入需要多角色消息的链 ----------
print(prompt.to_messages())
print(type(prompt.to_messages()))
