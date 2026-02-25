"""
【第 11 章 - 文本提示词模板：from_template 创建 PromptTemplate】
对应笔记：11-提示词与输出解析.md → 1.6 文本提示词模板（PromptTemplate）

知识点速览：
- from_template("...")：只需传入带占位符的字符串，LangChain 会**自动推断**有哪些变量（即所有 {xxx}），无需手写 input_variables。
- 适合快速写模板；和构造函数二选一即可。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 用 from_template 创建：自动从字符串里识别 {role}、{question} ----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. format 填入变量，得到最终一条字符串 ----------
prompt = template.format(role="python开发", question="快速排序怎么写？")
print(prompt)
print("\n\n")

# ---------- 3. 再举一例：{topic}、{type} 两个占位符 ----------
template = PromptTemplate.from_template("请给我一个关于{topic}的{type}解释。")
prompt = template.format(topic="量子力学", type="详细")
print(prompt)  # 请给我一个关于量子力学的详细解释。
# 类型是 str，可直接传给 model.invoke(prompt)（若模型支持）
print(type(prompt)) # <class 'str'>

# 【输出示例】
# 你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：快速排序怎么写？

# 请给我一个关于量子力学的详细解释。
# <class 'str'>