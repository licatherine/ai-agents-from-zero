"""
【案例】PromptTemplate 的 format() 方法

对应教程章节：第 13 章 - 提示词与消息模板 → 4、文本提示词模板（PromptTemplate）

知识点速览：
- format(kwargs)：为模板里所有占位符赋值，返回**一条字符串**。若少传了某个变量会报错。
- 最常用：得到字符串后可以交给 model.invoke(prompt)（若模型支持纯文本输入）。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建模板（from_template 自动推断 {role}、{question}）----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. format 填入变量，得到「最终一条提示词字符串」----------
prompt = template.format(role="python开发", question="二分查找算法怎么写？")
print(prompt)
# 类型是 str，可直接传给 model.invoke(prompt)（若模型支持）
print(type(prompt))

# 【输出示例】
# 你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：二分查找算法怎么写？
# <class 'str'>