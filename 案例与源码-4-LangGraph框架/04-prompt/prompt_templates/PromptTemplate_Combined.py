"""
【第 11 章 - 文本提示词模板：组合多个 Prompt】
对应笔记：11-提示词与输出解析.md → 1.6 PromptTemplate 的用法（组合使用）

知识点速览：
- 可将多个子 Prompt 按逻辑拼接，形成更长的整体提示（多阶段、多输入源等）。
- 写法：两个 PromptTemplate 用 + 相加，得到一个新的「组合模板」，再 format 时传入所有占位符。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 方式一：一个 from_template 与一段字符串用 + 拼接 ----------
template1 = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n") + "内容不超过{length}个字"
prompt1 = template1.format(topic="LangChain", length=100)
print(prompt1)

# ---------- 2. 方式二：两个独立模板相加，再一起 format ----------
prompt_a = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
prompt_b = PromptTemplate.from_template("内容不超过{length}个字")
prompt_all = prompt_a + prompt_b
prompt2 = prompt_all.format(topic="LangChain", length=200)
print(prompt2)

# 【输出示例】
# 请用一句话介绍LangChain，要求通俗易懂
# 内容不超过100个字
# 请用一句话介绍LangChain，要求通俗易懂
# 内容不超过200个字