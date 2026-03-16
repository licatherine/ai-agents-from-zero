"""
【案例】文本提示词模板：组合多个 PromptTemplate

对应教程章节：第 13 章 - 提示词与消息模板 → 4、文本提示词模板（PromptTemplate）

知识点速览：
- 多个 PromptTemplate 可用 + 按逻辑拼接，形成更长的整体提示（多阶段、多输入源等）。
- 相加后得到新的「组合模板」，format 时传入所有占位符即可。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 方式一：一个 from_template 与一段字符串用 + 拼接 ----------
template1 = (
    PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
    + "内容不超过{length}个字"
)
prompt1 = template1.format(topic="LangChain", length=100)
print(prompt1)

# ---------- 2. 方式二：两个独立模板相加，再一起 format ----------
prompt_a = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
prompt_b = PromptTemplate.from_template("内容不超过{length}个字")
prompt_all = prompt_a + prompt_b
prompt2 = prompt_all.format(topic="LangChain", length=200)
print(prompt2)

"""
【输出示例】
请用一句话介绍LangChain，要求通俗易懂
内容不超过100个字
请用一句话介绍LangChain，要求通俗易懂
内容不超过200个字
"""

