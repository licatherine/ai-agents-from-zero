"""
【案例】PromptTemplate 的 partial() 方法

对应教程章节：第 13 章 - 提示词与消息模板 → 4、文本提示词模板（PromptTemplate）

知识点速览：
- partial(kwargs)：固定模板中部分占位符，返回新模板，之后只需 format 剩余变量。
- 典型用法：先 partial(role="...") 定好角色，后面多次 format(question="...") 问不同问题，避免重复传 role。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建带两个占位符的模板 ----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. partial(role="python开发")：固定 role，得到「新模板」----------
# 新模板只剩 {question} 需要填，适合多轮只换问题、不换角色的场景
partial = template.partial(role="python开发")
print(partial)
print(type(partial))
print()

# ---------- 3. 对新模板 format，只传 question 即可 ----------
prompt = partial.format(question="冒泡排序怎么写？")
print(prompt)
print(type(prompt))

"""
【输出示例】
input_variables=['question'] input_types={} partial_variables={'role': 'python开发'} template='你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}'
<class 'langchain_core.prompts.prompt.PromptTemplate'>
"""

# 你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写？
# <class 'str'>

