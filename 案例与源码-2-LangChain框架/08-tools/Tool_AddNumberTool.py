"""
【案例】基础加法工具：使用 @tool 装饰器将普通函数转为 LangChain Tool

对应教程章节：第 17 章 - Tools 工具调用 → 4、自定义 Tool → 4.1 使用 @tool 装饰器 / 4.2 基础案例：加法工具

知识点速览：
- @tool 是 LangChain 提供的装饰器，把任意 Python 函数包装成「工具」，模型通过函数的名称、文档字符串（description）和参数来决定是否调用及如何传参。
- 装饰后可通过 tool.name、tool.description、tool.args 查看工具元信息；调用时用 tool.invoke({"a": 1, "b": 2}) 传入参数字典。
- 函数的 docstring 会作为工具的 description 提供给 LLM，因此建议写清「做什么、参数含义」。
"""

from langchain_core.tools import tool


# @tool 装饰器：不写参数时，工具名默认为函数名 add_number，description 取自下方 docstring
@tool
def add_number(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


# 调用工具：invoke 接收一个字典，键为参数名，值为参数值（与函数签名对应）
result = add_number.invoke({"a": 1, "b": 12})
print(result)

print()

# 查看工具常用属性（教程 4.1 表格中的 name、description、args）
print(f"{add_number.name=}\n{add_number.description=}\n{add_number.args=}")

"""
【输出示例】
13

add_number.name='add_number'
add_number.description='两个整数相加'
add_number.args={'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
"""

