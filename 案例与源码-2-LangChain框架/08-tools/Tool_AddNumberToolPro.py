"""
【案例】加法工具（Pydantic 版）：用 args_schema 绑定参数模型，让模型看到规范参数说明

对应教程章节：第 17 章 - Tools 工具调用 → 4、自定义 Tool → 4.3 Pydantic 与参数 schema

知识点速览：
- 用 Pydantic 的 BaseModel 定义参数类（如 FieldInfo），字段用 Field(description=...) 写清说明，便于 LLM 生成正确参数。
- @tool(args_schema=FieldInfo) 将参数模型绑定到工具，tool.args 会包含各字段的类型与 description。
- 可选参数：@tool 还可传 name、description、return_direct 等；return_direct=True 时 Agent 调用该工具后直接返回结果给用户。
"""
from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field


# Pydantic 模型：定义「工具参数」的结构与描述，LLM 会据此生成调用参数
class FieldInfo(BaseModel):
    """定义加法运算所需的参数信息（该 docstring 可能被用作工具 description 的补充）"""

    a: int = Field(description="第1个参数")
    b: int = Field(description="第2个参数")


# args_schema=FieldInfo：把参数模型绑定到工具，模型看到的工具描述中会包含 a、b 的类型与 description
@tool(args_schema=FieldInfo)
def add_number(a: int, b: int) -> int:
    return a + b


# 打印工具属性：带 args_schema 时，args 中会包含 Field 的 description
logger.info(f"name = {add_number.name}")
logger.info(f"args = {add_number.args}")
logger.info(f"description = {add_number.description}")
logger.info(f"return_direct = {add_number.return_direct}")

# 调用工具：传入字典，Pydantic 会做类型校验与转换
res = add_number.invoke({"a": 1, "b": 2})
logger.info(res)

"""
【输出示例】
name = add_number
args = {'a': {'description': '第1个参数', 'title': 'A', 'type': 'integer'}, 'b': {'description': '第2个参数', 'title': 'B', 'type': 'integer'}}
description = 定义加法运算所需的参数信息（或函数 docstring）
return_direct = False
3
"""

