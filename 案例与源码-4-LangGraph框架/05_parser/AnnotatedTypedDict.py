"""
【案例】Python 入门：Annotated 与 TypedDict 的「仅类型提示、无运行时校验」（对应教程 2.5 节理解 Annotated）

一、本文件在说明什么？
  - 用 **typing.TypedDict** 和 **typing.Annotated** 定义 Person，其中 age2 用了 Annotated[int, "年龄，范围0-150"]。
  - 重点：**Annotated 里的描述（如「年龄，范围0-150」）只是元数据**，给文档、静态类型检查或 LangChain 生成提示用；**Python 运行时不会按这个范围做校验**，所以 age2=188 不会报错。

二、为什么 188 不报错？
  - Annotated 的设计目的是「给类型加注释/元数据」，不是「给类型加运行时校验规则」。
  - Python 的类型提示（Type Hints）总体是「装饰性」的：运行时只关心是不是 int，不关心 Annotated 里写的范围。若需要运行时校验，要用 Pydantic 等库（见 AnnotatedPydantic.py）。
"""

from typing import Annotated, TypedDict

# Annotated[int, "年龄，范围0-150"]：类型仍是 int，后面的字符串只是元数据，运行时不会做 0–150 的校验
Age = Annotated[int, "年龄，范围0-150"]


class Person(TypedDict):
    name: str
    age: int
    age2: Age  # 本质还是 int，元数据 "年龄，范围0-150" 不参与运行时校验


# TypedDict 实例化时不会校验 age2 是否在 0–150，只要类型是 int 即可
p = Person(name="z3", age=111, age2=188)
print(p)

# p = Person(name="z3", age="1111")  # 若用字符串赋给 age，部分环境可能报错，取决于具体运行时
