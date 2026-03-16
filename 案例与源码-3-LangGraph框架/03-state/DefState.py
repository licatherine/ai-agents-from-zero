"""
【案例】最简 State 定义与「无中间节点」图：用 TypedDict 定义状态，构建一条直接从 START 到 END 的边，验证 invoke(initial_state) 的用法。

对应教程章节：第 23 章 - LangGraph API：图与状态 → 2、Graph API 之 State（状态）

知识点速览：
- State 用 TypedDict 声明字段及类型，图中所有节点读写同一份状态结构。
- add_edge(START, END) 表示没有业务节点，图从入口直接到出口，常用于验证「状态原样透传」。
- invoke() 只接收一个核心位置参数：状态字典；不要传入多个独立参数。可选第二参数为 config。
- 嵌套类型（如 process_data: dict）在 initial_state 中需传入合法字典。
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class BasicState(TypedDict):
    """基本 State 定义：包含常用字段示例。"""

    user_input: str
    response: str
    count: int
    process_data: dict


# 创建状态图并指定状态结构
basicState = StateGraph(BasicState)
# 无中间节点：直接从 START 到 END，状态会原样透传
basicState.add_edge(START, END)
app = basicState.compile()

# invoke 只接收一个核心参数（状态字典）；process_data 为 dict，需传入嵌套字典
initial_state = {
    "user_input": "a",
    "response": "resp",
    "count": 25,
    "process_data": {"k1": "v1"},
}

result = app.invoke(initial_state)
print("执行结果：", result)

"""
【输出实例】
执行结果： {'user_input': 'a', 'response': 'resp', 'count': 25, 'process_data': {'k1': 'v1'}}
"""

