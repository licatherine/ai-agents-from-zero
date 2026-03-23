"""
【案例】自定义 Reducer：用函数签名 (current, update) -> 合并结果，解决 operator.mul 与 LangGraph 首次规约时
current 为类型默认值 0.0 导致「乘以 0」的问题；在函数内把「第一次」当作乘以单位元 1.0 处理。

对应教程章节：第 23 章 - LangGraph API：图与状态 → 2、Graph API 之 State（状态）

知识点速览：
- Reducer 可写成普通函数：接收「当前字段值 current」与「本节点返回的增量 update」，返回值写回该字段。
- LangGraph 在合并时可能会先用默认值参与一次规约（float 常为 0.0），因此乘法场景下要区分「尚未有有效旧值」与「旧值就是 0」等边界（本例用 current == 0.0 识别首次，与教程及 StateReducer_OperatorMul 思路一致）。
- 节点仍只返回增量（如 {"factor": 2.0}），由 Reducer 决定如何与 state["factor"] 合并。
"""

from typing import Annotated

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


def MyOperatorMul(current: float, update: float) -> float:
    """自定义乘法 Reducer：首次规约时 current 常为 0.0，按乘法单位元 1.0 处理，再与 update 相乘。"""
    # 第一次调用时 current 往往是类型默认值 0.0，若直接 current * update 会得到 0，后续无法恢复
    if current == 0.0:
        print(f"current:{current}")
        print(f"update:{update}")
        # 等价于从 1.0 开始乘：1.0 * update
        return 1.0 * update
    return current * update


class MultiplyState(TypedDict):
    factor: Annotated[float, MyOperatorMul]


def multiplier(state: MultiplyState) -> dict:
    # 节点返回的 update 会与 state["factor"] 经 MyOperatorMul 合并
    return {"factor": 2.0}


def run_demo():
    print("使用自定义reducer解决乘法问题:")
    builder = StateGraph(MultiplyState)
    builder.add_node("multiplier", multiplier)
    builder.add_edge(START, "multiplier")
    builder.add_edge("multiplier", END)
    graph = builder.compile()

    # 初始 factor=5.0 与节点返回 2.0 经 Reducer 合并为 5.0 * 2.0 = 10.0
    result = graph.invoke({"factor": 5.0})
    print(f"初始状态: {{'factor': 5.0}}")
    print(f"执行结果: {result}")
    print(f"解释: 5.0 * 2.0 = 10.0\n")


if __name__ == "__main__":
    run_demo()

"""
【输出实例】
使用自定义reducer解决乘法问题:
current:0.0
update:5.0
初始状态: {'factor': 5.0}
执行结果: {'factor': 10.0}
解释: 5.0 * 2.0 = 10.0
"""
