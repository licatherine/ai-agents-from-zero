"""
【案例】不接入大模型的业务图：自定义加法/减法节点，演示「State 在节点间传递、每步更新同一字段」的完整流程，对应教程中「图 + 状态」的入门示例。

对应教程章节：第 22 章 - LangGraph 概述与快速入门 → 2、HelloWorld 快速入门

知识点速览：
- 用 dict 作为 State 类型时，无需预定义 TypedDict，直接传字典即可；适合快速试验。
- 节点函数接收 state，返回要更新的键值对（如 {"x": state["x"] + 1}），LangGraph 会按默认规则合并（覆盖）。
- add_edge 串联 START → addition → subtraction → END，形成固定执行顺序。
- graph.edges / graph.nodes 可查看当前图已注册的边与节点；compile() 后用 invoke(initial_state) 执行。
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph


def addition(state):
    """加法节点：将 state 中的 x 加 1。"""
    print(f"加法节点收到的初始值:{state}")
    return {"x": state["x"] + 1}


def subtraction(state):
    """减法节点：将 state 中的 x 减 2。"""
    print(f"减法节点收到的初始值:{state}")
    return {"x": state["x"] - 2}


# 使用 dict 作为状态类型，无需预定义 TypedDict
graph = StateGraph(dict)
graph.add_node("addition", addition)
graph.add_node("subtraction", subtraction)

# 定义执行顺序：START → addition → subtraction → END
graph.add_edge(START, "addition")
graph.add_edge("addition", "subtraction")
graph.add_edge("subtraction", END)

# 查看图的边与节点（调试用）
print(graph.edges)
print(graph.nodes)

# 编译图构建器生成计算图
app = graph.compile()
# invoke()方法只接收状态字典作为核心参数，定义一个初始状态字典，包含键值对"x": 5
initial_state = {"x": 5}
# invoke 只接收一个核心参数：初始状态字典
result = app.invoke(initial_state)
print(f"最后的结果是:{result}")

# 打印图的可视化结构
print(app.get_graph().print_ascii())
print()
# 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
print(app.get_graph().draw_mermaid())

"""
【输出示例】
{('subtraction', '__end__'), ('addition', 'subtraction'), ('__start__', 'addition')}
{'addition': StateNodeSpec(runnable=addition(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class 'dict'>, retry_policy=None, cache_policy=None, ends=(), defer=False), 'subtraction': StateNodeSpec(runnable=subtraction(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class 'dict'>, retry_policy=None, cache_policy=None, ends=(), defer=False)}
加法节点收到的初始值:{'x': 5}
减法节点收到的初始值:{'x': 6}
最后的结果是:{'x': 4}
 +-----------+
 | __start__ |
 +-----------+
        *
        *
        *
  +----------+
  | addition |
  +----------+
        *
        *
        *
+-------------+
| subtraction |
+-------------+
        *
        *
        *
  +---------+
  | __end__ |
  +---------+
None

---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        addition(addition)
        subtraction(subtraction)
        __end__([<p>__end__</p>]):::last
        __start__ --> addition;
        addition --> subtraction;
        subtraction --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
