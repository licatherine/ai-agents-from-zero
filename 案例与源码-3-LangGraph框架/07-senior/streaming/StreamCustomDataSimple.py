"""
【案例】自定义流（custom）最简版：在节点内通过 get_stream_writer() 写入任意可序列化数据，stream 侧用 custom 接收。

对应教程章节：第 25 章 - LangGraph 高级特性 → 1、流式处理（Streaming）

知识点速览：
- get_stream_writer() 仅在图执行（stream/astream）过程中有效，在 writer(...) 中传入 dict 等即可推送。
- 调用 graph.stream 时 stream_mode 须包含 "custom"（可与其他模式组合，如 ["values", "custom"]）。
- 自定义块与状态 updates 相互独立：前者给 UI/日志用，后者仍走 State 的 Reducer。
"""

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    answer: str


def node(state: State):
    writer = get_stream_writer()
    writer({"custom_key": "欢迎来到线上Agent班级学习，O(∩_∩)O"})
    return {"answer": "some data"}


def main():
    graph = (
        StateGraph(State)
        .add_node(node)
        .add_edge(START, "node")
        .add_edge("node", END)
        .compile()
    )

    # 仅 custom：for chunk in graph.stream({"query": "example"}, stream_mode=["custom"]): print(chunk)
    # custom + updates：for mode, chunk in graph.stream(..., stream_mode=["updates", "custom"]): ...
    for chunk in graph.stream({"query": "example"}, stream_mode=["values", "custom"]):
        print(chunk)


if __name__ == "__main__":
    main()

"""
【输出示例】
('values', {'query': 'example'})
('custom', {'custom_key': '欢迎来到线上Agent班级学习，O(∩_∩)O'})
('values', {'query': 'example', 'answer': 'some data'})
"""
