"""
【案例】多种 Reducer 并存：同一 State 里 messages 用 add_messages 追加、tags 用 operator.add 拼接列表、
score 用 operator.add 做数值累加；演示从 START 同时连出两条边时两节点并行、再各自到 END 的合并效果。

对应教程章节：第 23 章 - LangGraph API：图与状态 → 2、Graph API 之 State（状态）

知识点速览：
- add_messages：节点只返回「增量」消息，自动与历史合并为一条对话链；invoke 里也可传 openai 风格的 {"role","content"} 字典，运行时会转为 HumanMessage/AIMessage。
- operator.add 作用于列表时相当于拼接（extend）；作用于 float 时为普通加法累加。
- 从同一 START 连到多个节点：这些节点会按图调度执行（本例两节点均从 START 出发再到 END），对同一字段的多次更新由各字段的 Reducer 合并。
- 节点返回 dict 的键即「要更新的状态字段」；未返回的字段保持 Reducer 合并后的结果。
"""

from typing import Annotated, List

import operator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    # 消息历史：add_messages 规约，新消息追加而非整表覆盖（与 StateReducer_AddMessages 一致可用 List）
    messages: Annotated[List, add_messages]
    # 标签列表：operator.add 将各节点返回的列表拼到已有列表后
    tags: Annotated[List[str], operator.add]
    # 累计分数：operator.add 做浮点数相加
    score: Annotated[float, operator.add]


def process_user_message(state: ChatState) -> dict:
    # 获取最新消息；修复/注意：须用 .content 读正文（dict 入参在运行时已转为 HumanMessage 等对象，勿当普通 str 用）
    user_message = state["messages"][-1]
    return {
        # add_messages 会把本条 assistant 回复与历史合并
        "messages": [("assistant", f"Echo: {user_message.content}")],
        "tags": ["processed"],
        "score": 1.0,
    }


def add_sentiment_tag(state: ChatState) -> dict:
    # 本节点不写 messages，则 messages 仅由其他节点更新；tags/score 仍参与 operator.add 合并
    return {"tags": ["positive"], "score": 0.5}


def run_demo():
    builder = StateGraph(ChatState)
    builder.add_node("process", process_user_message)
    builder.add_node("sentiment", add_sentiment_tag)

    # 两节点都从 START 接入：并行分支，各自跑到 END
    builder.add_edge(START, "process")
    builder.add_edge(START, "sentiment")
    builder.add_edge("process", END)
    builder.add_edge("sentiment", END)

    graph = builder.compile()

    # invoke 只接收一个状态字典；messages 可用 dict 列表，与 Chat API 习惯一致
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "tags": ["greeting"],
            "score": 0.0,
        }
    )
    print(result)


if __name__ == "__main__":
    run_demo()

"""
【输出实例】
{'messages': [HumanMessage(content='Hello, how are you?', additional_kwargs={}, response_metadata={}, id='4350252b-ace7-429a-8cc8-67d232d91f42'), AIMessage(content='Echo: Hello, how are you?', additional_kwargs={}, response_metadata={}, id='ab394788-89d0-45f2-a6b0-5252a448ebb1', tool_calls=[], invalid_tool_calls=[])], 'tags': ['greeting', 'processed', 'positive'], 'score': 1.5}
"""
