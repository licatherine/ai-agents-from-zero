"""
【案例】多工具并行调用与聚合回答（V0.3：Agent + AgentExecutor）

对应教程章节：第 21 章 - Agent 智能体 → 5、案例代码

知识点速览：
- Tool 与 Agent 关系：Tool 提供能力（如查天气），Agent 负责决策「何时用、用哪个、如何聚合结果」。
  本案例中一次问题「北京和上海哪个更热」触发多次工具调用，再由 Agent 汇总比较。
- V0.3 流程：模型 + 工具 + 提示模板 → create_tool_calling_agent 得到 Agent → 用 AgentExecutor 执行，
  对应教程「3、Agent 工作原理（V0.3 视角）」：Agent 只做决策，Executor 负责真正调用工具并把结果传回 Agent。
- 关键组件：ChatPromptTemplate 定义对话结构（含 agent_scratchpad 占位符）、AgentExecutor(verbose=True) 便于观察推理与工具调用过程。
"""

import json
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


@tool
def get_weather(loc):
    """
    查询即时天气函数

    :param loc: 必要参数，字符串类型，表示查询天气的城市名称；中国城市需用英文名，如 Beijing、Shanghai。
    :return: OpenWeather API 返回的天气信息，JSON 序列化后的字符串。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY") or "fc19f7b552b4c1ae467e36fe69556668",
        "units": "metric",
        "lang": "zh_cn"
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    print(json.dumps(data))
    return json.dumps(data)


# 初始化大模型，用于理解用户问题并决定是否调用工具、如何组合结果
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义 Agent 的对话结构：system 定角色，human 为用户输入，placeholder 供 Executor 填入中间推理与工具调用记录
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # V0.3 必备：Agent 的「草稿本」，记录多轮推理与工具输出
    ]
)

tools = [get_weather]

# 将 LLM、工具列表、提示模板组装成「可做工具调用决策」的 Agent（尚未执行）
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 负责循环：调用 Agent → 执行其选中的工具 → 把结果写回 agent_scratchpad → 再交给 Agent，直到结束
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 一次问题触发多工具调用（北京、上海天气）并聚合回答
result = agent_executor.invoke({"input": "请问今天北京和上海的天气怎么样，哪个城市更热？"})

print(result)
