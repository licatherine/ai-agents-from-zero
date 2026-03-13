"""
【案例】多工具并行调用与聚合回答（V1.0：create_agent 一步创建 + 结构化输出）

对应教程章节：第 21 章 - Agent 智能体 → 5、案例代码

知识点速览：
- V1.0 与 V0.3 对比：不再手写 PromptTemplate、create_tool_calling_agent、AgentExecutor，改为
  create_agent(model, tools, system_prompt, response_format=...) 一步得到可调用的 Agent，对应教程「4、Agent 工作原理（V1.0）」。
- 结构化输出：通过 response_format 指定 TypedDict（如 WeatherCompareOutput），Agent 的返回中会包含
  structured_response 字段，便于程序化处理（如比温度、写结论），而不必从自然语言里再解析。
- 调用方式：agent.invoke({"input": "..."})，底层由 LangGraph 负责「推理 → 行动 → 反馈」循环。
"""

import os
import json
import httpx
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(loc: str) -> dict:
    """
    查询即时天气函数
    :param loc: 城市英文名，如 Beijing、Shanghai。
    :return: OpenWeather API 返回的天气信息（JSON 字符串）。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn"
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    return json.dumps(data, ensure_ascii=False)


# 定义结构化输出：Agent 最终回答会按此结构填充，便于代码中直接取字段
class WeatherCompareOutput(TypedDict):
    beijing_temp: float
    shanghai_temp: float
    hotter_city: str
    summary: str


model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# V1.0 一步创建 Agent：模型、工具、系统提示、输出格式一次传入
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "你是天气助手。"
        "当用户询问多个城市天气时，"
        "你需要分别调用工具获取数据，并进行比较分析。"
    ),
    response_format=WeatherCompareOutput,
)

# 调用 Agent，返回结果中包含 messages 与 structured_response（若指定了 response_format）
result = agent.invoke(
    {"input": "请问今天北京和上海的天气怎么样，哪个城市更热？"}
)
print(result)
print()
print(json.dumps(result["structured_response"], ensure_ascii=False, indent=2))
