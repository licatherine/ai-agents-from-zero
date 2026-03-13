"""
【案例】用 FastMCP 实现天气查询 MCP 服务（SSE + 指定 host/port）

对应教程章节：第 20 章 - MCP 模型上下文协议 → 6、案例实战：本地 MCP 天气服务与客户端

与 McpServer.py 的对比：本文件使用 FastMCP，host/port 不在构造函数里传，而是在 run() 时传入。
正确写法：mcp = FastMCP("服务名")  →  mcp.run(transport="sse", host="127.0.0.1", port=8000)
错误写法：mcp = FastMCP("服务名", host="127.0.0.1", port=8000)  # FastMCP 构造函数不支持 host/port
"""

from typing import Any


import json
import os

# pip install mcp httpx python-dotenv
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import httpx

load_dotenv()

# 构造函数只接受「服务名」（任意字符串，用于标识本服务，客户端/日志里会看到），不能传 host/port
mcp = FastMCP(
    "WeatherServerSSE"
)  # "WeatherServerSSE" 就是你自己起的名，可改成 "MyWeather" 等


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的即时天气信息。city 为城市英文名，如 Beijing、Shanghai。"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn",
    }
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    # host、port 在 run() 时传入，不是构造函数
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
