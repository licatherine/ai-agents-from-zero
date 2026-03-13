"""
【案例】本地 MCP 天气服务端（极简实现，无 FastMCP 依赖）

对应教程章节：第 20 章 - MCP 模型上下文协议 → 6、案例实战：本地 MCP 天气服务与客户端

知识点速览：
- MCP 采用客户端-服务器架构：服务端负责「暴露工具」，客户端负责「调用工具」。
- 本案例用自定义 MCP 服务类模拟「工具注册与暴露」：通过 @mcp.tool() 将 get_weather 注册为 MCP 工具，
  供客户端按协议调用；传输方式示例为 SSE（实际为简化版，仅保持进程运行）。
- 与第 17 章 Tool 的区别：Tool 是「能力封装」，在单进程内直接调用；MCP 是「跨进程/跨应用」的
  标准化协议，同一套工具定义可被多种 AI 应用复用。
"""

import json
import os
import httpx
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------- 极简版 MCP 服务类（无 FastMCP 依赖，纯手写）----------------------
# 若使用 FastMCP，则不需要下面这一整段 class，直接 mcp = FastMCP("名") + @mcp.tool() + mcp.run() 即可，见 McpServerWeatherByFastMCP.py
class MCPWeatherServer:
    """极简版 MCP 服务类，替代原 FastMCP，无 fastmcp 残留"""

    def __init__(self, name: str, host: str, port: int):
        # 保留原实例化参数，与原代码配置对齐
        self.name = name
        self.host = host
        self.port = port
        # 存储已注册的工具函数；客户端通过协议可发现并调用这些工具
        self._tools = {}

    def tool(self):
        """实现 @mcp.tool() 装饰器：把普通函数注册为 MCP 工具，供客户端调用"""

        def decorator(func):
            self._tools[func.__name__] = func  # 注册工具函数，key 为函数名
            return func

        return decorator

    def run(self, transport: str):
        """实现 mcp.run(transport=\"sse\") 调用格式；本实现仅保持进程运行，便于客户端连接"""
        if transport != "sse":
            logger.warning(f"不支持的传输协议 {transport}，默认使用 SSE")
        logger.info(f"启动 MCP SSE 天气服务器，监听 http://{self.host}:{self.port}/sse")
        self._keep_alive()

    def _keep_alive(self):
        """简单保持进程运行，替代原 FastMCP 的监听逻辑"""
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("MCP 天气服务器已停止")


# ---------------------- 创建 MCP 实例并注册工具 ----------------------
# 对应教程：MCP 架构中的「MCP 服务器」角色，为客户端提供工具与上下文
# 若改用 FastMCP：构造函数只接受服务名，不能写 FastMCP(..., host=..., port=...)；
# host/port 在 run() 时传，如 mcp.run(transport="sse", host="127.0.0.1", port=8000)。参见 McpServerWeatherByFastMCP.py
mcp = MCPWeatherServer("WeatherServerSSE", host="127.0.0.1", port=8000)


@mcp.tool()  # 将 get_weather 注册为 MCP 工具，客户端可通过协议发现并调用
def get_weather(city: str) -> str:
    """
    查询指定城市的即时天气信息。
    参数 city: 城市英文名，如 Beijing
    返回: OpenWeather API 的 JSON 字符串
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv(
            "OPENWEATHER_API_KEY"
        ),  # 从环境变量读取 API Key，避免写死密钥
        "units": "metric",  # 使用摄氏度
        "lang": "zh_cn",  # 输出语言为简体中文
    }
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()
    logger.info(f"查询 {city} 天气结果：{data}")
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse")
    mcp.run(transport="sse")

# 【输出示例】
# 2026-03-13 10:53:49.293 | INFO     | __main__:<module>:86 - 启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse
# 2026-03-13 10:53:49.293 | INFO     | __main__:run:46 - 启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse
