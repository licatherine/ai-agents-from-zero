"""
【案例】本地 MCP 天气客户端（直接调用服务端已注册工具）

对应教程章节：第 20 章 - MCP 模型上下文协议 → 6、案例实战：本地 MCP 天气服务与客户端

知识点速览：
- MCP 客户端与服务器 1:1 对应：客户端负责向服务端发起请求、调用其暴露的工具。
- 本案例为「同机、同进程演示」：客户端通过 Python 的 `from McpServer import mcp` 直接拿到服务端的
  mcp 实例，再用 mcp._tools 调用已注册工具，并没有走网络（没有连 127.0.0.1:8000）。这样做的目的
  是便于理解「工具暴露 → 发现 → 调用」的流程；实际生产里客户端会通过 SSE/STDIO 等协议连接独立进程。
- 运行方式：直接运行本文件即可（会自动导入 McpServer 并创建 mcp，无需先单独启动 McpServer.py 进程）。
"""

import json
from loguru import logger

# 「连接」方式：通过导入获取服务端的 mcp 对象，同进程内直接访问 mcp._tools，并非网络连接
from McpServer import mcp


class MCPWeatherClient:
    """MCP 天气服务客户端，用于访问 MCPWeatherServer 服务端暴露的工具"""

    def __init__(self, mcp_instance):
        self.mcp_instance = mcp_instance
        # 获取服务端已注册的所有工具（字典：工具名 -> 可调用函数）
        self.available_tools = mcp_instance._tools

    def check_tool_availability(self, tool_name: str) -> bool:
        """检查指定工具是否在服务端已注册，避免调用不存在的工具"""
        is_available = tool_name in self.available_tools
        if is_available:
            logger.info(f"工具 '{tool_name}' 可用")
        else:
            logger.warning(f"工具 '{tool_name}' 未在服务端注册")
        return is_available

    def call_get_weather(self, city: str) -> str or None:
        """调用服务端的 get_weather 工具，查询指定城市天气"""
        tool_name = "get_weather"
        if not self.check_tool_availability(tool_name):
            return None

        try:
            # 直接调用服务端已注册的工具函数（本案例同进程；实际可通过 SSE/STDIO 协议调用）
            weather_result = self.available_tools[tool_name](city)
            logger.info(
                f"成功获取 {city} 天气数据，返回结果长度：{len(weather_result)}"
            )
            return weather_result
        except Exception as exc:
            logger.error(f"调用 {tool_name} 工具失败：{str(exc)}")
            return None


def run_client_demo():
    """客户端演示：初始化客户端，依次查询多城市天气并格式化输出"""
    logger.info("初始化 MCP 天气客户端...")
    client = MCPWeatherClient(mcp)

    # 调用天气查询工具（支持 Beijing、Shanghai、Guangzhou 等英文城市名）
    target_cities = ["Beijing", "Shanghai"]
    for city in target_cities:
        logger.info(f"\n========== 查询 {city} 天气 ==========")
        weather_data = client.call_get_weather(city)
        if weather_data:
            # 格式化输出结果（可选，方便阅读）
            formatted_data = json.dumps(
                json.loads(weather_data), indent=4, ensure_ascii=False
            )
            print(f"格式化天气结果：\n{formatted_data}")
        print("-" * 50)


if __name__ == "__main__":
    logger.info("启动 MCP 天气客户端...")
    run_client_demo()

# 【输出示例】
# 2026-03-13 11:09:46.119 | INFO     | __main__:<module>:75 - 启动 MCP 天气客户端...
# 2026-03-13 11:09:46.119 | INFO     | __main__:run_client_demo:57 - 初始化 MCP 天气客户端...
# 2026-03-13 11:09:46.119 | INFO     | __main__:run_client_demo:63 -
# ========== 查询 Beijing 天气 ==========
# 2026-03-13 11:09:46.119 | INFO     | __main__:check_tool_availability:32 - 工具 'get_weather' 可用
# 2026-03-13 11:09:46.830 | INFO     | McpServer:get_weather:84 - 查询 Beijing 天气结果：{'coord': {'lon': 116.3972, 'lat': 39.9075}, 'weather': [{'id': 804, 'main': 'Clouds', 'description': '阴，多云', 'icon': '04d'}], 'base': 'stations', 'main': {'temp': 4.94, 'feels_like': 4.03, 'temp_min': 4.94, 'temp_max': 4.94, 'pressure': 1027, 'humidity': 41, 'sea_level': 1027, 'grnd_level': 1021}, 'visibility': 10000, 'wind': {'speed': 1.39, 'deg': 6, 'gust': 1.02}, 'clouds': {'all': 100}, 'dt': 1773371280, 'sys': {'type': 1, 'id': 9609, 'country': 'CN', 'sunrise': 1773354606, 'sunset': 1773397087}, 'timezone': 28800, 'id': 1816670, 'name': 'Beijing', 'cod': 200}
# 2026-03-13 11:09:46.830 | INFO     | __main__:call_get_weather:46 - 成功获取 Beijing 天气数据，返回结果长度：570
# 格式化天气结果：
# {
#     "coord": {
#         "lon": 116.3972,
#         "lat": 39.9075
#     },
#     "weather": [
#         {
#             "id": 804,
#             "main": "Clouds",
#             "description": "阴，多云",
#             "icon": "04d"
#         }
#     ],
#     "base": "stations",
#     "main": {
#         "temp": 4.94,
#         "feels_like": 4.03,
#         "temp_min": 4.94,
#         "temp_max": 4.94,
#         "pressure": 1027,
#         "humidity": 41,
#         "sea_level": 1027,
#         "grnd_level": 1021
#     },
#     "visibility": 10000,
#     "wind": {
#         "speed": 1.39,
#         "deg": 6,
#         "gust": 1.02
#     },
#     "clouds": {
#         "all": 100
#     },
#     "dt": 1773371280,
#     "sys": {
#         "type": 1,
#         "id": 9609,
#         "country": "CN",
#         "sunrise": 1773354606,
#         "sunset": 1773397087
#     },
#     "timezone": 28800,
#     "id": 1816670,
#     "name": "Beijing",
#     "cod": 200
# }
