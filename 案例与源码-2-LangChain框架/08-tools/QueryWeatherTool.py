"""
【案例】天气查询工具：用 @tool 定义可被 LLM 调用的天气接口，请求 OpenWeather API 并返回 JSON

对应教程章节：第 17 章 - Tools 工具调用 → 5、天气助手实战 → 5.3 定义天气工具

知识点速览：
- 与 4.1/4.2 一致：用 @tool 装饰器把普通函数变成 LangChain Tool，函数的 docstring 会作为工具的 description 提供给模型，模型据此决定是否调用及如何传参。
- 工具内部实现「真正执行」：本案例中由代码发 HTTP 请求、解析 JSON，模型只负责输出「要调 get_weather、传 loc=xxx」。
- API Key 安全：使用 os.getenv("OPENWEATHER_API_KEY") 从环境变量读取密钥，避免写死在代码中（教程 5.2 要求写入 .env）。
- 返回格式：用 json.dumps(data) 将 API 返回的字典序列化为字符串，便于后续链（如 LLMQueryWeatherDemo 中的输出链）或模型直接消费。
"""

from langchain_core.tools import tool
import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")


# @tool 装饰器：函数名 get_weather 即工具名，下方 docstring 作为工具的 description 供模型理解「何时调、传什么」
@tool
def get_weather(loc):
    """
    查询即时天气函数

    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称。
                注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，
                则 loc 参数需要输入 'Beijing'/'shanghai'。
    :return: OpenWeather API 查询即时天气的结果。具体 URL 请求地址为：
             https://home.openweathermap.org/users/sign_in。
             返回结果对象类型为解析之后的 JSON 格式对象，并用字符串形式进行表示，
             其中包含了全部重要的天气信息。
    """
    # Step 1. 构建请求 URL（OpenWeather 当前天气接口，见教程 5.2 API 文档）
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2. 设置查询参数：q=城市名，appid 从环境变量读取（安全实践），units=metric 为摄氏度，lang=zh_cn 为中文描述
    params = {
        "q": loc,
        "appid": os.getenv(
            "OPENWEATHER_API_KEY"
        ),  # 从 .env 读取，勿将 Key 写死在代码中
        "units": "metric",  # 温度单位：metric=摄氏度
        "lang": "zh_cn",  # 天气描述语言：简体中文
    }

    # Step 3. 发送 GET 请求；httpx 与 requests 用法类似，timeout 避免长时间阻塞
    response = httpx.get(url, params=params, timeout=30)

    # Step 4. 解析响应为 Python 字典后，再序列化为 JSON 字符串返回，供模型或后续链使用（见 5.4 LLMQueryWeatherDemo）
    data = response.json()
    return json.dumps(data)


# 本地测试：invoke 可传单参数值 get_weather.invoke("beijing") 或字典 get_weather.invoke({"loc": "beijing"})
# result = get_weather.invoke("shanghai")
result = get_weather.invoke("beijing")
print(result)

"""
【输出示例】
{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 800, "main": "Clear", "description": "\u6674", "icon": "01d"}], "base": "stations", "main": {"temp": 10.76, "feels_like": 8.26, "temp_min": 10.76, "temp_max": 10.76, "pressure": 1033, "humidity": 14, "sea_level": 1033, "grnd_level": 1027}, "visibility": 10000, "wind": {"speed": 1.6, "deg": 232, "gust": 2.63}, "clouds": {"all": 0}, "dt": 1773034935, "sys": {"country": "CN", "sunrise": 1773009388, "sunset": 1773051236}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}
"""

