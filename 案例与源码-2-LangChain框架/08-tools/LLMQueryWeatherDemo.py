"""
【案例】天气助手完整链路：bind_tools → 解析 tool_calls → 执行工具 → 结果回填 → 模型生成自然语言回复

对应教程章节：第 17 章 - Tools 工具调用 → 5、天气助手实战 → 5.4 大模型调用天气工具并生成回复

知识点速览：
- bind_tools([get_weather])：把工具列表绑定到模型，请求时会把工具的名称、描述、参数 schema 一并发给模型，模型可返回 tool_calls。
- JsonOutputKeyToolsParser：从模型输出中解析出「调用了哪个工具、参数是什么」，得到可传给 tool.invoke 的参数字典。
- 链式编排：模型→解析器→工具 得到天气 JSON，再通过 prompt|llm|parser 把 JSON 转成自然语言描述（LCEL 见第 15 章）。
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

import os
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from QueryWeatherTool import get_weather

# 初始化大模型（教程 5.4：需可调用工具的大模型）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 将工具绑定到模型：请求时会把 get_weather 的名称、描述、参数 schema 发给模型，模型可返回 function_call
llm_with_tools = llm.bind_tools([get_weather])

# 解析器：从模型输出中提取「第一个工具调用」的 name 与 arguments，得到可传给 tool.invoke 的字典
parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)

# 天气查询链：用户问题 → 模型（可能返回 tool_call）→ 解析出参数 → 执行 get_weather → 得到天气 JSON 字符串
get_weather_chain = llm_with_tools | parser | get_weather

# 输出链：把天气 JSON 塞进提示词，由模型转成自然语言（LCEL：prompt | llm | StrOutputParser）
output_prompt = PromptTemplate.from_template(
    """你将收到一段 JSON 格式的天气数据{weather_json}，请用简洁自然的方式将其转述给用户。
    以下是天气 JSON 数据：
    请将其转换为中文天气描述，例如：
    "北京现在天气：多云，气温 28℃，体感有点闷热（约 32℃），湿度 75%，微风（东南风 2 米/秒），
    能见度很好，大约 10 公里。建议穿短袖短裤。适合做户外运动。"
    """
)
output_parser = StrOutputParser()
output_chain = output_prompt | llm | output_parser

# 完整链：先走天气查询链得到 JSON，再包装成 {"weather_json": x} 送入输出链，得到最终中文描述
full_chain = get_weather_chain | (lambda x: {"weather_json": x}) | output_chain

result = full_chain.invoke("请问北京今天的天气如何？")
logger.info(result)

"""
【输出示例】
{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 800, "main": "Clear", "description": "\u6674", "icon": "01d"}], "base": "stations", "main": {"temp": 10.76, "feels_like": 8.26, "temp_min": 10.76, "temp_max": 10.76, "pressure": 1033, "humidity": 14, "sea_level": 1033, "grnd_level": 1027}, "visibility": 10000, "wind": {"speed": 1.6, "deg": 232, "gust": 2.63}, "clouds": {"all": 0}, "dt": 1773034935, "sys": {"country": "CN", "sunrise": 1773009388, "sunset": 1773051236}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}
2026-03-09 13:45:53.809 | INFO     | __main__:<module>:57 - 北京现在天气：晴，气温 10.8℃，体感偏凉（约 8.3℃），湿度仅 14%，非常干燥，微风（西南风，2.3 米/秒），云量 0%，能见度极佳，达 10 公里。
日出时间：约 05:56，日落时间：约 18:13（北京时间）。
建议添件薄外套，注意保湿润肤，适合户外活动。
"""

