"""
【案例】JsonOutputParser + get_format_instructions()：用格式说明引导模型输出

对应教程章节：第 14 章 - 输出解析器 → 2、常用输出解析器用法

知识点速览：
一、get_format_instructions() 做什么？
  - 返回一段**格式说明字符串**，描述「希望模型输出成什么样子」（例如 JSON 里有哪些键、类型是什么）。
  - 把这段说明拼进 Prompt（如放在 {format_instructions} 占位符），模型更容易输出可被解析的 JSON，减少格式错误。

二、本案例做法：用 Pydantic 模型 Person 定义「时间、人物、事件」结构 → 用 JsonOutputParser(pydantic_object=Person) 创建解析器 → 用 parser.get_format_instructions() 得到说明 → 把说明拼进 human 消息，再调用模型与解析器。
  - 这样模型会按 Person 的字段来生成 JSON，解析后得到符合 Person 结构的数据（或解析失败时抛错）。
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv(encoding="utf-8")


class Person(BaseModel):
    """定义一条「新闻」的结构：时间、人物、事件。用于约束模型输出的 JSON 形状。"""
    time: str = Field(description="时间")
    person: str = Field(description="人物")
    event: str = Field(description="事件")


# 创建 JSON 解析器，并绑定 Pydantic 模型：解析结果会按 Person 校验与转换
parser = JsonOutputParser(pydantic_object=Person)

# 获取「格式说明」：描述 Person 各字段，便于拼进提示词让模型按此输出
format_instructions = parser.get_format_instructions()

# 在 human 消息里加入 {format_instructions}，模型会看到「请按如下格式输出 JSON …」
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化JSON数据。"),
    ("human", "请生成一个关于{topic}的新闻。{format_instructions}")
])

# 填 topic 和 format_instructions，得到消息列表
prompt = chat_prompt.format_messages(topic="小米su7跑车", format_instructions=format_instructions)
logger.info(prompt)

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

result = model.invoke(prompt)
logger.info(f"模型原始输出:\n{result}")

# 用同一解析器解析，得到符合 Person 结构的数据（dict，或可转成 Person 实例）
response = parser.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")
logger.info(f"结果类型: {type(response)}")
