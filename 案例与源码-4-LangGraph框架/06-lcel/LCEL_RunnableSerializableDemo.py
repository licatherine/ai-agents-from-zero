"""
【案例】串行链：多步串联，前一步输出作为后一步输入

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.3 RunnableSerializable / 串行链（多步串联）

知识点速览：
- 当需要多次调用大模型、把多个步骤串联时，可用 | 或 lambda 将多条子链串成一条「串行链」。
- 前一步的输出会作为后一步的输入；若后一步需要的入参键名与上一步输出不一致，可用 lambda 做映射（如把 content 转为 {"input": content}）。
- 链本身是 Runnable，串起来后仍可用一次 invoke 完成多步调用。
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 子链 1：用中文介绍某主题，输出为 str
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

result1 = chain1.invoke({"topic": "langchain"})
logger.info(result1)

# 子链 2：将用户输入翻译成英文，期望入参为 {"input": 文本}
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
    ("human", "{input}")
])
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

# 串行组合：chain1 输出 str，用 lambda 转为 {"input": content} 以匹配 chain2 的占位符
full_chain = chain1 | (lambda content: {"input": content}) | chain2

# 一次 invoke：先执行 chain1，再把结果作为 chain2 的 input
result = full_chain.invoke({"topic": "langchain"})
logger.info(result)
