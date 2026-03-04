"""
【案例】函数链：用 RunnableLambda 将普通 Python 函数接入 LCEL 链

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.5 RunnableLambda（函数链）

知识点速览：
- RunnableLambda 将普通 Python 函数包装成 Runnable，从而可放在 LCEL 链中与 prompt、model、parser 等用 | 连接。
- 作用：自定义逻辑（如打印中间结果、数据格式转换）作为链的一个节点；可用 RunnableLambda(函数) 或直接把函数写在 | 之间（LangChain 会自动包装）。
- 函数的输入为上一步输出，返回值作为下一步输入，便于在链中插入调试或适配层。
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def debug_print(x):
    """将上一步输出打印并转为 chain2 需要的 dict 格式 {"input": 文本}。"""
    logger.info(f"中间结果:{x}")
    return {"input": x}


# 子链 1：中文介绍某主题，输出 str
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

# 子链 2：将 input 翻译成英文
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
    ("human", "{input}")
])
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

# 方式一：直接把函数放在 | 之间，LCEL 会自动包装成 Runnable
full_chain = chain1 | debug_print | chain2
result1 = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果111:{result1}")

# 方式二：显式使用 RunnableLambda(函数)，效果相同
debug_node = RunnableLambda(debug_print)
full_chain = chain1 | debug_node | chain2
result2 = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果222:{result2}")
