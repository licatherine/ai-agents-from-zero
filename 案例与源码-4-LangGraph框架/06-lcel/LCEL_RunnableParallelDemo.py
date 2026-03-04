"""
【案例】并行链：同时运行多条子链，汇总结果

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.4 RunnableParallel（并行链）

知识点速览：
- 并行链同时运行多条子链，待全部完成后汇总结果；使用 RunnableParallel({ "键名": runnable, ... }) 定义。
- 适用场景：同一问题用中/英文各答一遍并聚合、多模型并行、多路径推理等。
- invoke 的返回值为 dict，键为 RunnableParallel 的键名，值为对应子链的输出；可用 get_graph().print_ascii() 查看图结构（为 LangGraph 铺垫）。
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from loguru import logger

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 子链 1：中文简短介绍
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

# 子链 2：英文简短介绍（与 chain1 同结构，仅提示词语言不同）
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用英文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

# RunnableParallel：同一输入会同时喂给多个子链，结果汇总为 dict
parallel_chain = RunnableParallel({
    "chinese": chain1,
    "english": chain2
})

# 一次 invoke，返回 {"chinese": "...", "english": "..."}
result = parallel_chain.invoke({"topic": "langchain"})
logger.info(result)

# 可选：打印并行链的 ASCII 图结构，便于理解数据流与后续学习 LangGraph
parallel_chain.get_graph().print_ascii()
