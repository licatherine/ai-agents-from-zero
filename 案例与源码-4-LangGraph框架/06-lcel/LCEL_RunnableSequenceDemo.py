"""
【案例】顺序链：Prompt → Model → Parser 一条线执行

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.1 RunnableSequence（顺序链）

知识点速览：
- Chain 的典型结构由三部分组成：提示词模板（Prompt）、大模型（LLM/Chat Model）、结果结构化解析器（Output Parser，可选）。
- LCEL 用管道符 | 将多个 Runnable 串联，等价于 RunnableSequence；数据从左到右依次流过，前一步输出作为后一步输入。
- prompt、model、parser 都实现了 Runnable 接口，统一用 .invoke() 调用，也可用 | 组合成链后一次 invoke。
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

# 创建聊天提示模板（Runnable 子类）：包含系统角色与用户问题占位符
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题"),
    ("human", "请回答:{question}")
])

# 使用 invoke 渲染提示词，返回 PromptValue，可直接交给模型（统一接口）
prompt = chat_prompt.invoke({"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"})
logger.info(prompt)

# 初始化聊天模型（同样实现 Runnable，支持 invoke/stream/batch）
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 模型接收上一步的 PromptValue，返回 AIMessage
result = model.invoke(prompt)
logger.info(f"********>模型原始输出:\n{result}")

# 字符串输出解析器（Runnable）：从 AIMessage 中取出文本，返回 str
parser = StrOutputParser()

# 解析器接收 AIMessage，返回结构化结果（此处为字符串）
response = parser.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")
logger.info(f"结果类型: {type(response)}")

print()
print("*" * 60)
print()

# 用管道符 | 构建顺序链，等价于 RunnableSequence([chat_prompt, model, parser])
chain = chat_prompt | model | parser

# 链整体也是 Runnable：一次 invoke 完成「渲染 → 模型 → 解析」，入参为提示词变量
result_chain = chain.invoke({"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"})
logger.info(f"Chain执行结果:\n{result_chain}")
logger.info(f"Chain执行结果类型: {type(result_chain)}")

print()
print(type(chain))
