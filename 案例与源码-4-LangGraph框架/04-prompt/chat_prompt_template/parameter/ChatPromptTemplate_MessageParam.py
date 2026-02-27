"""
【案例】用「Message 类」定义 ChatPromptTemplate 的消息

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、Message 类是什么？
  - LangChain 里用不同「消息类型」区分角色：SystemMessage、HumanMessage、AIMessage。
  - SystemMessage：系统设定（如「你是 AI 助手」）；HumanMessage：用户说的话；AIMessage：模型回复。
  - 多轮对话、或需要明确类型时，用 Message 类比写 ("system", "xxx") 更直观，也便于后续扩展。

二、为什么用 Message 而不是元组/字典？
  - 语义更清晰：一眼看出是「系统消息」还是「用户消息」。
  - 某些高级用法（如带 tool_calls、additional_kwargs）时，必须用 Message 实例。
  - 初学者掌握：用元组够用；想写得更「面向对象」时用 Message。

三、和元组、字典的对比：
  - 元组：("system", "你是{name}")  ← 见 ChatPromptTemplate_TupleParam.py
  - 字典：{"role": "system", "content": "你是{name}"}  ← 见 ChatPromptTemplate_DictParam.py
  - Message：SystemMessage(content="你是{name}")  ← 本文件
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# 用 Message 类列表定义模板：系统消息 + 用户消息，内容里仍可带占位符 {name}、{question}
chat_prompt = ChatPromptTemplate(
    [
        SystemMessage(content="你是AI助手，你的名字叫{name}。"),
        HumanMessage(content="请问：{question}")
    ]
)

# 传入占位符变量，得到消息列表
message = chat_prompt.format_messages(name="亮仔", question="什么是LangChain")
print(message)

# 【输出示例】
# [SystemMessage(content='你是AI助手，你的名字叫{name}。', additional_kwargs={}, response_metadata={}), HumanMessage(content='请问：{question}', additional_kwargs={}, response_metadata={})]