"""
【案例】用「字典」定义 ChatPromptTemplate 的消息

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、为什么用字典？
  - 除了 (role, content) 元组，消息列表里也可以用 dict，格式：{"role": "角色名", "content": "内容"}。
  - role 常见取值："system"（系统设定）、"user" 或 "human"（用户）、"ai"（助手）。
  - content 里同样可以写占位符 {name}、{question}，在 format_messages 时传入。

二、和元组、Message 的对比：
  - 元组：("system", "你是{name}")  ← 见 ChatPromptTemplate_TupleParam.py
  - 字典：{"role": "system", "content": "你是{name}"}  ← 本文件
  - Message：SystemMessage(content="你是{name}")  ← 见 ChatPromptTemplate_MessageParam.py
  三种写法等价，字典写法对「从 JSON 等配置里读入」比较友好。创建时用 from_messages 或构造函数均可，本文件同时演示两种方式。
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------- 方式一：from_messages ----------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
        {"role": "user", "content": "请问：{question}"},
    ]
)
message = chat_prompt.format_messages(name="小问", question="什么是LangChain")
print("from_messages:", message)

# ---------- 方式二：构造函数（传入同样字典列表，效果一致）----------
chat_prompt2 = ChatPromptTemplate(
    [
        {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
        {"role": "user", "content": "请问：{question}"},
    ]
)
message2 = chat_prompt2.format_messages(name="小问", question="什么是LangChain")
print("构造函数:", message2)

"""
【输出示例】
[SystemMessage(content='你是AI助手，你的名字叫小问。', additional_kwargs={}, response_metadata={}), HumanMessage(content='请问：什么是LangChain', additional_kwargs={}, response_metadata={})]
"""

