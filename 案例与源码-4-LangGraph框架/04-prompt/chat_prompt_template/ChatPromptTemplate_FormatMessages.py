"""
【案例】from_messages 创建模板 + format_messages / invoke / format 的用法

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、from_messages 是什么？
  - ChatPromptTemplate.from_messages([...]) 是「用消息列表创建对话模板」的常用写法。
  - 列表里每条消息可以是：(role, content) 元组、dict、或 Message 类实例。
  - 例如：("system", "你是{role}")、("human", "{question}")，其中的 {role}、{question} 就是占位符。

二、填充模板的三种方式（初学者重点区分）：
  - format_messages(**kwargs)：传入占位符变量，返回「消息列表」List[BaseMessage]，
    可直接给 model.invoke(prompt_value) 使用，或用来调试、查看最终消息结构。
  - invoke({"变量名": 值})：传入字典，同样返回消息列表（PromptValue），
    常用 .to_string() 转成整段文本查看，或直接把返回值交给 model.invoke。
  - format(**kwargs)：传入占位符变量，返回的是「拼接后的纯文本字符串」，
    不是消息列表，适合只看一整段文字时用，不能直接替代 format_messages 交给聊天模型。

三、小结：和模型对话时，用 format_messages 或 invoke 得到消息列表，再 model.invoke(消息列表)。

四、Python 入门：两种传参写法与 ** 解包
  - 写法一（关键字参数）：直接写 参数名=值，例如 format_messages(role="xxx", question="yyy")。
  - 写法二（字典 + **）：先有一个字典 d = {"role": "xxx", "question": "yyy"}，再写 format_messages(**d)。
  - ** 叫做「解包」：**字典 会把字典「拆开」成「键=值」的形式传给函数。也就是说：
      format_messages(**{"role": "A", "question": "B"})  等价于  format_messages(role="A", question="B")
  - 函数定义里的 **kwargs 表示「接收任意多个关键字参数，并收成一个字典」；调用时的 **字典 则相反，是「把字典展开成关键字参数」。
  - 什么时候用 **？当参数已经在一个字典里时（例如从配置、接口返回、循环中得到），用 ** 就不用手写一长串 role=..., question=...，直接 **params 即可。
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 用 from_messages 创建模板：一条 system（带 {role}）、一条 human（带 {question}）
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，请回答我提出的问题"),
        ("human", "请回答:{question}")
    ]
)

# ---------- 方式一：format_messages ----------
# 下面两种写法完全等价，都是把 role、question 传给模板里的 {role}、{question}：
#   写法 A（关键字参数）：format_messages(role="python开发工程师", question="堆排序怎么写")
#   写法 B（字典 + ** 解包）：format_messages(**{"role": "python开发工程师", "question": "堆排序怎么写"})
# ** 表示把字典「解包」成 key=value 的形式传入，适合参数已经在 dict 里的场景。
prompt_value = chat_prompt.format_messages(**{"role": "python开发工程师", "question": "堆排序怎么写"})
print(prompt_value)

print()
# ---------- 方式二：invoke（传字典）----------
# 传入一个字典，键为占位符变量名，值为要填充的内容；返回的也是「消息列表」（PromptValue）
# .to_string() 可把整段对话转成纯文本，方便打印查看
prompt_value2 = chat_prompt.invoke({"role": "python开发工程师", "question": "堆排序怎么写"})
print(prompt_value2.to_string())

print()

# ---------- 方式三：format（注意：返回的是字符串，不是消息列表）----------
# 适合只想得到「一整段文本」时用；若要把对话发给聊天模型，请用 format_messages 或 invoke
prompt_value3 = chat_prompt.format(**{"role": "python开发工程师", "question": "快速排序怎么写"})
print(prompt_value3)


# llm = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
# print()
# print("======================")
#
# result = llm.invoke(prompt_value)
# print(result)
# print(result.content)
