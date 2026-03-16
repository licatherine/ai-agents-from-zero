"""
【案例】显式使用 MessagesPlaceholder：在模板里预留一段消息的位置

对应教程章节：第 13 章 - 提示词与消息模板 → 5、对话提示词模板（ChatPromptTemplate）

知识点速览：
一、为什么需要「消息占位符」？
  - 有时消息条数、内容要等到「调用时」才知道，例如：把「历史对话」插进当前提示词里。
  - 若不用占位符，就得在代码里手动拼好几条 SystemMessage、HumanMessage、AIMessage，很麻烦。
  - MessagesPlaceholder 的作用：在模板里占一个「坑」，invoke 时传入一个「消息列表」，会整块插入到这个位置。

二、显式使用是什么意思？
  - 显式：在 from_messages 里明确写上 MessagesPlaceholder("变量名")，变量名可自取（如 "memory"、"history"、"chat_history" 等）。
  - 调用时传入的字典键要与占位符变量名一致，例如占位符为 "memory" 则传 {"memory": [...]}，若改为 "history" 则传 {"history": [...]}。

三、典型用法：多轮对话
  - 模板结构：系统设定 + [历史消息占位] + 当前用户问题。
  - invoke 时：memory=上一轮/多轮的对话列表，question=这一轮用户问的那一句。
  - 模型就能看到「之前的对话 + 当前问题」，实现带上下文的回复。
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 模板：系统消息 + 一个「消息占位符」memory + 当前用户问题 {question}
# memory 位置会在 invoke 时被替换成你传入的消息列表（如历史对话）
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题",
        ),  # role 须为标准名：system / human / ai（不可自创如 system1）；也可写 SystemMessage(content="...")
        MessagesPlaceholder(
            "memory"
        ),  # 变量名可自取，如 "history"、"chat_history" 等，invoke 时键与之一致即可
        ("human", "{question}"),
    ]
)

# invoke 时传入：memory = 历史消息列表，question = 当前问题
# 这里用两条消息模拟「上一轮」的对话，再问「我的名字叫什么」来测试模型是否利用上下文
prompt_value = prompt.invoke(
    {
        "memory": [
            HumanMessage("我的名字叫亮仔，是一名程序员111"),
            AIMessage("好的，亮仔你好222"),
        ],
        "question": "请问我的名字叫什么？",
    }
)

# 把整段 prompt 转成字符串查看（系统设定 + 历史 + 当前问题）
print(prompt_value.to_string())

"""
【输出示例】
System: 你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题
Human: 我的名字叫亮仔，是一名程序员111
AI: 好的，亮仔你好222
Human: 请问我的名字叫什么？
"""

