"""
【案例】模型调用：同步 batch（批量调用）

对应教程章节：第 13 章 - 提示词与消息模板 → 2、模型调用方法

知识点速览：
- batch：一次提交多条输入，模型内部并行处理，返回与输入一一对应的多条结果。
- 适用场景：数据清洗、批量评估、离线任务；通常比「循环里多次 invoke」更高效。
- 输入：可以是字符串列表，或消息列表的列表；这里用「字符串列表」最简单，每条会当作一次用户问题。
- zip：Python 内置函数，名字来自「拉链」（zipper），表示把多个序列按位置一一配对，与压缩文件 .zip 无关。
  例：zip([1,2,3], ["a","b","c"]) → (1,"a"), (2,"b"), (3,"c")。这里用来把「问题列表」和「回答列表」成对遍历。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# 本示例用「字符串列表」作为 batch 输入，无需 Message 类型；若改为多角色可引入 HumanMessage、SystemMessage

load_dotenv()

# ---------- 1. 实例化模型 ----------
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ---------- 2. 准备多条独立问题（批量输入的列表）----------
# 每一条字符串会作为「一次请求」发给模型；batch 会并行处理这些请求，最后返回与之一一对应的响应列表。
questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内"
]

# ---------- 3. 批量调用：model.batch(questions) ----------
# 返回的是一个列表，每个元素对应一条问题的 AIMessage，用 .content 取该条回复的文本。
response = model.batch(questions)
print(f"响应类型：{type(response)}")
print()

# zip(questions, response)：zip 取自「拉链」，表示按位置配对，不是压缩。把两个列表一一对应：(第1个问题, 第1个回答), ...
# 循环时每次同时拿到「当前问题 q」和「对应的回答 r」。可去掉 zip，改用下标 for i in range(len(questions)) 配合 questions[i]、response[i]。
for q, r in zip(questions, response):
    print(f"问题：{q}\n回答：{r.content}\n")
