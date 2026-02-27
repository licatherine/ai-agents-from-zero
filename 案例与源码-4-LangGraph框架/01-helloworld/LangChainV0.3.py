"""
【案例】LangChain 0.x 写法：ChatOpenAI + 三种配置方式（硬编码 / 环境变量 / .env）

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld → 4、实战：基于阿里百炼的 HelloWorld

知识点速览：
- 0.x 写法从各厂商包直接导入具体类（如 ChatOpenAI），通过 base_url 接国内兼容接口。
- 配置方式演进：硬编码（不推荐）→ 环境变量 → .env + load_dotenv（推荐），避免 API Key 进版本库。
- invoke 同步调用、response.content 取回复正文。了解即可，当前主推 1.0 的 init_chat_model 写法。
"""

from langchain_openai import ChatOpenAI  # OpenAI 兼容的聊天模型封装，可配合 base_url 接国内平台
import os
from dotenv import load_dotenv  # 从 .env 文件加载环境变量，避免把 API Key 写进代码

# ========== 1. 大模型客户端初始化（三种配置方式，推荐第 3 版） ==========

# 第 1 版：硬编码写死（仅演示，不推荐）
# 缺点：API Key 会进版本库，有泄露风险；换环境要改代码。
# llm = ChatOpenAI(
#     model="qwen-plus",
#     api_key="你自己的api-key",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# 第 2 版：用系统环境变量（需先 export 或在运行前 set）
# 缺点：若未先执行 load_dotenv()，.env 里的变量不会生效，需依赖外部已注入的环境。
# llm = ChatOpenAI(
#     model="qwen-plus",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# 第 3 版（推荐）：用 python-dotenv 从 .env 加载，再通过 os.getenv 读取
# 项目根目录放 .env 文件，内容如：QWEN_API_KEY=sk-xxx（不要提交到 Git）
load_dotenv(encoding='utf-8')  # encoding 指定 utf-8，避免 .env 中中文注释乱码

llm = ChatOpenAI(
    model="deepseek-v3.2",           # 模型名，需与平台「模型广场」中的名称一致
    api_key=os.getenv("QWEN_API_KEY"),  # 从环境变量取 Key（已由 load_dotenv 从 .env 加载）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里百炼 OpenAI 兼容接口地址
)

# ========== 2. 调用大模型并打印结果 ==========
# invoke：同步调用，传入用户问题字符串，返回 AIMessage 等消息对象
response = llm.invoke("你是谁")

# response 为 LangChain 消息对象，包含 content、additional_kwargs 等元数据
print(response)        # 打印完整对象（含元数据，便于调试）
print()
print(response.content)  # 只取「正文」文本，即模型回复内容

print()

