"""
【案例】Redis 持久化对话历史：RunnableWithMessageHistory + RedisChatMessageHistory

对应教程章节：第 16 章 - 记忆与对话历史 → 6、案例代码 → 6.2 持久化：Redis 存储 → Redis 对话历史示例

知识点速览：
- RedisChatMessageHistory(session_id=..., url=REDIS_URL) 将消息存到 Redis，重启后仍可恢复；同一 session_id 对应同一会话历史。
- RunnableWithMessageHistory 的 get_session_history 返回 RedisChatMessageHistory 实例，链会在每次 invoke 时读/写该会话历史。
- redis_client.save() 可强制持久化到 dump.rdb（按需调用）；默认连接 redis://localhost:6379，可通过环境变量 REDIS_URL 覆盖（如 Redis Stack 用 26379 时可设 REDIS_URL=redis://localhost:26379）。
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
import os
import redis
from loguru import logger

# 支持环境变量 REDIS_URL；未设置时默认 localhost:6379（标准 Redis），教程 Docker 可能用 26379
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def _check_redis():
    """启动时检查 Redis 是否可达，不可达时给出明确提示后退出。"""
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        r.close()
    except (redis.ConnectionError, redis.ResponseError) as e:
        logger.error(
            "Redis 连接失败（{}）。请先启动 Redis，例如：\n"
            "  docker run -d -p 6379:6379 redis\n"
            "若使用其他端口，可设置环境变量：REDIS_URL=redis://localhost:端口",
            REDIS_URL,
        )
        raise SystemExit(1) from e


_check_redis()

# 原生 Redis 客户端，decode_responses=True 使返回值为 str 而非 bytes
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder("history"), ("human", "{question}")]
)


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """为每个 session_id 创建/返回对应的 Redis 历史实例，实现持久化存储。"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
    )


chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
config = RunnableConfig(configurable={"session_id": "user-001"})

print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ["quit", "exit", "q"]:
        break
    response = chain.invoke({"question": question}, config)
    logger.info(f"AI回答:{response.content}")
    # 可选：强制将 Redis 内存数据持久化到磁盘（等同 redis-cli SAVE）
    redis_client.save()

"""
【输出示例】
开始对话（输入 'quit' 退出）
"""

# 输入问题：你好
# 2026-03-09 15:48:12.211 | INFO     | __main__:<module>:82 - AI回答:你好！很高兴见到你～😊 有什么我可以帮你的吗？无论是学习、工作、生活中的问题，还是想聊聊天、获取信息、一起动脑筋，我都很乐意帮忙！

# 输入问题：我叫黎明
# 2026-03-09 15:48:18.845 | INFO     | __main__:<module>:82 - AI回答:你好，黎明！✨ 很高兴认识你～这个名字真有诗意，让人联想到晨光初现、万物苏醒的温柔时刻 🌅
# 希望我们接下来的交流也能像黎明一样，带来清晰、温暖和一点点小期待 😊

# 你今天过得怎么样？或者有什么想聊的、想问的、想一起探索的？我在这儿认真听着呢～ 🌟

# 输入问题：我叫什么
# 2026-03-09 15:48:24.063 | INFO     | __main__:<module>:82 - AI回答:你叫**黎明** ✨
# ——这个名字我记住了，而且很喜欢它蕴含的光与希望感 🌅

# 需要我帮你做点什么吗？比如：
# 🔹 起个昵称或笔名（呼应“黎明”主题）
# 🔹 写一句专属的晨光小诗
# 🔹 规划一个清晨习惯计划
# 🔹 或者……单纯陪你聊聊今天的“光”在哪里？

# 随时告诉我～ 🌟

# 输入问题：redis-cli -h 127.0.0.1 -p 6379 PING
# 2026-03-09 15:48:49.818 | INFO     | __main__:<module>:82 - AI回答:你执行的是 Redis 的连通性测试命令：

# ```bash
# redis-cli -h 127.0.0.1 -p 6379 PING
# ```

# ✅ **预期正常响应是：**
# ```
# PONG
# ```

# 这表示：
# - Redis 服务正在本地（`127.0.0.1`）运行；
# - 端口 `6379` 可访问；
# - 客户端能成功连接并通信。

# ---

