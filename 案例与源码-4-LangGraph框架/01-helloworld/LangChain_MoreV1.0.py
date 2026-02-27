"""
【案例】多模型共存：同一脚本中接入通义与 DeepSeek

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld → 6、实战：多模型共存（通义 + DeepSeek）

知识点速览：
- 同一脚本可初始化多个聊天模型实例（不同 model、base_url、api_key），按场景选用或对比调用。
- 每个实例用 init_chat_model 单独配置，变量名区分（如 llm_qwen、llm_deepseek）便于后续复用。
- 通义用 model_provider="openai" + 阿里百炼 base_url；DeepSeek 可用 model_provider="deepseek" 或兼容接口。
"""

# ========== 1. 导入依赖与环境 ==========
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

load_dotenv(encoding='utf-8')  # 从 .env 加载，建议在 .env 中配置 QWEN_API_KEY、deepseek-api 等

# ========== 2. 实例化模型一：通义/百炼（OpenAI 兼容） ==========
llm_qwen = init_chat_model(
    model="qwen-plus",
    model_provider="openai",              # 阿里百炼为 OpenAI 兼容接口
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

print(llm_qwen.invoke("你是谁").content)

print("*" * 70)

# ========== 3. 实例化模型二：DeepSeek 官方 ==========
# model_provider 说明：deepseek-chat + base_url 为 DeepSeek 时，部分版本可自动推断 provider；
# 显式写 model_provider="deepseek" 更稳妥。若接其他厂商（如 OpenAI 兼容），则需写 model_provider="openai"。
llm_deepseek = init_chat_model(
    model="deepseek-chat",                # 对应 DeepSeek 非思考模式，思考模式常用 deepseek-reasoner
    model_provider="deepseek",
    api_key=os.getenv("deepseek-api"),    # .env 中配置 DeepSeek API Key
    base_url="https://api.deepseek.com"
)

# 多模型共存：两个实例可同时保留，按需调用
print(llm_deepseek.invoke("你是谁").content)
# 调试时可查看实例属性：print(llm_deepseek.__dict__)