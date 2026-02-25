# 示例：模型常用参数在代码中的用法（temperature、max_tokens 等）
# 对应文档 10 章 1.5.2。演示标准化参数如何影响模型行为（如 temperature 控制随机性）。
#
# 依赖：pip install langchain langchain-openai
# 运行前：在 .env 中配置 deepseek-api

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# ========== 2. 实例化时设置常用参数 ==========
# temperature：控制输出随机性，0 更确定、高重复，越大越随机、越有创意。
# 通常取 0~2，源于 OpenAI API 约定；具体上下界以所用 API 文档为准。
# 超过 2（如 2.1）可能被部分接口拒绝或截断，且与 2.0 效果差异不大，建议不超过 2。
model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com",
    temperature=0.7,   # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_tokens=256,  # 可选：限制单次回复长度
)

# ========== 3. 多次调用观察参数效果（如 temperature 对多样性的影响） ==========
for i in range(3):
    print(f"--- 第 {i + 1} 次 ---")
    print(model.invoke("写一句关于春天的词，14 字以内").content)


# 【输出示例】温度为 2.0 时，输出如下结果
# 莺惊柳浪春犁响，一鞭云水碧
# **诗家酥雨润，闲趁卖花声。**
# 寒威已退先春雨，万枝吐翠流云间。

# 【输出示例】温度为 0 时，输出如下结果
# 风软一溪云，花明两岸春。
# 风软一溪云，花明两岸春。
# 风软一溪云，花明两岸春。