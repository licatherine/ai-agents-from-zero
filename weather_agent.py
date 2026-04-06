import os
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义天气查询工具
@tool
def get_weather(location: str) -> str:
    """查询指定城市的天气状态。参数 location 必须是城市的名称（例如 'Beijing', 'Shanghai', 'London'）。"""
    try:
        # 这里使用了 wttr.in 这个免费且简单的天气 API
        # format=3 会返回形如 "Beijing: ☀️ +13°C" 的简洁文本
        response = requests.get(f"https://wttr.in/{location}?format=3")
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"查询 {location} 天气失败: {e}"

# 2. 初始化语言模型（使用通义千问）
# 通义千问完全兼容 OpenAI 的接口格式，所以我们可以无缝使用 ChatOpenAI 取代
# 请确保你的系统环境变量中设置了 DASHSCOPE_API_KEY（阿里云通义千问的 API Key）
# 例如：export DASHSCOPE_API_KEY="sk-xxxx"
llm = ChatOpenAI(
    model="qwen-plus", # 你可以改成 qwen-max 或 qwen-turbo
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# 3. 编写 Agent 的 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个得力的智能助手，你可以使用提供的工具来帮助用户解答问题。请使用中文回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # 这个占位符用于存放 Agent 思考和工具调用的中间记录
])

# 4. 创建 Agent 以及 Executor
tools = [get_weather]
# 将模型、工具和提示词组装成一个支持工具调用的 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
# AgentExecutor 负责实际运行上面的 agent，处理重试、报错及解析输出（verbose=True 会在控制台打印思考过程）
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 运行和测试 Agent
if __name__ == "__main__":
    print("Agent 即将开始工作...\n")
    
    question = "请帮我查一下上海和北京今天的天气分别怎么样？出门需要带伞吗？"
    
    # 传入输入字典调用 agent
    result = agent_executor.invoke({"input": question})
    
    print("\n" + "="*40)
    print("最终回答：")
    print(result["output"])
