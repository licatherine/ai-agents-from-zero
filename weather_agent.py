import os
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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

# 4. 创建 Agent
tools = [get_weather]
# 使用 LangChain 的 create_agent 创建支持工具调用的 Agent
agent_executor = create_agent(llm, tools)

# 5. 运行和测试 Agent
if __name__ == "__main__":
    print("Agent 即将开始工作...\n")
    
    question = "请帮我查一下上海和北京今天的天气分别怎么样？出门需要带伞吗？"
    
    # 调用 agent
    result = agent_executor.invoke({"messages": [("human", question)]})
    
    print("\n" + "="*40)
    print("最终回答：")
    # LangGraph 返回的结果中，最后一条消息是 AI 的回复
    print(result["messages"][-1].content)
