"""
【案例】基于 mcp.json + LangChain Agent 的 MCP 客户端（LLM + MCP 工具）

对应教程章节：第 20 章 - MCP 模型上下文协议 → 6、案例实战：本地 MCP 天气服务与客户端

知识点速览：
- 从同目录的 mcp.json 加载 MCP 服务配置，使用 langchain_mcp_adapters 的 MultiServerMCPClient 连接多台
  MCP 服务器并获取工具列表，再交给 LangChain 的 create_tool_calling_agent + AgentExecutor 形成「LLM + MCP 工具」的对话 Agent。
- 流程：加载 mcp.json → 初始化 MultiServerMCPClient 并 get_tools() → 创建 DeepSeek 模型与提示模板 →
  组装 Agent 与 AgentExecutor → 启动命令行聊天循环（输入 quit 退出）。
- 依赖：pip install langchain-mcp-adapters langchain-openai langchain-classic loguru；部分适配器要求 Python 3.12 及以下。需配置环境变量 deepseek-api（或改用其他兼容 OpenAI 的 api_key/base_url）。
"""

import asyncio
import json
import os
from pathlib import Path

from loguru import logger

# 默认 mcp.json 路径（与本文件同目录）
_MCP_JSON_PATH = Path(__file__).resolve().parent / "mcp.json"


def load_servers(file_path: str | Path | None = None) -> dict:
    """
    加载 MCP 服务器配置。
    :param file_path: 配置文件路径，默认使用同目录下的 mcp.json
    :return: 完整配置字典，如 {"mcpServers": {"weather": {...}, "fetch": {...}}}
    """
    path = Path(file_path) if file_path else _MCP_JSON_PATH
    if not path.exists():
        logger.warning(f"未找到 mcp 配置文件: {path}")
        return {"mcpServers": {}}
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"已加载 mcp 配置: {path}，共 {len(config.get('mcpServers', {}))} 个服务")
    return config


async def run_chat_loop(config_path: str | Path | None = None) -> None:
    """
    启动并运行一个基于 MCP 工具的聊天 Agent 循环。
    该函数会：1）加载 MCP 服务器配置；2）初始化 MCP 客户端并获取工具；
    3）创建基于 DeepSeek 的语言模型和 Agent；4）启动命令行聊天循环；5）退出时清理资源。
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        logger.error(
            "请先安装 langchain-mcp-adapters: pip install langchain-mcp-adapters（部分环境需 Python 3.12 及以下）"
        )
        raise e

    from langchain_openai import ChatOpenAI
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    config = load_servers(config_path)
    servers = config.get("mcpServers", {})
    if not servers:
        logger.warning("mcp.json 中未配置任何服务，无法获取 MCP 工具")
        return

    # 初始化 MCP 客户端：connections 为 mcpServers 字典（服务名 -> url/transport 或 command/args/transport）
    client = MultiServerMCPClient(connections=servers)

    async with client.session():
        tools = client.get_tools()
        if not tools:
            logger.warning("未从 MCP 服务获取到任何工具，请确认服务已启动且 mcp.json 配置正确")
            return

        logger.info(f"已获取 {len(tools)} 个 MCP 工具: {[t.name for t in tools]}")

        # 语言模型（DeepSeek，与截图一致；可改为其他 OpenAI 兼容接口）
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("deepseek-api"),
            base_url="https://api.deepseek.com",
        )

        # 对话提示：系统提示要求使用工具完成用户请求，agent_scratchpad 供 Executor 填入中间步骤
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有用的助手，需要使用提供的工具来完成用户请求。"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="解析用户请求失败，请重新输入清晰的指令",
        )

        logger.info("\n MCP Agent 已启动，请先输入一个提问给(LLM+MCP)，输入 'quit' 退出")

        while True:
            try:
                user_input = input("\n您: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    logger.info("已退出")
                    break
                result = agent_executor.invoke({"input": user_input})
                output = result.get("output", result)
                print(f"\nAgent: {output}")
            except KeyboardInterrupt:
                logger.info("已退出")
                break


def main() -> None:
    asyncio.run(run_chat_loop())


if __name__ == "__main__":
    main()
