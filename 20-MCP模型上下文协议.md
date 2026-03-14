# 20 - MCP 模型上下文协议

---

**本章课程目标：**

- 理解 **MCP（Model Context Protocol，模型上下文协议）** 是什么、解决什么痛点，以及和 [Tool](17-Tools工具调用.md) / [RAG](19-RAG检索增强生成.md) 的定位区别。
- 掌握 MCP 的客户端-服务器架构与两种通信模式（STDIO / SSE），能阅读和编写简单的 MCP 配置与本地服务端/客户端示例。
- 通过本地 MCP 天气服务与客户端案例，理解 MCP 的「工具暴露与调用」流程，为后续学习 [第 21 章 Agent 智能体](21-Agent智能体.md) 打基础。

**前置知识建议：** 已学习 [第 17 章 Tools 工具调用](17-Tools工具调用.md)，了解 Tool / Function Calling 的基本概念与 `@tool`、`bind_tools` 的用法；建议已学 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 1-3 章 RAG、微调、续训与智能体](1-3-RAG、微调、续训与智能体.md) 中关于智能体与 MCP 的概述。

**学习建议：** 先建立「为什么需要 MCP」的直观印象，再按「MCP 概念 → 架构与传输模式 → 本地案例」顺序学习；案例需 Python 3.12 及以下时会在文中标明。**与前后章衔接**：[第 17 章](17-Tools工具调用.md) 讲的是单进程内 Tool 的定义与调用，本章讲工具如何按**协议**跨进程/跨应用暴露与发现；学完本章后 [第 21 章 Agent](21-Agent智能体.md) 会演示如何把 MCP 提供的工具交给 Agent 使用。

---

## 1、为什么需要 MCP

### 1.1 之前痛点是什么

**知识局限性**：大模型如 DeepSeek Chat 的知识截止到 2024 年 7 月，无法自动获取最新信息，需要手动联网查询。

**功能分散**：不同 AI 框架(LangChain/Spring AI)各自实现相似功能(Function Calling/Tool Calling)，缺乏统一标准。

![](images/20/20-1-1-1.jpeg)

### 1.2 直观类比：「贾维斯」与万能适配器

很多同学都记得钢铁侠的助手「贾维斯」——一个能连接战甲、实验室、家里所有设备的智能管家。现实中，AI 也需要连接各种「外部能力」：查天气、读文档、查数据库、发邮件等。若每种能力都要单独开发接口，成本会非常高。

![](images/20/20-1-2-1.jpeg)

![](images/20/20-1-2-2.jpeg)

**为什么需要 MCP？** 可以概括为三点：

- **统一接口**：不同服务和数据库各有各的「说话方式」，AI 若逐个适配会很麻烦。MCP 提供统一的「翻译官」，让 AI 只需学一种协议就能与多种服务交互。
- **减少重复开发**：开发者不必为每个服务、每个 AI 应用单独写连接逻辑；按 MCP 规范暴露一次，支持 MCP 的应用都能复用。
- **更好协作与生态**：工具定义和元数据规范化后，更容易被社区检验和复用，跨模型、跨应用的适配成本也会降低。

**一个具体场景**：

- **现状**：让「同一个 AI 应用」同时做到联网搜索、发邮件、发博客等，每个功能单独做都不难，但全部集成进一个系统就非常吃力。
- **举例**：假如 IDE 里有一个 AI 助手，你希望它能：
  - 查本地数据库辅助开发；
  - 搜 GitHub Issue 判断是不是已知 bug；
  - 把 PR 的修改意见发到 Slack 做 Code Review；
  - 查询甚至修改 AWS、Azure 的配置完成部署。  
    这些能力背后对应数据库、GitHub、Slack、云厂商等一大堆不同接口，没有统一标准的话，每接一个就要写一套对接逻辑。
- **有了 MCP 之后**：只要这些服务都按 MCP 标准暴露，就像有了「万能接口」——AI 应用只需按 MCP 去连，就能同时用上上述各类能力，开发更高效、集成成本大大降低。

这也是 MCP 架构要解决的核心问题：让 AI 通过**一套协议**与**多种外部资源**协作。

---

## 2、MCP 是什么（入门概念）

- **一句话**：MCP（Model Context Protocol）是一套**标准化的通讯协议**，用于规范「大模型 / AI 应用」与「外部工具、数据源」之间的连接方式，让 AI 能以统一方式获取上下文（工具、资源、提示等）。
- **类比**：
  - **大模型版的 OpenFeign**：OpenFeign 用于微服务之间通讯，MCP 用于大模型与工具/数据源之间的通讯。  
    架构类比：传统微服务里「订单服务 → 支付服务 → 优惠券服务」通过 **OpenFeign** 通信；AI 系统里「订单智能体 → 支付智能体 → 优惠券智能体」通过 **MCP** 通信。
  - **AI 世界的「万能适配器」**：各种服务和数据库各有各的接口，MCP 像统一插头，让 AI 用同一套方式连接它们。
  - **后端同学可类比 gRPC**：gRPC 用标准化方式让不同语言的服务互相通信；MCP 则是专为 AI 设计的「接口与连接」标准，让 AI 与各种应用、数据源交互。

**官方与生态链接：**

- MCP 协议官网：https://modelcontextprotocol.io/introduction
- LangChain 对 MCP 的支持：https://docs.langchain.com/oss/python/langchain/mcp

**通俗理解**：MCP 是一种**开放协议**，规定了「应用程序如何把上下文提供给大模型」的标准。可以把它想象成 AI 应用的「USB-C 口」——就像 USB-C 用同一套接口连接手机、电脑、耳机、充电器等各种设备一样，MCP 用同一套方式把大模型与不同的数据源、工具连接起来，避免各接各的、互不兼容。

![](images/20/20-2-1-1.jpeg)

---

## 3、MCP 能做什么

### 3.1 统一接入与抽象

- **统一上下文接入**：以标准化方式把 LLM 需要的**上下文**（工具、资源、提示等）连接起来。可以把它理解为 **Agent 时代的「Type-C 协议」**——希望把不同来源的数据、工具、服务统一起来供大模型调用。
- **合久必分、分久必合**：早期每个软件（如微信、Excel）都要单独给 AI 做接口；MCP 统一标准后，类似「所有电器都用 USB-C」，AI 一个协议就能连接多种工具与数据源。
- **比 Function Calling 更高一层的抽象**：Function Calling 是「模型会调工具」的底层能力；MCP 是在此之上的**协议层**——规定工具如何暴露、如何被调用、如何被多端复用，是实现智能体的重要基础。

下面两图分别从「分」（各自为政）与「合」（统一协议）的角度做了对比。

![](images/20/20-3-1-1.jpeg)

> **说明**：「分」——各应用、各数据源各自对接，重复开发、难以复用。

![](images/20/20-3-1-2.jpeg)

> **说明**：「合」——通过 MCP 等统一协议，一次开发、多端复用。

**小结**：MCP 的核心价值是**不用重复造轮子**；按标准暴露成 MCP 服务后，可被多个 AI 应用复用，工具定义也更规范、更易维护。

---

## 4、怎么用 MCP

- **直接使用现成的 MCP 服务**：无需自己搭服务端，可到公开站点选用已部署的 MCP 服务器，在支持 MCP 的 AI 应用（如 Cursor、Claude Desktop）中配置后即可使用。**通俗理解**：访问某大厂或第三方提供的 MCP 服务，就是在用对方通过 MCP 协议暴露出来的一批 **Tool（工具方法）**（如搜索、读文档、查数据库等），你的 AI 应用连上后就能调用这些工具，无需自己实现对接逻辑。
  - 例如：https://mcp.so/zh 等平台收录了大量通用 MCP 服务，可按需选用。
- **本地自建 MCP 服务端 / 客户端**：用于学习协议、调试或对接内部系统。本课程会在后文给出「本地 MCP 天气服务端 + 客户端」的案例；若要使用 LangChain + 多服务 MCP 客户端（如 `langchain-mcp-adapters`），需注意其依赖与 Python 版本（如部分适配器要求 Python 3.12 及以下）。

![](images/20/20-4-1-1.jpeg)

> **说明**：MCP 资源站示例，可浏览、筛选并配置到 IDE 或 AI 应用中。

---

## 5、MCP 架构知识

MCP 采用**客户端-服务器架构**，核心角色如下。

![](images/20/20-5-1-1.jpeg)

> **说明**：MCP 架构概览。**MCP 主机（MCP Hosts）**：发起请求的 AI 应用（如聊天机器人、AI 驱动的 IDE）。**MCP 客户端（MCP Clients）**：在主机内部，与 MCP 服务器保持 1:1 连接。**MCP 服务器（MCP Servers）**：为客户端提供上下文、工具和提示。**本地资源 / 远程资源**：服务器可访问的本地文件、数据库，或通过 API 访问的远程服务。

| 角色           | 说明                                                        |
| -------------- | ----------------------------------------------------------- |
| **MCP 主机**   | 发起请求的 AI 应用程序（如聊天机器人、AI 驱动的 IDE）       |
| **MCP 客户端** | 在主机程序内部，与 MCP 服务器保持 1:1 的连接                |
| **MCP 服务器** | 为 MCP 客户端提供上下文、工具和提示信息                     |
| **本地资源**   | 本地计算机中可供 MCP 服务器安全访问的资源（如文件、数据库） |
| **远程资源**   | 通过 API 等访问的远程数据或服务                             |

**这些角色在流程里如何配合**：假设你在用 AI 编程助手写代码，这个助手就是 **MCP 主机**；它需要访问外部资源（代码库、文档、调试工具等）时，**MCP 服务器**就像「中介」，把这些资源和 AI 连在一起。下面用两个例子说明整体流程。

- **简单流程（查函数用法）**：你查某函数用法时，AI 通过 MCP 客户端向 MCP 服务器发请求 → 服务器在代码库或文档里查找 → 把结果返回给 AI → AI 根据返回信息生成代码或解释，展示给你。
- **复杂任务示例**：你可以直接对 AI 说：「帮我查一下最近数学考试的平均分，把不及格同学名单整理到值日表里，并在微信群提醒他们补考。」AI 会通过 MCP 自动完成：连接电脑读取 Excel 成绩、连接微信查找相关群聊、修改在线文档更新值日表。整个过程可以无需你一步步操作，数据也可以在本地处理，既安全又高效。

![](images/20/20-5-1-2.jpeg)

### 5.2 两种通信模式（STDIO / SSE）

MCP 常见有两种传输模式，分别适用于不同部署与集成场景。

![](images/20/20-5-2-1.jpeg)

> **说明**：左为 **SSE** 模式（基于 HTTP），右为 **STDIO** 模式（基于标准输入/输出）。

| 维度         | SSE（Server-Sent Events）             | STDIO                                  |
| ------------ | ------------------------------------- | -------------------------------------- |
| **典型场景** | 独立部署的 MCP 服务、长连接或流式推送 | 本地集成、命令行工具、与主进程同机部署 |
| **传输方式** | HTTP 长连接（Keep-Alive）             | 操作系统标准输入/输出（stdin、stdout） |
| **数据方向** | 服务端 → 客户端（单向推送）           | 双向流，可同时读、写                   |
| **连接保持** | 长连接，适合持续推送                  | 随进程存在，进程结束则连接结束         |
| **数据格式** | 文本流（如 EventStream 格式）         | 原始字节流                             |
| **异常情况** | 可依赖 HTTP 状态码、重连机制处理      | 进程退出或管道断开即结束               |

**一句话概括**：SSE 适合「独立服务、多客户端、网络访问」；STDIO 适合「单机、轻量、进程内或本地管道」。

### 5.3 MCP 服务端基本写法与常用 API（以 FastMCP 为例）

写一个 MCP 服务端时，通常需要：**创建 MCP 实例 → 用装饰器注册工具/资源/提示词 → 指定传输方式并启动**。下面以 FastMCP 为例说明常用 API，便于看案例代码时能对应到概念。

**1. 创建 MCP 实例**

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("服务名称")   # 传入一个名字，用于标识本服务
```

**2. 注册工具：`@mcp.tool()`**

把普通 Python 函数暴露为 MCP **工具**，客户端可以按名称和参数调用。

| 写法                         | 说明                                                                |
| ---------------------------- | ------------------------------------------------------------------- |
| `@mcp.tool()`                | 无参：工具名、描述由函数名和 docstring 自动生成，参数由函数签名推断 |
| `@mcp.tool("自定义工具名")`  | 可指定工具在协议中的名称                                            |
| `@mcp.tool(description="…")` | 可显式写描述，便于客户端/大模型理解何时调用                         |

函数需带类型注解（如 `def add(a: int, b: int) -> int`），客户端会根据签名传参。例如：

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b
```

**3. 注册资源：`@mcp.resource(uri)`**

把一段**静态或动态内容**通过 URI 暴露出去，供客户端按 URI 读取（如文档、配置）。

| 写法                             | 说明                                  |
| -------------------------------- | ------------------------------------- |
| `@mcp.resource("scheme://path")` | 必填：资源 URI，客户端通过该 URI 访问 |

例如：`@mcp.resource("greeting://default")`，返回的内容可由客户端拉取后作为上下文使用。

**4. 注册提示词模板：`@mcp.prompt()`**

把**带占位符的提示词**暴露为「提示词模板」，客户端传入参数后得到最终提示文本，再交给大模型使用。

| 写法            | 说明                                           |
| --------------- | ---------------------------------------------- |
| `@mcp.prompt()` | 函数参数即模板参数，返回值即生成的提示词字符串 |

例如：`greet_user(name, style="friendly")` 返回「为{name}写一句友善的问候」等，客户端可传入不同 `name`、`style` 生成不同提示。

**5. 启动服务：`mcp.run(transport=...)`**

| 参数                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| `transport="stdio"` | 通过标准输入/输出与调用方通信，需由 MCP 客户端进程启动本进程 |
| `transport="sse"`   | 以 HTTP + SSE 方式对外提供接口，可独立部署、多客户端连接     |

**小结**：`@mcp.tool()` = 暴露可调用的**工具**；`@mcp.resource(uri)` = 暴露可读的**资源**；`@mcp.prompt()` = 暴露**提示词模板**；`mcp.run(transport=...)` = 选择传输方式并启动服务。后续 6.2 的案例会直接用到这些 API。

**完整基本案例**（仅一个工具 + STDIO 启动，可直接保存为 `mcp_basic_demo.py` 后用 Cursor/Claude 等配置为 MCP 服务器运行）：

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BasicDemo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加，返回和。"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- **步骤**：创建实例 → 用 `@mcp.tool()` 注册一个 `add` 工具（含类型注解和 docstring）→ `mcp.run(transport="stdio")` 启动。客户端连接后即可发现并调用 `add`。
- **运行方式**：在终端直接运行会因无客户端而报 Invalid JSON，属正常；需在 Cursor/Claude Desktop 等中配置「MCP 服务器」为该脚本（如 `python mcp_basic_demo.py`），由客户端启动并通信。

---

## 6、案例实战：本地 MCP 天气服务与客户端

本小节给出「本地 MCP 天气服务端 + 简单客户端」的完整示例，便于理解 MCP 的暴露与调用方式。若使用基于 FastMCP 的 SSE 服务或 `langchain-mcp-adapters` 的多服务客户端，需注意：**部分依赖要求 Python 3.12 及以下**，请以当前环境与官方文档为准。

### 6.1 环境与依赖

- 安装示例中用到的库（如 `httpx`、`loguru` 等）；若使用 `langchain-mcp-adapters`，请按官方说明安装并注意 Python 版本。
- 天气接口需 OpenWeather API Key，可写入 `.env`（如 `OPENWEATHER_API_KEY=xxx`），参见 [第 17 章 Tools 工具调用](17-Tools工具调用.md) 中的天气助手准备。

### 6.2 MCP 服务端（天气查询）

**本节知识点速览：**

- **目标**：把「查天气」等能力封装成 MCP 工具，供客户端按协议发现与调用；服务端负责「注册并暴露」工具。
- **两种实现**：① 不用 FastMCP（本仓库 McpServer.py）：手写服务类与 `@mcp.tool()`，只演示工具注册思路，无真实网络监听，适合学原理、兼容高版本 Python。② 用 FastMCP（McpServerByFastMCP.py）：官方库提供 `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()` 及 STDIO/SSE 传输，可被 Cursor/Claude 等标准客户端连接，需 `pip install mcp`。
- **传输方式**：STDIO = 标准输入/输出，需由客户端进程启动本进程；SSE = HTTP 长连接，可独立部署。直接运行 STDIO 服务会因无客户端发 JSON 而报 Invalid JSON，属正常。
- **与 [第 17 章 Tool](17-Tools工具调用.md) 的关系**：Tool 是单进程内能力封装；MCP 服务端是把「工具」按协议暴露出去，供跨进程/跨应用调用。

---

本案例将「查询指定城市天气」封装为 MCP 工具，供客户端调用。下面先对比两种服务端写法，再给出对应源码。

**「用 FastMCP」和「不用 FastMCP」两种服务端实现有何区别？**

| 维度           | 本仓库 McpServer.py（不用 FastMCP）              | 使用 FastMCP（如 McpServerByFastMCP.py）                                                                    |
| -------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **实现方式**   | 自己写一个类，手写 `@mcp.tool()`、`_tools` 等    | 用官方库 `from mcp.server.fastmcp import FastMCP`，直接 `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()` |
| **协议与传输** | 仅演示「工具注册」思路，未实现真实 HTTP/SSE 监听 | 内置 STDIO/SSE 等传输，符合 MCP 协议，可被标准客户端连接                                                    |
| **适用场景**   | 学习原理、多 Python 版本兼容（如 3.13）          | 快速搭可被 Cursor/Claude 等连接的正式 MCP 服务                                                              |
| **依赖**       | 仅 httpx、loguru 等，无 mcp 包                   | 需 `pip install mcp`，部分环境需 Python 3.12 及以下                                                         |

**手写实现**（不依赖 FastMCP，适合学原理、多 Python 版本）：

【案例源码】`案例与源码-4-LangGraph框架/11-mcp/McpServer.py`

[McpServer.py](案例与源码-4-LangGraph框架/11-mcp/McpServer.py ":include :type=code")

> **说明**：在基于 FastMCP 的 SSE 实现中，常用 **HTTP 202 Accepted** 表示请求已接受、结果将通过 SSE 流式返回，以适配 MCP SSE 的流式处理特性；200 OK 则多用于一次性请求-响应。本仓库中的 `McpServer.py` 为简化版，重点展示「工具注册与暴露」的思路。

**使用 FastMCP 的实现**（无需手写 class，本质相同，仅传输/业务不同）：

下面两个示例都是 FastMCP：`mcp = FastMCP("名")` + `@mcp.tool()` + `mcp.run(...)`，与上面手写 class 形成对比。区别仅在于：左列演示多能力（tool/resource/prompt）+ STDIO；右列演示天气查询 + SSE，且 host/port 在 `run(transport="sse", host=..., port=...)` 时传入（构造函数不能传 host/port）。

| 手写（McpServer.py）           | FastMCP 示例一（Demo，STDIO）                           | FastMCP 示例二（天气，SSE）                                             |
| ------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| 需自己写 `MCPWeatherServer` 类 | 多能力：add、resource、prompt；`run(transport="stdio")` | 单工具 get_weather；`run(transport="sse", host="127.0.0.1", port=8000)` |

【案例源码】`案例与源码-4-LangGraph框架/11-mcp/McpServerByFastMCP.py` · 【案例源码】`案例与源码-4-LangGraph框架/11-mcp/McpServerWeatherByFastMCP.py`

[McpServerByFastMCP.py](案例与源码-4-LangGraph框架/11-mcp/McpServerByFastMCP.py ":include :type=code")

### 6.3 MCP 配置文件（多服务时使用）

**说明**：本节讲的是**客户端侧**的配置文件，不是「另一种 MCP 服务器」。当你的 **MCP 客户端**（如 LangChain 的 `MultiServerMCPClient` 或 Cursor/Claude）需要连接**多台** MCP 服务器时，通过一份 `mcp.json` 声明「连谁、用什么传输方式」即可。

**mcp.json 的定位与约束作用（调用说明）**

- **唯一契约**：`mcp.json` 是 Agent 调用工具的**「唯一契约」**——在规则层面严格约束 Agent 能调用哪些工具、怎么调用，从而消除多余调用和「随意性」，从根上减少错误。
- **操作边界**：配置中明确定义**可调用工具的全集**、**每个工具的调用规则**以及**入参/出参格式**；Agent 的理论行为被限制在该范围内，相当于为 Agent 划定了「操作边界」，未在配置中声明的工具对 Agent 不可见，从而自然避免越权或误调用。
- **可控性**：模型的随机性可通过 `mcp.json` 的结构化定义与 Agent 执行规则完全约束；若仍出现异常，多为约束不足，可通过完善配置与工程手段规避。

**工具定义规范（配置中应包含的要素）**

在 `mcp.json` 或与 MCP 协议配套的工具定义中，通常需要约定：

| 要素                     | 说明                                                           |
| ------------------------ | -------------------------------------------------------------- |
| **工具名**               | 唯一、简洁、无歧义，禁止同名或近似名                           |
| **工具描述**             | 限定使用场景并排除非场景，杜绝多余调用                         |
| **入参类型 / 必选性**    | 必选参数写死，无参数时直接拒绝调用                             |
| **参数格式**             | 如 city 仅支持地级市名称等，在 description 中写清              |
| **返回值规范**           | 约定返回结构，便于 Agent 解析                                  |
| **additionalProperties** | 建议设为 `false`，禁止传入配置外的任何参数，从根上避免参数传错 |

**单工具定义示例（get_weather）**

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "仅用于查询指定城市的实时天气，非天气查询需求禁止调用",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "仅支持国内地级市名称，如北京、上海，不接受省/区/县"
          }
        },
        "required": ["city"],
        "additionalProperties": false
      }
    }
  ]
}
```

上述示例中：`name` 唯一标识工具；`description` 限定场景；`required` 强制必填；`additionalProperties: false` 禁止额外参数，避免调错。

---

仓库中已提供 `mcp.json`，路径为 `案例与源码-4-LangGraph框架/11-mcp/mcp.json`，供多服务客户端加载；可按需修改。

**配置示例说明（服务连接）：**

- **天气服务（SSE 模式）**：`weather` 使用 `transport: "sse"`，通过 HTTP 长连接与本地 8000 端口的 `/sse` 通信，适合独立运行的 MCP 服务。
- **网页抓取服务（STDIO 模式）**：`fetch` 使用 `transport: "stdio"`，通过 `uvx` 运行 `mcp-server-fetch`，适合本地命令行工具、无需单独网络端点。

【配置文件】`案例与源码-4-LangGraph框架/11-mcp/mcp.json`

[mcp.json](案例与源码-4-LangGraph框架/11-mcp/mcp.json ":include :type=code")

- **作用**：调用 `weather` 可获取天气数据；调用 `fetch` 可抓取并解析网页内容，让 AI 能基于链接内容回答问题。

### 6.4 MCP 客户端（本地调用示例）

下面示例演示如何在本机直接调用 MCP 服务端已注册的天气工具，不依赖 LangChain Agent，也不依赖 `langchain-mcp-adapters`（通过同进程导入 mcp 实例或加载 mcp.json 仅读配置）。若要将 MCP 工具与 LangChain Agent 结合（从 mcp.json 加载配置、MultiServerMCPClient 获取工具并交给 AgentExecutor），见 [第 21 章 - Agent 智能体](21-Agent智能体.md) 中的 **5.4 Agent + MCP 工具（mcp.json）**。

【案例源码】`案例与源码-4-LangGraph框架/11-mcp/McpClient.py`

[McpClient.py](案例与源码-4-LangGraph框架/11-mcp/McpClient.py ":include :type=code")

### 6.5 测试建议

- **问题 1（调用 MCP 天气）**：先启动 `McpServer.py`，再运行 `McpClient.py`，查询如「北京」对应城市（如 Beijing）的天气，确认能拿到 OpenWeather 返回并正确解析。
- **问题 2（若已配置 fetch 等 MCP 服务）**：在支持 MCP 的 AI 应用或自写 Agent 中，可提问如「请总结 https://github.langchain.ac.cn/langgraph/reference/mcp/ 这篇文档」，由 AI 通过 fetch 类工具抓取页面后再总结。

---

**本章小结：**

- **MCP** 是规范「大模型与外部工具/数据源」连接的**标准化协议**，解决接口不统一、重复开发、协作难等问题；可类比「大模型版的 OpenFeign」或 AI 世界的「万能适配器」。架构上采用客户端-服务器模型，常见传输模式有 **STDIO**（本地/进程内）和 **SSE**（网络、流式）。
- 学习时可通过本地 MCP 天气服务端与客户端（`McpServer.py`、`McpClient.py`）理解「工具暴露与调用」；多服务场景可配合 `mcp.json` 与 LangChain MCP 适配器（注意 Python 版本与依赖）。
- **定位速记**：[Tool](17-Tools工具调用.md) 让大模型能用工具；[RAG](19-RAG检索增强生成.md) 让大模型获得检索上下文；**MCP** 让大模型与工具/服务之间的连接标准化、可复用。与 [Agent](21-Agent智能体.md)、Function Calling 等的详细对比见 [第 21 章 Agent 智能体](21-Agent智能体.md) 第 6 节。

**建议下一步：** 在本地跑通 `McpServer.py` 与 `McpClient.py`，巩固 MCP 的配置与调用；接着学习 [第 21 章 Agent 智能体](21-Agent智能体.md)，理解 [Tool](17-Tools工具调用.md) 与 Agent 的配合及 ReAct、A2A 等案例。若需更复杂的图编排与多步工作流，可继续学习 **LangGraph** 相关章节。MCP 在 Java 生态也有实现，可参考 B 站视频（74–80 集）：https://www.bilibili.com/video/BV1pvWGznEqh。
