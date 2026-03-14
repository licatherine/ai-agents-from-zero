# 17 - Tools 工具调用

---

**本章课程目标：**

- 理解**工具调用（Tool / Function Calling）**是什么、能干什么，以及「模型输出调用意图、代码真正执行」的分工关系。
- 掌握使用 **@tool** 装饰器定义 LangChain 工具，会配合 **Pydantic** 定义参数 schema，并会查看工具的 name、description、args 等属性。
- 完成「天气查询助手」从定义工具、绑定模型、解析 tool_calls 到执行工具并生成自然语言回复的完整链路。

**前置知识建议：** 已学习 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)、[第 15 章 LCEL 与链式调用](15-LCEL与链式调用.md)；建议已学 [第 16 章 记忆与对话历史](16-记忆与对话历史（含Redis基础）.md)。具备 Python 与 HTTP 请求（如 httpx）基础更佳。工具返回结果会以 [ToolMessage](13-提示词与消息模板.md) 形式回传模型，与 [第 11 章](11-Model-I-O与模型接入.md) 的 AIMessage.tool_calls 对应。

**学习建议：** 先理解「为什么不调用工具有局限」与工作流程，再从小工具（加法）入手掌握 @tool 与 Pydantic，最后按步骤完成天气助手的定义、绑定与调用链。

---

## 1、为什么不调用工具会有局限

大模型虽然具备强大的语言理解与生成能力，但本质上是**静态的、不可直接交互**的：

- 不能直接访问数据库或调用外部 API
- 不能执行代码或文件操作
- 无法实时访问互联网或动态数据

因此需要**工具（Tool）**机制：由模型决定「要不要调、调哪个、传什么参数」，由我们的代码**真正执行**工具并把结果返回给模型，再由模型组织成自然语言回复。本章掌握单工具与多工具的手动调用；**何时调、按什么顺序调**则由 [第 21 章 Agent 智能体](21-Agent智能体.md) 的决策与编排来完成，二者配合形成完整「工具 + 智能体」能力。

<img src="images/17/17-1-1-1.jpeg" alt="大模型不具备直接访问外部数据与执行能力，需通过工具扩展" width="70%" />

---

## 2、Tool 是什么与能干嘛

**是什么**：**Tool（工具）** 让模型具备「调用外部函数」的能力，可与外部 API、数据库或自定义逻辑交互。在 Tool Calling（也称 Function Calling）里，**LLM 只负责输出「要调哪个工具、传什么参数」**，真正执行工具并把结果送回模型的是你的代码。

**能干嘛**：

- **访问实时数据**：天气、股票、新闻等（需由你提供工具接口，否则模型只能依赖训练知识或猜测）。
- **执行操作**：发邮件、查订单、调支付、查快递等，只要封装成工具并暴露给模型即可。

**一句话**：工具是 LLM 的「外部能力」——模型决定是否调、调谁、传什么；程序负责执行并回传结果。

**参考链接**：

- LangChain 工具文档：https://docs.langchain.com/oss/python/langchain/tools
- LangChain 内置工具列表：https://docs.langchain.com/oss/python/integrations/tools
- Spring AI 工具：https://docs.spring.io/spring-ai/reference/api/tools.html

---

## 3、工作流程概览

典型流程为：用户提问 → 模型判断是否需要调工具 → 若需要，返回 tool_calls（函数名 + 参数）→ 你的代码执行对应工具 → 将工具结果塞回对话 → 模型根据结果生成最终回复。

![工具调用整体流程：用户输入 → 模型决策 → 执行工具 → 结果回填 → 模型生成回复](images/17/17-3-1-1.jpeg)

> **小知识：泳道图 vs 普通流程图（面试可考）**

> 上图为**泳道图**（Swimlane diagram），按角色/系统分栏表示「谁在什么阶段做什么」。
>
> - **普通流程图**：只表示步骤的先后顺序和分支/判断，不区分「谁」执行哪一步；适合单角色、单系统内的流程（如算法步骤、单一业务线）。
> - **泳道图**（Swimlane diagram）：按角色/部门/系统划分泳道，每个步骤落在对应责任方的泳道里，一眼看出「谁做啥」；适合多角色协作、跨部门/跨系统流程。
> - **何时用泳道图**：流程涉及多个责任主体（用户、模型、应用程序、第三方服务等）、需要明确责任边界与交接点时用泳道图。例如工具调用（用户 → 模型 → 你的代码 → 模型）、审批流、跨系统对接等。

---

## 4、自定义 Tool

### 4.1 使用 @tool 装饰器

**装饰器小知识**：Python 的**装饰器**（decorator）是一种语法 `@xxx`，在不修改原函数代码的前提下，给函数「包一层」逻辑（如校验、日志、注册到框架等）。本质是「函数作为参数传入、返回新函数」的高阶函数。常见装饰器举例：

- **标准库**：`@staticmethod` / `@classmethod`（方法类型）、`@property`（像属性一样访问）、`@functools.lru_cache`（缓存返回值）。
- **Web 框架**：Flask 的 `@app.route("/path")`、FastAPI 的 `@app.get("/path")`。
- **本课程**：LangChain 的 **`@tool`** —— 把普通函数包装成可供 LLM/Agent 调用的工具。

用 **@tool** 装饰器可以把一个 Python 函数变成 LangChain 的 Tool，模型会通过函数的**名称、文档字符串（description）和参数**来决定是否调用以及如何传参。

![@tool 装饰器将函数转为 Tool，供模型识别与调用](images/17/17-4-1-1.jpeg)

**Tool 常用属性**（了解即可）：可通过 `tool.name`、`tool.description`、`tool.args` 查看；含义如下表。

| 属性            | 类型               | 描述                                                                               |
| :-------------- | :----------------- | :--------------------------------------------------------------------------------- |
| `name`          | `str`              | 必选，在提供给 LLM 或 Agent 的工具集中必须唯一。                                   |
| `description`   | `str`              | 可选但建议提供，描述工具的功能；LLM/Agent 会据此决定是否及如何调用该工具。         |
| `args_schema`   | Pydantic BaseModel | 可选但建议，用于提供更多信息（如 few-shot 示例）或校验预期参数。                   |
| `return_direct` | `boolean`          | 仅对 Agent 有意义：为 `True` 时，调用该工具后 Agent 会停止并将结果直接返回给用户。 |

**示例**（以加法工具 `add_number` 为例）：

```python
@tool
def add_number(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

# 常用属性示例
add_number.name          # → 'add_number'（默认取函数名，也可在 @tool(name="xxx") 中指定）
add_number.description   # → '两个整数相加'（默认取函数 docstring）
add_number.args          # → {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
add_number.return_direct # → False（默认）；若需直接返回结果可写 @tool(return_direct=True)
```

使用 Pydantic 时，可通过 `args_schema` 绑定参数模型，此时 `tool.args` 会反映 schema 中的字段与描述，例如：

```python
from pydantic import BaseModel, Field

class FieldInfo(BaseModel):
    a: int = Field(description="第1个参数")
    b: int = Field(description="第2个参数")

@tool(args_schema=FieldInfo)
def add_number(a: int, b: int) -> int:
    return a + b
# add_number.args 会包含 a、b 的类型与 description
```

![Tool 参数与返回的约定](images/17/17-4-1-2.jpeg)

### 4.2 基础案例：加法工具

【案例源码】`案例与源码-4-LangGraph框架/08-tools/Tool_AddNumberTool.py`

[Tool_AddNumberTool.py](案例与源码-4-LangGraph框架/08-tools/Tool_AddNumberTool.py ":include :type=code")

### 4.3 Pydantic 与参数 schema

使用 **Pydantic** 定义参数模型（如 `FieldInfo`），再通过 `args_schema` 传给 `@tool`，可以更精确地描述参数类型与说明，便于模型生成正确的参数。下面先对 Pydantic 做个简要入门。

---

#### Pydantic 简要入门

**是什么**：Pydantic 是 Python 里基于**类型注解**做「数据校验 + 自动转换」的库。你定义一个「模型类」（继承 `BaseModel`），在**创建实例的那一刻**就会按类型和规则校验、转换传入的数据，不合法就抛出 `ValidationError`。

**基本概念**：

- **模型（Model）**：继承 `pydantic.BaseModel` 的类，每个属性就是一个字段。
- **字段（Field）**：用类型注解声明（如 `name: str`），可用 `Field(...)` 加描述、默认值、取值范围等。
- **实例化 = 校验**：`User(name="张三", age=18)` 时，Pydantic 会检查类型、做合理转换（如 `"18"` → `18`），不合格就报错。

**基本语法示例**：

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str                          # 必填，必须是 str
    age: int = 0                       # 可选，默认 0，传入会被转成 int
    score: float = Field(gt=0, le=100) # 可选，且必须在 (0, 100] 之间
```

**常用写法**：

- **类型**：`str`、`int`、`float`、`bool`、`list[...]`、`dict` 等；可选用 `Optional[str]`、`str | None`。
- **Field 常用参数**：`description`（描述，给 LLM 看）、`default`（默认值）、`gt`/`ge`/`lt`/`le`（大于/大于等于/小于/小于等于，用于数值范围）。
- **常用方法**：`model.model_dump()` 得到字典，`model.model_dump_json()` 得到 JSON 字符串。

**用了 Pydantic vs 不用**（对比）：

| 维度        | 不用 Pydantic                                       | 用 Pydantic                                                                             |
| ----------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 类型与格式  | 需手写 `if type(x) != int`、自己转 `int(x)`，易漏写 | 声明类型即可，实例化时自动校验 + 合理转换                                               |
| 错误反馈    | 可能后面才报错或静默出错，难排查                    | 在「创建对象」处统一抛 `ValidationError`，字段级错误信息清晰                            |
| 与 LLM/工具 | 参数说明散落在代码里，不好给模型看                  | 用 `Field(description=...)` 写好说明，直接当 `args_schema` 给模型，模型更易生成正确参数 |

在本章里，把「工具参数」定义成 Pydantic 模型并传给 `@tool(args_schema=...)`，既能让运行时更安全，也能让模型看到规范的参数说明，一举两得。

**一句话**：Pydantic = 「类型注解 + 自动校验 + 转换」神器，让 Python 在运行时也能享受「静态类型」的安全感。

【案例源码】`案例与源码-4-LangGraph框架/08-tools/PydanticDemo.py`

[PydanticDemo.py](案例与源码-4-LangGraph框架/08-tools/PydanticDemo.py ":include :type=code")

【案例源码】`案例与源码-4-LangGraph框架/08-tools/Tool_AddNumberToolPro.py`

[Tool_AddNumberToolPro.py](案例与源码-4-LangGraph框架/08-tools/Tool_AddNumberToolPro.py ":include :type=code")

---

## 5、天气助手实战

### 5.1 Tool Calling 原理简述

请求时把「工具列表」（名称、描述、参数 schema）一并发给模型。模型若判断需要查天气，会返回 **function_call** 类型的消息（包含工具名与参数）。你的代码根据返回结果**真正调用**天气 API，再把结果和已有对话一起发给模型，由模型整理成自然语言回复。

![模型返回 tool_calls → 执行工具 → 结果回填 → 模型生成最终回答](images/17/17-5-1-1.jpeg)

#### 常见疑问：模型是如何判断「什么时候该调用工具」的？

**模型看到了什么**：使用 `bind_tools([get_weather])` 时，每次请求发往 API 的除了**用户消息**，还会带上**工具列表**——每个工具的名称（`name`）、描述（`description`，即你在 `@tool` 函数上写的 docstring）、以及参数 schema（如 `loc` 的含义与类型）。模型在同一轮输入里同时看到「用户问了什么」和「当前有哪些工具、每个工具是干什么的」。

**根据什么来判断**：模型依靠**语义理解**做匹配，而不是我们写 if/else 规则：

| 依据                        | 作用                                                                                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **用户消息**                | 理解用户意图（是在问天气、算数，还是闲聊）。                                                                                                                                                                                   |
| **工具描述（description）** | **最关键**。教程 4.1 表格中写明：LLM/Agent 会据此决定是否及如何调用该工具。例如 docstring 里写「查询即时天气」「参数 loc 为城市名」，用户问「北京今天天气如何」时，模型就会把二者对上，输出调用 `get_weather(loc="Beijing")`。 |
| **工具名称（name）**        | 辅助信号，如 `get_weather` 进一步暗示与天气相关。                                                                                                                                                                              |
| **参数 schema**             | 告诉模型要传哪些参数、如何填（如 `loc` 填城市名），便于生成正确的 arguments。                                                                                                                                                  |

**能力从哪来**：支持 Function Calling 的模型在训练/微调时学过「当输入里既有用户问题又有工具定义时，可以输出结构化 tool_calls」。推理时我们只负责：① 用 `bind_tools` 把工具列表交给模型；② 根据模型返回的 tool_calls 去真正执行 `get_weather.invoke(...)`。**「要不要调、调哪个、传什么」由模型根据上述依据一次性推断，我们的代码只提供工具定义并负责执行。**

### 5.2 需求与准备

- **功能**：实现天气查询——调用 OpenWeather API 获取指定城市实时天气，并将结果用自然语言描述给用户。
- **步骤**：构建请求、发送 HTTP 请求、解析 JSON、格式化为中文描述。
- **API Key**：在 https://home.openweathermap.org/api_keys 免费申请，写入 `.env`（如 `OPENWEATHER_API_KEY=xxx`）。天气 API 文档：https://openweathermap.org/

![API Keys](images/17/17-5-2-1.png)

### 5.3 定义天气工具

【案例源码】`案例与源码-4-LangGraph框架/08-tools/QueryWeatherTool.py`

[QueryWeatherTool.py](案例与源码-4-LangGraph框架/08-tools/QueryWeatherTool.py ":include :type=code")

### 5.4 大模型调用天气工具并生成回复

下面示例：将 `get_weather` 绑定到模型（`bind_tools`），用 `JsonOutputKeyToolsParser` 解析模型返回的 tool_calls，再执行天气工具，最后用另一条链把 JSON 天气数据转成自然语言描述。

【案例源码】`案例与源码-4-LangGraph框架/08-tools/LLMQueryWeatherDemo.py`

[LLMQueryWeatherDemo.py](案例与源码-4-LangGraph框架/08-tools/LLMQueryWeatherDemo.py ":include :type=code")

> **注意**：`LLMQueryWeatherDemo.py` 中通过 `from QueryWeatherTool import get_weather` 引用同目录下的天气工具，运行前请确保已配置 `OPENWEATHER_API_KEY`（或在 `QueryWeatherTool.py` 中按需改为从环境变量读取）。

---

**本章小结：**

- **工具调用（Tool / Function Calling）**让[模型](11-Model-I-O与模型接入.md)具备「调用外部函数」的能力；模型只负责输出要调用的工具名与参数，**真正执行工具的是你的代码**。
- **自定义工具**：用 **@tool** 装饰器将函数转为 Tool，通过文档字符串与参数类型供模型识别；可用 **Pydantic**（[第 14 章](14-输出解析器.md) 亦有 Pydantic 结构化输出）定义 `args_schema` 细化参数说明与校验。
- **完整链路**：定义工具 → 用 `bind_tools` 绑定到模型 → 解析模型返回的 tool_calls → 执行工具 → 将结果回填对话 → 模型生成自然语言回复。天气助手案例贯穿上述步骤，并借助 OpenWeather API 与[链](15-LCEL与链式调用.md)完成「问天气 → 查 API → 中文描述」的闭环。

**建议下一步：** 在本地配置 OpenWeather API Key 并跑通天气助手案例；若需更复杂的多步决策、多工具编排或与[记忆](16-记忆与对话历史（含Redis基础）.md)结合，可继续学习 [第 21 章 Agent 智能体](21-Agent智能体.md)；若需跨进程、标准化工具协议，可了解 [第 20 章 MCP 模型上下文协议](20-MCP模型上下文协议.md)。
