# 22 - LangGraph 概述与快速入门

---

**本章课程目标：**

- 理解 LangGraph 是什么、与 LangChain / Agent 的关系，以及为何需要「图」来编排复杂工作流。
- 掌握 LangGraph 四大核心概念：**State（状态）、Nodes（节点）、Edges（边）、Graph（图）**，能独立完成从定义状态、加节点、连边到编译运行的完整流程。
- 会使用 HelloWorld 与简单业务图案例，并了解图的多种可视化方式（ASCII、Mermaid、PNG）。

**前置知识建议：** 已学习 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)，了解 LangChain 的链式思维与模型调用方式；若已学 [第 21 章 Agent 智能体](21-Agent智能体.md)，可更好理解「可控图」与「黑箱 Agent」的对比。

**学习建议：** 先建立「为什么需要 LangGraph」的直觉（Chain 太线性、Agent 太黑箱），再按「State → Nodes → Edges → 编译 → 运行」的顺序动手写 HelloWorld；图的构建流程与可视化在本章完成，**Graph API 中「图」与「状态」的深入讲解见 [第 23 章](23-LangGraphAPI：图与状态.md)**。

---

## 1、LangGraph 是什么

### 1.1 简介与定位

**一句话定义：** LangGraph 是基于 LangChain 构建的、面向**智能体多轮交互 / 状态持久化 / 分支与并行执行**的**图结构工作流框架**。可以记作：**LangGraph = LangChain + 图编排 + 状态机**。

**官方文档：**

- 英文：https://docs.langchain.com/oss/python/langgraph/overview
- 中文：https://docs.langchain.org.cn/oss/python/langgraph/overview

**与 LangChain 的关系**：LangGraph **基于** LangChain 构建。无论图结构多复杂，**单次执行链路仍然是线性的**，背后仍依赖 LangChain 的 Chain 等组件。可以这样理解：**LangGraph 是 LangChain 工作流的「高级编排工具」**——能按**图结构**编排工作流，而不再局限于单链。

在 LangChain 课程中我们说过：**Chain 就像一条工作流水线**——原料 → A 处理 → B 处理 → … → 最终产品。这种模式清晰且高效，例如「先总结文本，再翻译成英文」这类一步接一步的线性任务，用 Chain 非常合适。但现实中的很多任务并不是一条直线，而是充满**循环、判断和分支**，此时就需要图结构来编排。

### 1.2 从案例理解「线性」的局限

想象一个真实场景：你让一名员工**编写一份小米 SU7 跑车市场分析报告**。他的工作流程可能是：

1. 上网搜索相关资料
2. 根据资料写出第一版草稿
3. 自己审阅草稿，发现「数据不够支撑论点」
4. **返回第一步**，进行新一轮搜索，补充更多数据
5. 重写或修改草稿
6. 又觉得「结构有点乱」
7. 于是不搜索了，而是**直接对现有内容重新组织**
8. 最后觉得「差不多了」，才把报告交给你

请思考：上述过程还是**一条流水线**吗？能否「打直球」一步走到底？显然不能——过程中有**循环**（重新搜索）、**判断**（是继续搜还是改结构）和**分支**（不同决策走向不同步骤）。他会根据当前草稿的**状态**，决定下一步是「重新搜索」「重新组织」还是「提交」。

![基于 LLM 的智能体工作流示意：开始 → 大模型节点 → 边节点（下一步动作）→ 结束或执行 Action 并回到大模型](images/22/22-1-2-1.jpeg)

若用 LangChain 的 **Chain** 来模拟上述过程，会非常痛苦：Chain 天生是单线执行，很难实现「返回上一步」或「根据条件跳转到某一步」这种灵活控制流，开发者往往要写大量不优雅的胶水代码，逻辑容易变成一团乱麻。

### 1.3 那用 Agent 不行吗？

LangChain 的 **Agent** 在某种程度上解决了「循环与决策」的问题。Agent 像一名有自主决策能力的「将军」，基于 ReAct（Reason + Act）框架，可以自己决定调用什么工具、何时再搜一次等，确实能实现循环。

但 Agent 的最大问题在于它是一个**黑箱**：你给它目标和工具，它就开始「自言自语」（Reasoning）和「手忙脚乱」（Acting），你作为开发者很难对工作流程做**精细化控制和干预**。例如：

1. **无法强制流程**：你不能要求它「必须先写草稿再批判」，它可能搜完觉得够了就直接给最终答案。
2. **难以纠错**：一旦在错误思路上循环十几次，会浪费大量时间和 API Token，最后可能仍给出错误结论。
3. **行为不稳定**：同样的问题，这次可能是 A→B→C，下次可能是 A→C→B，结果难以预测。

对于需要**可靠、可控、可预测**的商业级 AI 应用来说，这种黑箱智能体就像「能力很强但野性难驯」的员工，真正核心的任务不敢完全交给它。因此需要：**疑人要用，用人要疑——监控过程，核实结果**。

下面两图分别从**人机协作（HITL）**和**多智能体协作**的角度，补充了智能体在实际应用中的需求，这些恰恰是 LangGraph 可以更好支持的场景。

![人机协作（HITL）：课程内容经 AI 分析师建议 → AI 改写器执行修改，不确定时交由人类专家审核决策再回到改写器，最终输出](images/22/22-1-3-1.jpeg)

![多智能体协作：分层规划与共创协作两种模式，模拟现实团队工作方式](images/22/22-1-3-2.gif)

### 1.4 小结：为何选 LangGraph、如何入门

**为何需要 LangGraph**：Chain 太「流水线」，难以处理循环与条件分支；Agent 太自由、像黑箱，难以控制与保证稳定性。LangGraph 把工作流抽象为**有向图**，在可控性与表达能力之间取得平衡，把基础单元从「链」升级为「图」。

**LangGraph 带来的能力**：

- **状态管理**：在不同节点之间传递和维护信息，支持长期记忆与多轮对话。
- **精确控制**：通过定义**节点和边**，可以精确控制执行逻辑，包括条件分支、循环和并行执行。
- **工具集成**：无缝集成搜索引擎、数据库、API 等外部工具，扩展 LLM 能力边界。
- **可观测性**：图结构使运行路径清晰可见，便于理解决策过程并快速定位、调试问题。
- **模块化与可复用**：每个节点可以是独立、可复用的组件；通过子图机制，复杂工作流可拆成多个可独立开发与测试的模块。

**四大核心概念**（构建流程：先定义状态 → 添加节点 → 用边连接 → 编译 → 执行）：

| 概念              | 含义                         |
| ----------------- | ---------------------------- |
| **State（状态）** | 数据在节点间存储与传递的载体 |
| **Nodes（节点）** | 执行具体逻辑的函数           |
| **Edges（边）**   | 定义节点间的流转顺序         |
| **Graph（图）**   | 由前三者构成的完整工作流     |

![构建基石：State 为中心，Nodes 执行逻辑，Edges 定义顺序](images/22/22-1-4-1.jpeg)

**应用场景决策**：何时选 LangGraph？可参考下表——适合时选它，否则用更简单方案更高效。

| 更适合用 LangGraph 的场景                    | 不适合用 LangGraph 的场景（建议用 LangChain 等方式）                |
| -------------------------------------------- | ------------------------------------------------------------------- |
| **多步复杂推理**：多轮思考、条件分支或循环   | **简单一次性查询**：一问一答、无状态与分支 → 用 **LangChain Chain** |
| **长时间运行**：断点续跑、状态持久化         | **快速原型**：只验证想法 → 用 **Coze、Dify** 等快速搭 demo          |
| **人在环（HITL）**：关键步骤需人工审核或干预 |                                                                     |
| **生产级稳定性**：可观测、可调试、可复现     |                                                                     |

**安装与文档**：安装见 https://docs.langchain.com/oss/python/langgraph/install ；入门只需记住四词：**State、Nodes、Edges、Graph**。图的五步构建与可视化方式见下一节「HelloWorld 快速入门」。

![LangGraph 把复杂 AI 应用从「一条直线」变成「一张网」](images/22/22-1-4-2.gif)

---

## 2、HelloWorld 快速入门

### 2.1 前置约定

动手前建议约定好**模型与配置**（API Key、模型名、Base URL），与 [第 10 章](10-LangChain快速上手与HelloWorld.md) 的「调用三件套」一致。下表示意课程中常用的模型与平台约定；其他模型可参考官方文档或本仓库其他章节。

| 模型/平台                   | 主要特点                                 | 优势                                           | 备注                                                                              |
| --------------------------- | ---------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| **OpenAI**                  | GPT 系列，文本生成与理解能力强           | 灵活、适用场景广，业界常用标杆                 | 国内 API 已暂停，需通过 Azure 等渠道访问                                          |
| **阿里百炼**                | 通义千问等大模型服务                     | 性能接近 GPT-4、价格较低、支持企业迁移与私有化 | 主要面向企业；新用户有较多免费 Token 与绘图额度                                   |
| **DeepSeek**                | 开源大模型，多语言支持                   | 推理与代码能力突出，社区活跃，可做多种应用     | 性价比高；注意内容审核与账号风控政策                                              |
| **智谱清言（Zhipu）**       | 基于 GLM 架构，多轮对话与复杂指令        | 指令理解好，支持多场景定制                     | 模型较全；活动期常有大幅降价或免费额度（如国庆 1 折、亿级 Token 等）              |
| **硅基流动（SiliconFlow）** | AI 基础设施与云平台                      | 推理高效、多模态、降低使用门槛，提升开发效率   | 面向开发者；多款开源模型（如 Qwen2、GLM4、Yi1.5）API **永久免费**，适合练手与开发 |
| **Ollama**                  | 本地部署，集成多款开源模型，数据不出本机 | 隐私与自主可控，不依赖外网 API                 | 需本机具备一定显卡/算力，适合有本地环境的同学                                     |

**官方 Quickstart 中的两套 API**：在官方快速入门里，LangGraph 提供了两种写智能体的方式——**Graph API**（用「节点 + 边」组成一张图来定义）和 **Functional API**（用单个函数来定义）。本教程**先采用上面这一套：Graph API**，即「把智能体定义成节点和边构成的图」的方式，便于理解状态流转、分支与循环；Functional API 适合更简单、单函数式的场景，可在掌握图结构后再查阅官方文档学习。

![Quickstart 关键步骤示意：官方提供 Graph API 与 Functional API 两种方式，图中红框突出 Graph API（本教程采用）](images/22/22-2-1-1.jpeg)

### 2.2 LangGraph 的技术架构

下图从「预构建 Agent API」「Agent Node API」「构建图语法 & API」三个层次概括了 LangGraph 的核心功能划分，便于建立整体印象。入门阶段重点掌握「构建图语法 & API」中的 State、节点、边、编译与执行即可。

![LangGraph 核心功能划分：预构建 Agent API、Agent Node API、构建图语法与 API](images/22/22-2-2-1.jpeg)

### 2.3 图的构建流程与可视化

**图的构建流程（五步）：**

1. 初始化 `StateGraph`（指定状态类型）。
2. 添加节点（`add_node`）。
3. 定义边，连接节点并指定入口/出口（`add_edge`，含 `START`、`END` 常量）。
4. 编译图（`graph.compile()`）。
5. 执行工作流（`app.invoke(initial_state)`）。

**可视化方式：** LangGraph 提供多种图表可视化方式。通过 `app.get_graph()` 可获取图的结构信息，进而：

- 生成 **ASCII** 文本图；
- 生成 **Mermaid** 代码，可复制到 [ProcessOn Mermaid 编辑器](https://www.processon.com/mermaid) 或其它支持 Mermaid 的工具中查看；
- 生成 **PNG** 并写入文件（依赖 mermaid.ink 或本地渲染，有时需重试或使用 Pyppeteer 等方式）。

示例代码（仅示意，完整可运行见下方案例源码）：

```python
# 1. 打印图的 ASCII 可视化结构
print(app.get_graph().print_ascii())
print("=" * 50)

# 2. 打印 Mermaid 代码，可复制到 ProcessOn 等编辑器查看
print(app.get_graph().draw_mermaid())
print("=" * 50)

# 3. 生成 PNG 并写入文件（不太稳定，可能需重试或本地渲染）
png_bytes = app.get_graph().draw_mermaid_png(max_retries=2, retry_delay=2.0)
output_path = "langgraph_" + str(uuid.uuid4())[:8] + ".png"
with open(output_path, "wb") as f:
    f.write(png_bytes)
print(f"图片已生成：{output_path}")
```

> **说明**：第 3 种方式依赖外部服务或本地环境，可能不稳定。若报错可参考错误信息中的 `max_retries`、`retry_delay` 或 `draw_method=MermaidDrawMethod.PYPPETEER` 等建议。

### 2.4 案例一：最简 HelloWorld

下面案例完整走通 2.3 节的五步流程，并打印 ASCII / Mermaid，是理解四要素的最佳起点。

【案例源码】`案例与源码-3-LangGraph框架/01-helloworld/LangGraphHello.py`

[LangGraphHello.py](案例与源码-3-LangGraph框架/01-helloworld/LangGraphHello.py ":include :type=code")

**下图说明**：这是将 LangGraph 的 `draw_mermaid()` 输出粘贴到 **Mermaid 在线编辑器**后的效果。左侧为 Mermaid 源码，右侧为渲染出的流程图。

- **左侧代码结构**：`config` 中 `flowchart.curve: linear` 表示连线为直线；`graph TD` 表示自上而下布局。节点依次为 `__start__`（椭圆、透明）、`greeting`、`add_emoji`（圆角矩形）、`__end__`（椭圆、紫色填充）；箭头 `-->` 表示执行顺序：start → greeting → add_emoji → end。`classDef` 为节点样式（如 default 浅紫、first 透明、last 深紫），便于区分起止与普通节点。
- **与案例的对应关系**：运行 LangGraphHello 后执行 `print(app.get_graph().draw_mermaid())` 会得到类似代码；可复制到 [Mermaid 在线编辑器](https://mermaid.live) 或 ProcessOn 的 Mermaid 编辑器中查看、导出为 PNG/SVG 或做二次编辑。

![Mermaid 在线编辑器：左侧为 draw_mermaid() 生成的代码，右侧为渲染的流程图（__start__ → greeting → add_emoji → __end__）](images/22/22-2-4-1.png)

### 2.5 案例二：加一点业务

在不接入大模型的前提下，用自定义的加法和减法函数作为节点，构建一个简单的「输入 → 加法 → 减法 → 输出」图，体会状态在节点间的传递。

【案例源码】`案例与源码-3-LangGraph框架/01-helloworld/LangGraphBiz.py`

[LangGraphBiz.py](案例与源码-3-LangGraph框架/01-helloworld/LangGraphBiz.py ":include :type=code")

### 2.6 案例三：接入大模型

在图中接入 LangChain 的聊天模型，实现「用户消息 → 模型节点 → 回复」的最小对话流。状态中的 `messages` 使用 LangGraph 内置规约器 **`add_messages`**：表示对该字段「追加」而非「覆盖」，节点只返回新增消息，框架会自动合并到列表末尾，适合多轮对话。

【案例源码】`案例与源码-3-LangGraph框架/01-helloworld/LangGraphLLM.py`

[LangGraphLLM.py](案例与源码-3-LangGraph框架/01-helloworld/LangGraphLLM.py ":include :type=code")

---

**本章小结**：LangGraph = LangChain + 图编排 + 状态机，四要素为 State、Nodes、Edges、Graph；构建流程见 2.3 节五步。三个案例（LangGraphHello、LangGraphBiz、LangGraphLLM）位于 `案例与源码-3-LangGraph框架/01-helloworld`，可视化可用 `print_ascii()`、`draw_mermaid()`。

**建议下一步**：本地运行 `LangGraphHello.py`、`LangGraphBiz.py`，尝试改状态或增删节点/边；接着学 [第 23 章 - LangGraph API：图与状态](23-LangGraphAPI：图与状态.md)，深入图的 API、State 的 Schema/Reducer 及 TypedDict 与 Pydantic 选型；需条件分支、循环或多智能体时再学后续进阶章节。
