# 22 - LangGraph 概述与快速入门

---

**本章课程目标：**

- 理解 LangGraph 是什么、与 LangChain / Agent 的关系，以及为何需要「图」来编排复杂工作流。
- 掌握 LangGraph 四大核心概念：**State（状态）、Nodes（节点）、Edges（边）、Graph（图）**，能独立完成从定义状态、加节点、连边到编译运行的完整流程。
- 会使用 HelloWorld 与简单业务图案例，并了解图的多种可视化方式（ASCII、Mermaid、PNG）。

**前置知识建议：** 已学习 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)，了解 LangChain 的链式思维与模型调用方式；若已学 [第 21 章 Agent 智能体](21-Agent智能体.md)，可更好理解「可控图」与「黑箱 Agent」的对比。

**学习建议：** 先建立「为什么需要 LangGraph」的直觉（Chain 太线性、Agent 太黑箱），再按「State → Nodes → Edges → 编译 → 运行」的顺序动手写 HelloWorld；图的构建流程与可视化在本章完成，**Graph API 中「图」与「状态」的深入讲解见 [第 23 章](23-LangGraphGraphAPI与State.md)**。

---

## 1、LangGraph 是什么

### 1.1 简介与定位

**一句话定义：** LangGraph 是基于 LangChain 构建的、面向**智能体多轮交互 / 状态持久化 / 分支与并行执行**的**图结构工作流框架**。可以记作：**LangGraph = LangChain + 图编排 + 状态机**。

**官方文档：**

- 英文：https://docs.langchain.com/oss/python/langgraph/overview
- 中文：https://docs.langchain.org.cn/oss/python/langgraph/overview

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

![基于 LLM 的智能体工作流示意：开始 → 大模型节点 → 边节点（下一步动作）→ 结束或执行 Action 并回到大模型](images/22/image4.jpeg)

若用 LangChain 的 **Chain** 来模拟上述过程，会非常痛苦：Chain 天生是单线执行，很难实现「返回上一步」或「根据条件跳转到某一步」这种灵活控制流，开发者往往要写大量不优雅的胶水代码，逻辑容易变成一团乱麻。

### 1.3 那用 Agent 不行吗？

LangChain 的 **Agent** 在某种程度上解决了「循环与决策」的问题。Agent 像一名有自主决策能力的「将军」，基于 ReAct（Reason + Act）框架，可以自己决定调用什么工具、何时再搜一次等，确实能实现循环。

但 Agent 的最大问题在于它是一个**黑箱**：你给它目标和工具，它就开始「自言自语」（Reasoning）和「手忙脚乱」（Acting），你作为开发者很难对工作流程做**精细化控制和干预**。例如：

1. **无法强制流程**：你不能要求它「必须先写草稿再批判」，它可能搜完觉得够了就直接给最终答案。
2. **难以纠错**：一旦在错误思路上循环十几次，会浪费大量时间和 API Token，最后可能仍给出错误结论。
3. **行为不稳定**：同样的问题，这次可能是 A→B→C，下次可能是 A→C→B，结果难以预测。

对于需要**可靠、可控、可预测**的商业级 AI 应用来说，这种黑箱智能体就像「能力很强但野性难驯」的员工，真正核心的任务不敢完全交给它。因此需要：**疑人要用，用人要疑——监控过程，核实结果**。

下面两图分别从**人机协作（HITL）**和**多智能体协作**的角度，补充了智能体在实际应用中的需求，这些恰恰是 LangGraph 可以更好支持的场景。

![人机协作（HITL）：课程内容经 AI 分析师建议 → AI 改写器执行修改，不确定时交由人类专家审核决策再回到改写器，最终输出](images/22/image5.jpeg)

![多智能体协作：分层规划与共创协作两种模式，模拟现实团队工作方式](images/22/image6.gif)

### 1.4 Before / After 小结

**Before（LangChain 的困境）：**

- **Chain** 太「流水线」，无法优雅地处理循环和条件分支，不适合复杂多步任务。
- **Agent** 太自由，像黑箱，难以控制、调试和保证稳定性。

**After（LangGraph 带来的改变）：**

- **状态管理**：在不同节点之间传递和维护信息，支持长期记忆与多轮对话。
- **精确控制**：通过定义**节点和边**，可以精确控制执行逻辑，包括条件分支、循环和并行执行。
- **工具集成**：无缝集成搜索引擎、数据库、API 等外部工具，扩展 LLM 能力边界。
- **可观测性**：图结构使运行路径清晰可见，便于理解决策过程并快速定位、调试问题。
- **模块化与可复用**：每个节点可以是独立、可复用的组件；通过子图机制，复杂工作流可拆成多个可独立开发与测试的模块。

**应用场景决策**：何时选择 LangGraph？下图给出了简明指南——适合多步复杂推理、长时间运行任务、需要人在环（HITL）以及追求生产级稳定性的场景；简单一次性查询建议用 LangChain Chains，快速原型验证可用更高层工具。

![应用场景决策：何时选择 LangGraph（适用 vs 不适用场景对比）](images/22/image7.jpeg)

### 1.5 一句话小结

LangGraph 是 LangChain 生态中专门用于构建**基于 LLM 的复杂、有状态、多智能体应用**的框架。核心思想是把应用的工作流程抽象为**有向图**：用**节点**表示执行步骤，用**边**表示逻辑流，从而支持条件分支、循环、并行等复杂控制流，并实现状态持久化、断点续跑、时间旅行、人机协作等高级能力。它把基础单元从「链」升级为「图」。

![LangGraph 彻底打破「链」的束缚，引入「图」的结构，让复杂 AI 应用从一条直线变成一张网](images/22/image8.gif)

### 1.6 能干嘛、去哪下、怎么玩

- **能干嘛**：彻底打破「链」的束缚，引入「图」的结构，让构建复杂 AI 应用从「一条直线」变成「一张网」。
- **去哪下（安装）**：https://docs.langchain.com/oss/python/langgraph/install
- **怎么玩**：记住四个词即可掌握 LangGraph 的灵魂：**State（状态）、Nodes（节点）、Edges（边）、Graph（图）**。

![构建基石：State 状态、Nodes 节点、Edges 边；State 为中心，Nodes 执行逻辑，Edges 定义顺序](images/22/image9.jpeg)

---

## 2、HelloWorld 快速入门

### 2.1 LangGraph 与 LangChain 的关系

LangGraph **基于** LangChain 构建。无论图结构多复杂，**单个任务执行链路仍然是线性的**，背后仍依赖 LangChain 的 Chain 等组件。可以这样理解：**LangGraph 是 LangChain 工作流的「高级编排工具」**——「高级」之处在于能按**图结构**来编排工作流，而不再局限于单链。

### 2.2 前置约定

动手前建议约定好**模型与配置**（API Key、模型名、Base URL），与 [第 10 章](10-LangChain快速上手与HelloWorld.md) 的「调用三件套」一致。下表示意课程中常用的模型与平台约定；其他模型可参考官方文档或本仓库其他章节。

![动手前模型约定：选用的大模型与平台](images/22/image11.jpeg)

**安装与快速开始：**

- 安装 LangGraph：参见官方 Install 文档，或在本项目中随 LangChain 一起安装。
- 配置门道与关键点（环境变量、Base URL 等）与第 10 章一致。

![安装 LangGraph 步骤示意](images/22/image14.gif)

![Quickstart 关键步骤示意](images/22/image15.jpeg)

### 2.3 LangGraph 的技术架构

下图从「预构建 Agent API」「Agent Node API」「构建图语法 & API」三个层次概括了 LangGraph 的核心功能划分，便于建立整体印象。入门阶段重点掌握「构建图语法 & API」中的 State、节点、边、编译与执行即可。

![LangGraph 核心功能划分：预构建 Agent API、Agent Node API、构建图语法与 API](images/22/image16.jpeg)

### 2.4 图的构建流程与可视化

**图的构建流程（六步）：**

1. 初始化一个 `StateGraph` 实例（并指定状态类型）。
2. 添加节点（`add_node`）。
3. 定义边，把所有节点连接起来（`add_edge`，含 `START` / `END`）。
4. 设置特殊节点入口和出口（可选，通常用 `START`、`END` 常量）。
5. 编译图（`graph.compile()`）。
6. 执行工作流（`app.invoke(initial_state)`）。

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

# 3. 生成 PNG 并写入文件（可能需重试或本地渲染）
# png_bytes = app.get_graph().draw_mermaid_png(max_retries=2, retry_delay=2.0)
# output_path = "langgraph_" + str(uuid.uuid4())[:8] + ".png"
# with open(output_path, "wb") as f:
#     f.write(png_bytes)
# print(f"图片已生成：{output_path}")
```

> **说明**：第 3 种方式依赖外部服务或本地环境，可能不稳定。若报错可参考错误信息中的 `max_retries`、`retry_delay` 或 `draw_method=MermaidDrawMethod.PYPPETEER` 等建议。

### 2.5 案例一：最简 HelloWorld（LangGraphHello.py）

下面案例完成「定义 State → 定义节点 → 构建图 → 加边 → 编译 → 运行」，并打印 ASCII / Mermaid，是理解「四要素」的最佳起点。

【案例源码】`案例与源码-5-LangGraph/01-helloworld/LangGraphHello.py`

[LangGraphHello.py](案例与源码-5-LangGraph/01-helloworld/LangGraphHello.py ":include :type=code")

### 2.6 案例二：加一点业务（LangGraphBiz.py）

在不接入大模型的前提下，用自定义的加法和减法函数作为节点，构建一个简单的「输入 → 加法 → 减法 → 输出」图，体会状态在节点间的传递。

【案例源码】`案例与源码-5-LangGraph/01-helloworld/LangGraphBiz.py`

[LangGraphBiz.py](案例与源码-5-LangGraph/01-helloworld/LangGraphBiz.py ":include :type=code")

### 2.7 案例三：接入大模型（LangGraphLLM.py）

在图中接入 LangChain 的聊天模型，实现「用户消息 → 模型节点 → 回复」的最小对话流，并演示带 `messages` 状态与 `add_messages` 的用法。

【案例源码】`案例与源码-5-LangGraph/01-helloworld/LangGraphLLM.py`

[LangGraphLLM.py](案例与源码-5-LangGraph/01-helloworld/LangGraphLLM.py ":include :type=code")

### 2.8 本节小结

- **LangGraph 的灵魂**：**State（状态）+ Nodes（节点）+ Edges（边）+ Graph（图）**。
- **图的构建流程**：初始化 StateGraph → 加节点 → 定义边（含 START/END）→ 编译 → 执行。
- 可视化可用 `print_ascii()`、`draw_mermaid()`，必要时再使用 `draw_mermaid_png()`。

---

**本章小结：**

- **LangGraph 是什么**：基于 LangChain 的图结构工作流框架，支持多轮交互、状态持久化、分支与循环；**LangGraph = LangChain + 图编排 + 状态机**。Chain 太线性、Agent 太黑箱，LangGraph 用「图」在可控性与表达能力之间取得平衡。
- **四要素**：**State（状态）、Nodes（节点）、Edges（边）、Graph（图）**；图的构建流程：初始化 StateGraph → 加节点 → 定义边（含 START/END）→ 编译 → `invoke(initial_state)` 执行。
- **HelloWorld 与案例**：LangGraphHello（最简问候）、LangGraphBiz（加减法业务）、LangGraphLLM（接入大模型）均位于 `案例与源码-5-LangGraph/01-helloworld`，可按顺序运行体会；可视化可用 `print_ascii()`、`draw_mermaid()`。

**建议下一步：** 在本地运行 `LangGraphHello.py` 和 `LangGraphBiz.py`，再尝试修改状态字段或增加节点、边；接着学习 [第 23 章 - LangGraph Graph API 与 State](23-LangGraphGraphAPI与State.md)，深入理解「图」的 API、State 的 Schema/Reducer、input_schema / output_schema 及 TypedDict 与 Pydantic 选型；若需条件分支、循环或多智能体，可继续学习后续 LangGraph 进阶章节。
