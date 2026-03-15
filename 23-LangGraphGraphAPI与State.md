# 23 - LangGraph Graph API 与 State

---

**本章课程目标：**

- 理解 Graph API 中「图」的正式定义与有向图在工作流中的作用，会构建多节点、固定边的完整图并运行案例。
- 理解 **State** 的组成（Schema + Reducer）、在节点间的传递方式，以及 **state_schema / input_schema / output_schema** 三要素的含义与用法。
- 会使用 **TypedDict** 或 **Pydantic BaseModel** 定义 State，能根据场景选型；会通过 `input_schema`、`output_schema` 限制图的输入输出接口。

**前置知识建议：** 已学习 [第 22 章 LangGraph 概述与快速入门](22-LangGraph概述与快速入门.md)，掌握 LangGraph 四要素（State、Nodes、Edges、Graph）、图的六步构建流程，并至少跑通 HelloWorld 或 LangGraphBiz 案例。

**学习建议：** 先通读「什么是图」「什么是 State」建立概念，再按顺序运行 BuildWholeGraphSummary、DefState、StateSchema 三个案例；Schema 三要素与 TypedDict/BaseModel 对比可作查阅参考。

---

## 1、Graph API 之 Graph（图）

### 1.1 什么是图

**知识出处：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#graphs（概念通用，Python 版可查对应文档）

**图**是一种由**节点**和**边**组成的、用于描述节点之间关系的数据结构，分为无向图和有向图；**有向图**的边带有方向。LangGraph 通过**有向图**定义 AI 工作流中的执行步骤与执行顺序，从而实现**复杂、有状态、可循环**的应用程序逻辑。

![图的定义：节点与边、有向图；LangGraph 用有向图定义执行步骤与顺序](images/22/image19.jpeg)

![有向图在工作流中的含义示意](images/22/image20.jpeg)

### 1.2 构建一个完整的图

**图的构建流程**（与第 22 章 2.4 一致）：初始化 StateGraph → 加节点 → 定义边 → 设置 START/END → 编译 → 执行。

下面案例用 `input → process → output` 三个节点和固定边，演示状态（`process_data`）在节点间的传递与打印，便于对照「流程」与「状态」的关系。

【案例源码】`案例与源码-5-LangGraph/02-graph/BuildWholeGraphSummary.py`

[BuildWholeGraphSummary.py](案例与源码-5-LangGraph/02-graph/BuildWholeGraphSummary.py ":include :type=code")

---

## 2、Graph API 之 State（状态）

### 2.1 什么是 State

**知识出处：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#state

在 LangGraph 中，**State** 是贯穿整个工作流执行过程的**共享数据结构**，代表当前「快照」。它存储从工作流开始到结束所需的信息（如历史对话、检索到的文档、工具执行结果等），在**各节点间共享**，且每个节点都可以按规则对其更新。State 包含两部分：一是**图的模式（Schema）**，二是**规约函数（Reducer functions）**——后者规定如何把节点产生的**更新**应用到状态上（例如追加列表、替换字段等）。

![State：图的记忆与血液；单一事实来源；所有数据通过 State 在节点间传递和更新；Reducer 定义状态如何被安全、原子化地更新](images/22/image22.gif)

**定义图时，首先要做的就是定义图的 State。** State 由**图的 Schema** 和 **Reducer 函数**组成：Schema 描述状态有哪些字段、类型；Reducer 指定当节点返回部分更新时，如何合并到当前状态。

![State 的组成：Schema + Reducer；所有节点向 State 发出更新，由 Reducer 应用](images/22/image23.jpeg)

### 2.2 基本 State 定义示例（DefState.py）

下面用 **TypedDict** 定义一个简单状态，并构建一张「无中间节点、直接从 START 到 END」的图，用于验证 `invoke(initial_state)` 的用法。注意：`invoke()` 只接收一个核心位置参数（状态字典），不要传入多个独立参数。

【案例源码】`案例与源码-5-LangGraph/03-state/DefState.py`

[DefState.py](案例与源码-5-LangGraph/03-state/DefState.py ":include :type=code")

### 2.3 State 的组成：Schema 与三要素

**知识出处（Schema）：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#schema

**一切之基石：State Schema。** State 是 LangGraph 应用的核心数据结构，它定义了应用中所有在节点间传递并被持久化的数据。设计时建议：最小化字段、清晰命名、类型安全（如 TypedDict / Annotated）、复杂状态分层嵌套。

![State Schema 核心定义与最佳实践：最小化、清晰命名、类型安全、分层设计](images/22/image25.gif)

**构成三要素：**

| 概念              | 含义                                         | 特点                                                               |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| **state_schema**  | 图的完整内部状态，包含所有节点可能读写的字段 | 必须指定，不能为空；是图的「全局状态空间」，所有节点都可访问和写入 |
| **input_schema**  | 图接受什么输入，是 state_schema 的子集       | 可选；不指定时默认等于 state_schema；用于限制图的输入接口          |
| **output_schema** | 图返回什么输出，是 state_schema 的子集       | 可选；不指定时默认等于 state_schema；用于限制图的输出接口          |

下图展示了 `graph.invoke(input_data)` 的典型流程：先按 **input_schema** 验证/过滤输入，再经 START → 普通节点（均接收 state_schema）→ END，最后按 **output_schema** 过滤输出并返回给调用者。

![graph.invoke 调用流程：input_schema 验证 → START → 普通节点（state_schema）→ END → output_schema 过滤 → 返回](images/22/image26.jpeg)

### 2.4 State 的类型：TypedDict 与 Pydantic BaseModel

State 可以是 **TypedDict**，也可以是 **Pydantic 的 BaseModel**。下表对比两者，便于选型；在 LangGraph 中通常**推荐使用 TypedDict** 作为 State 类型，简单轻量、无额外运行时校验开销。

![TypedDict 与 Pydantic BaseModel 对比：来源、定位、运行时检查、性能、序列化、用途及 LangGraph 支持](images/22/image27.jpeg)

**一句话选型：**

- 想要**轻量、无运行时开销、习惯字典写法** → 用 **TypedDict**。
- 想要**自动校验、默认值、嵌套结构、字段描述** → 用 **pydantic.BaseModel**。

两种写法在 LangGraph 里都能参与图的编译，只需按规则声明字段即可。

### 2.5 输入输出 Schema 示例（StateSchema.py）

下面案例演示如何通过 `input_schema` 和 `output_schema` 限制图的输入与输出类型，实现「调用时只传 question，返回时只拿 answer」的接口，适合需要明确对外契约的场景。

【案例源码】`案例与源码-5-LangGraph/03-state/schema/StateSchema.py`

[StateSchema.py](案例与源码-5-LangGraph/03-state/schema/StateSchema.py ":include :type=code")

---

**本章小结：**

- **Graph API（图）**：图由有向的**节点**和**边**组成，定义工作流的执行步骤与顺序；构建流程与第 22 章一致，BuildWholeGraphSummary 案例演示了多节点、固定边下状态（`process_data`）的传递。
- **State**：由 **Schema** 与 **Reducer** 组成，是节点间共享的「单一事实来源」；**state_schema** 为图的完整内部状态，**input_schema** / **output_schema** 可选的子集，用于约束图的输入输出接口；`graph.invoke` 会先按 input_schema 过滤输入、再经节点处理、最后按 output_schema 过滤输出。
- **State 类型**：常用 **TypedDict**（轻量、推荐），也可用 **pydantic.BaseModel**（校验、嵌套、描述更强）；DefState 演示基本定义，StateSchema 演示 input/output schema 的用法。

**建议下一步：** 在本地运行 BuildWholeGraphSummary、DefState、StateSchema，并尝试修改 State 字段或增加 input_schema/output_schema；若需深入 **Reducer**（如消息列表的追加方式），可学习本仓库中 `案例与源码-5-LangGraph/03-state/reducers` 下的 Reducer 示例；若需条件分支、循环或多智能体，可继续学习后续 LangGraph 进阶章节（条件边、子图、Agent 节点等）。
