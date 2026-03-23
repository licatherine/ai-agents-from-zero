# 23 - LangGraphAPI：图与状态

---

**本章课程目标：**

- 理解 Graph API 中「图」的正式定义与有向图在工作流中的作用，会构建多节点、固定边的完整图并运行案例。
- 理解 **State** 的组成（**Schema + Reducer**）、在节点间的传递方式，以及 **state_schema / input_schema / output_schema** 三要素的含义与用法；掌握 **Reducer（规约函数）** 的常见类型（默认覆盖、add_messages、operator.add/mul、自定义），能根据业务选择合并策略。
- 会使用 **TypedDict** 或 **Pydantic BaseModel** 定义 State，能根据场景选型；会通过 `input_schema`、`output_schema` 限制图的输入输出接口。

**前置知识建议：** 已学习 [第 22 章 LangGraph 概述与快速入门](22-LangGraph概述与快速入门.md)，掌握 LangGraph 四要素（State、Nodes、Edges、Graph）、图的六步构建流程，并至少跑通 HelloWorld 或 LangGraphBiz 案例。

**学习建议：** 先通读「什么是图」「什么是状态」建立概念，再按顺序跑通 BuildWholeGraphSummary、DefState、Reducer 案例与 StateSchema。**Node（节点）、Edge（边）与 Send/Command/Runtime 等高级控制**见 [第 24 章 LangGraph Graph API 之 Node、Edge 与高级控制](24-LangGraphGraphAPI-Node与Edge与高级控制.md)。

---

## 1、Graph API 之 Graph（图）

### 1.1 简介

**知识出处：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#graphs （概念通用，Python 版可查对应文档）

**图**是一种由**节点**和**边**组成的、用于描述节点之间关系的数据结构，分为无向图和有向图；**有向图**的边带有方向。LangGraph 通过**有向图**定义 AI 工作流中的执行步骤与执行顺序，从而实现**复杂、有状态、可循环**的应用程序逻辑。

![有向图在工作流中的含义示意](images/23/23-1-1-1.jpeg)

**图意说明：** 上图为有向图在工作流中的典型含义。左侧 **Start** 为图的入口，右侧 **End** 为图的出口；中间 **node 1 ～ node 5** 为工作流中的处理步骤（如调用 LLM、工具或业务逻辑）。**有向边**（edge 1 ～ edge 8）表示执行与数据流动的方向：从 Start 可经多条边进入不同节点（如 node 1、node 2、node 3），体现**分支或并行**；中间节点之间按边依次执行（如 node 2→node 4、node 3→node 5）；多条路径最终经边汇聚到 End。因此，有向图既能表达「步骤与顺序」，也能表达「分支、汇聚与多路径」，对应 LangGraph 中复杂、有状态、可循环的工作流结构。

### 1.2 构建一个完整的图

**图的构建流程**（与第 22 章 2.4 一致）：初始化 StateGraph → 加节点 → 定义边 → 设置 START/END → 编译 → 执行。

下面案例用 `input → process → output` 三个节点和固定边，演示状态（`process_data`）在节点间的传递与打印，便于对照「流程」与「状态」的关系。

【案例源码】`案例与源码-3-LangGraph框架/02-graph/BuildWholeGraphSummary.py`

[BuildWholeGraphSummary.py](案例与源码-3-LangGraph框架/02-graph/BuildWholeGraphSummary.py ":include :type=code")

---

## 2、Graph API 之 State（状态）

### 2.1 简介

**知识出处：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#state

在 LangGraph 中，**State** 是贯穿整个工作流执行过程的**共享数据结构**，代表当前「快照」。

它存储从工作流开始到结束所需的信息（如历史对话、检索到的文档、工具执行结果等），在**各节点间共享**，且每个节点都可以按规则对其更新。

State 由两部分组成：**图的 Schema（模式）** 和 **规约函数（Reducer）**。

为免与「示意图」混淆：**这里的「图」**就是第 1 节所讲的那张**工作流有向图**（节点、边、`START`/`END`，代码里常见为 `StateGraph(...)`）。**「图的 Schema」**指与这张图**绑在一起**的状态类型——也就是你传给 `StateGraph` 的那个类型（如 `StateGraph(MyState)` 里的 `MyState`），它规定运行时**共享状态**有哪些字段、各是什么类型。

Schema 描述状态有哪些字段、什么类型；Reducer 规定节点产生的**更新**如何合并到当前状态（例如追加列表、替换字段等）。**定义图时，首先要做的就是定义图的 State**，即把这两部分设计好。

可以这样理解 State 的角色：

- **State 是图的「记忆」与「血液」**：像记忆一样保存当前信息，像血液一样在节点之间流动，是工作流运转的核心。
- **State 是单一事实来源（Single Source of Truth）**：所有节点都从同一份状态读、往同一份状态写，避免数据各说各话、不一致。
- **所有数据都通过 State 在节点间传递和更新**：没有单独的数据线，状态就是数据流动的载体，是 LangGraph 的核心。
- **Reducer 定义状态如何被安全、原子化地更新**：多个节点可能同时改状态，Reducer 规定「怎么合并这些改动」，保证更新过程可控、不丢数据、不冲突。

![State 的组成：Schema + Reducer；所有节点向 State 发出更新，由 Reducer 应用](images/23/23-2-1-1.jpeg)

### 2.2 基本 State 定义示例

下面用 **TypedDict** 定义一个简单状态，并构建一张「无中间节点、直接从 START 到 END」的图，用于验证 `invoke(initial_state)` 的用法。注意：`invoke()` 只接收一个核心位置参数（状态字典），不要传入多个独立参数。

【案例源码】`案例与源码-3-LangGraph框架/03-state/DefState.py`

[DefState.py](案例与源码-3-LangGraph框架/03-state/DefState.py ":include :type=code")

### 2.3 State 的类型：TypedDict 与 Pydantic BaseModel

State 可以是 **TypedDict**，也可以是 **Pydantic 的 BaseModel**。下表对比两者，便于选型；在 LangGraph 中通常**推荐使用 TypedDict** 作为 State 类型，简单轻量、无额外运行时校验开销。

| 对比项                 | TypedDict                                | BaseModel（Pydantic）                                               |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| **来源**               | Python 标准库（`typing` 模块）           | 需安装 Pydantic 库                                                  |
| **定位**               | 轻量级「带类型的字典」，主要做类型提示   | 完整的数据模型，自带校验、解析等能力                                |
| **运行时检查**         | 无：不校验数据是否合法，只做类型标注     | 有：自动校验类型、默认值、约束等，不合规会报错                      |
| **性能**               | 快：几乎无额外开销，就是普通字典         | 稍慢：创建和更新时会做解析与校验，有额外成本                        |
| **序列化**             | 需自己转成 dict/JSON 或从 dict/JSON 读入 | 自带 `.model_dump()`、`.model_dump_json()` 等，方便与 JSON/API 对接 |
| **适用场景**           | 结构简单、以「传参/状态容器」为主        | 需要严格校验、默认值、嵌套结构或字段说明时                          |
| **LangGraph 中的用法** | 推荐作为 State 类型，写起来简单          | 可用，但官方更推荐 TypedDict；若用 Pydantic 需注意与图的兼容与转换  |

**一句话选型：**

- 想要**轻量、无运行时开销、习惯字典写法** → 用 **TypedDict**。
- 想要**自动校验、默认值、嵌套结构、字段描述** → 用 **pydantic.BaseModel**。

两种写法在 LangGraph 里都能参与图的编译，只需按规则声明字段即可。

### 2.4 State 的组成：Schema（模式）

**官方介绍：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#schema

**一切之基石：State Schema。** State 是 LangGraph 应用的核心数据结构；它定义了应用中「有哪些数据」，这些数据会在所有节点之间传递，并且可以被持久化保存。

设计 State Schema 时，建议遵循以下最佳实践：

- **最小化**：只放必要的数据，能少则少，避免重复和多余字段，方便维护和排查问题。
- **清晰命名**：字段名要一看就懂、语义明确，例如用 `user_messages` 而不是 `msg`，用 `retrieved_docs` 而不是 `docs`（若容易混淆）。
- **类型安全**：用 **TypedDict**、**Annotated** 等声明类型，让 IDE 和运行时都能做类型检查，减少写错字段或传错类型。
- **分层设计**：状态如果很复杂，不要全摊平在一层；可以按业务拆成嵌套结构（例如「用户信息」「会话上下文」各成一块），结构清晰、后续扩展也方便。

**构成三要素：**

| 概念              | 含义                                         | 特点                                                               |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| **state_schema**  | 图的完整内部状态，包含所有节点可能读写的字段 | 必须指定，不能为空；是图的「全局状态空间」，所有节点都可访问和写入 |
| **input_schema**  | 图接受什么输入，是 state_schema 的子集       | 可选；不指定时默认等于 state_schema；用于限制图的输入接口          |
| **output_schema** | 图返回什么输出，是 state_schema 的子集       | 可选；不指定时默认等于 state_schema；用于限制图的输出接口          |

`graph.invoke(input_data)` 的典型流程如下（按执行顺序）：

1. **外部调用**：调用方执行 `graph.invoke(input_data)`，把输入数据交给图。
2. **input_schema 验证/过滤**：图先用 **input_schema** 对传入数据做校验和过滤，只保留 schema 里允许的字段，不符合的会被拒之门外或忽略，保证「进图」的数据格式正确。
3. **START（虚拟节点）**：通过校验后进入图的内部流程，从 **START** 这个虚拟起点开始（没有实际业务逻辑，只表示「开始」）。
4. **普通节点 1 → 2 → … → N**：按边的定义依次执行各个普通节点；每个节点拿到的都是**完整的 state_schema**（即图的内部状态），读、算、写都针对这份状态。
5. **END（虚拟节点）**：所有节点跑完后到达 **END** 这个虚拟终点，表示图内执行结束。
6. **output_schema 过滤输出**：在把结果还给调用方之前，用 **output_schema** 对最终状态做过滤，只保留 schema 里声明要输出的字段，实现「对外只暴露约定好的字段」。
7. **返回结果**：过滤后的数据作为返回值交给调用者。

全程中，**input_schema** 管「进来什么」、**state_schema** 管「图内部用什么」、**output_schema** 管「出去什么」，三者各司其职。

### 2.5 State 的组成：Reducer（规约函数）

前面 2.1 已说明：State 由 **Schema** 与 **Reducer** 组成；Schema 描述状态有哪些字段与类型，**Reducer 规定节点产生的更新如何合并到当前状态**。本节展开 Reducer 的用法与常见类型，与 Schema 一起构成「State 组成」的完整图景。

#### 2.5.1 什么是 Reducer

**知识出处：** https://docs.langchain.com/oss/javascript/langgraph/graph-api#reducers

在 LangGraph 中，**Reducer（规约函数）** 决定**节点产生的更新如何作用到 State**。State 中的每个字段都可以拥有自己的规约函数；若未显式指定，则**默认对该字段的更新为覆盖**——后执行节点返回的值会直接覆盖先执行节点的值。

- **状态合并策略**：State 在工作流中贯穿所有节点、共享数据；每个节点可读取并更新 State。Reducer 定义多个节点之间对同一字段的更新方式（覆盖、合并、追加等）。
- **Reducer 的作用**：控制状态更新方式；处理并行更新时的数据一致性；支持覆盖、追加、相加等策略；支持自定义合并逻辑，便于构建复杂工作流与并行执行场景。

![Reducer 决定节点更新如何作用到 State；有向图与状态流转示意](images/23/23-2-5-1.jpeg)

#### 2.5.2 default：未指定 Reducer 时使用覆盖更新

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_Default.py`

[StateReducer_Default.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_Default.py ":include :type=code")

#### 2.5.3 add_messages：用于消息列表追加

对话场景中需要将多轮消息**追加**到列表。使用 `Annotated[List, add_messages]`，节点只返回增量消息，由 `add_messages` 规约器自动追加。

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_AddMessages.py`

[StateReducer_AddMessages.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_AddMessages.py ":include :type=code")

#### 2.5.4 operator.add：列表/字符串/数值追加

`operator.add` 作为 Reducer 时，对列表做 extend、对字符串做连接、对数值做累加。

**列表追加：**

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd.py`

[StateReducer_OperatorAdd.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd.py ":include :type=code")

**字符串连接：**

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd2.py`

[StateReducer_OperatorAdd2.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd2.py ":include :type=code")

**数值累加：**

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd3.py`

[StateReducer_OperatorAdd3.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorAdd3.py ":include :type=code")

#### 2.5.5 operator.mul：数值相乘

使用 `operator.mul` 时要注意：LangGraph 会用类型默认值（如 float 的 0.0）先做一次规约，导致 `0.0 * 初始值 = 0`，后续乘法始终为 0。**建议对乘法使用自定义 Reducer**，将「第一次」的 current 视为单位元 1.0。

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorMul.py`

[StateReducer_OperatorMul.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_OperatorMul.py ":include :type=code")

#### 2.5.6 自定义 Reducer

当内置 Reducer 不满足需求时（如 operator.mul 的初始值问题），可自定义规约函数：签名为 `(current, update) -> new_value`，在函数内处理「第一次」等边界。

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_Custom.py`

[StateReducer_Custom.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducer_Custom.py ":include :type=code")

#### 2.5.7 多种策略并存

同一 State 中可对不同字段使用不同 Reducer（如 messages 用 add_messages，tags 用 operator.add，score 用 operator.add）。下面案例供自行阅读与运行。

【案例源码】`案例与源码-3-LangGraph框架/03-state/reducers/StateReducersMyChatBot.py`

[StateReducersMyChatBot.py](案例与源码-3-LangGraph框架/03-state/reducers/StateReducersMyChatBot.py ":include :type=code")

### 2.6 输入输出 Schema 示例

下面案例演示如何通过 `input_schema` 和 `output_schema` 限制图的输入与输出类型，实现「调用时只传 question，返回时只拿 answer」的接口，适合需要明确对外契约的场景。

【案例源码】`案例与源码-3-LangGraph框架/03-state/schema/StateSchema.py`

[StateSchema.py](案例与源码-3-LangGraph框架/03-state/schema/StateSchema.py ":include :type=code")

---

**本章小结：**

- **Graph API（图）**：图由有向的**节点**和**边**组成，定义工作流的执行步骤与顺序；构建流程与第 22 章一致；BuildWholeGraphSummary 案例见 `02-graph/`。
- **State**：由 **Schema** 与 **Reducer** 组成，是节点间共享的「单一事实来源」。**Schema**：state_schema 为完整内部状态，input_schema / output_schema 为可选的输入输出子集。**Reducer**：规定节点更新如何合并（默认覆盖；add_messages 消息追加；operator.add 列表/字符串/数值；operator.mul 需注意默认 0；自定义 Reducer；多策略并存）。DefState、StateSchema、Reducer 案例见 `03-state/`。
- **State 类型**：常用 TypedDict（轻量），也可用 pydantic.BaseModel（校验、嵌套更强）。

**建议下一步：** 在本地运行 BuildWholeGraphSummary、DefState、Reducer 案例与 StateSchema，并尝试修改 State 字段或 Reducer 类型；接着学习 [第 24 章 LangGraph Graph API 之 Node、Edge 与高级控制](24-LangGraphGraphAPI-Node与Edge与高级控制.md)，掌握节点、边与 Send/Command/Runtime。
