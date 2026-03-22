# 24 - LangGraphAPI：节点、边与进阶

---

**本章课程目标：**

- 掌握 **Node（节点）** 的概念、START/END、节点缓存（CachePolicy）与重试机制（RetryPolicy），会使用 `set_entry_point` / `set_finish_point`。
- 掌握 **Edge（边）** 的类型：普通边、条件边、入口点与条件入口点，能编写路由函数实现分支与动态入口。
- 了解 **Send**（Map-Reduce 并行）、**Command**（状态更新 + 路由）与 **Runtime 上下文**（context_schema）的适用场景与基本用法。

**前置知识建议：** 已学习 [第 22 章 LangGraph 概述与快速入门](22-LangGraph概述与快速入门.md)、[第 23 章 LangGraph Graph API 与 State](23-LangGraphGraphAPI与State.md)，掌握图、State（Schema + Reducer）、TypedDict/BaseModel 及图的构建流程。

**学习建议：** 按「Node → Edge → Send/Command/Runtime」顺序学习；先跑通 Node、Edge 案例建立执行结构，再学习 Send、Command、Runtime 等高级控制。案例源码在 `案例与源码-3-LangGraph框架` 下（04-node、05-edge、06-specialApi）。

---

## 1、Graph API 之 Node（节点）

### 1.1 简介

**知识出处**：[LangGraph 官方文档 - Graph API > Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api#nodes)

在 LangGraph 中，节点（Node）即 Python 函数（同步或异步），可接收 `state`、`config`（RunnableConfig）、`runtime`（Runtime 对象）等参数；通过 `add_node` 将节点加入图，未指定名称时默认使用函数名。

![节点可接收 state、config、runtime 等参数](images/24/24-1-1-1.jpeg)


**节点可接收的三种参数（通俗理解）：**

- **state**：图的当前状态。节点通过它读取或修改整条工作流的进度和数据（例如上一节点的输出、用户输入、中间结果等），是节点之间「传话」的载体。
- **config**：类型为 `RunnableConfig`，表示**本次运行的配置与元数据**。例如 `thread_id`（标识当前会话或线程，多轮对话时区分不同用户）、`tags`（打标签，便于日志与监控）、`metadata` 等。不参与业务状态，主要用于「谁在跑、怎么记、怎么追踪」。
- **runtime**：类型为 `Runtime`，表示**执行时的环境与工具**。可包含：**runtime context**（通过 `context_schema` 传入的配置，如模型名、API Key、数据库连接等，详见本章 3.4 节）、**store**（持久化或访问外部存储）、**stream_writer**（流式写出，用于边跑边向调用方推送结果）等。适合放「和具体某次状态无关、但节点执行时需要用到」的依赖。

Node 是图中的基本处理单元，代表工作流中的一个操作步骤，可绑定 Agent、大模型、工具或任意 Python 函数，具体逻辑自由实现。本质上，**节点就是「逻辑的容器」**：用 Python 函数把一段逻辑包起来，输入来自 State（和可选的 config、runtime），输出再写回 State。

**写节点时的设计原则：**

- **单一职责**：一个节点只做一件事，方便维护和复用。
- **无状态 / 纯函数**：数据通过 state 传入，尽量不依赖外部可变状态；相同输入得到相同输出，便于重试和测试。
- **可测试性**：逻辑集中在节点函数内，便于单测和排查问题。

**常见用法**：在节点里做 LLM 调用、工具调用，或组合成 Agent 的一步。

【案例源码】`案例与源码-3-LangGraph框架/04-node/DefNode.py`（节点定义方式、`functools.partial` 绑定额外参数、`add_node` 时传入 `RetryPolicy`）

[DefNode.py](案例与源码-3-LangGraph框架/04-node/DefNode.py ":include :type=code")

### 1.2 START 与 END 节点

**START** 为图的入口，可用 `add_edge(START, node_id)` 或 `set_entry_point(node_id)` 指定；**END** 为终止节点，可用 `add_edge(node_id, END)` 或 `set_finish_point(node_id)` 指定。二者都是 LangGraph 内置的「虚拟节点」，表示流程的起点和终点，不是你自己定义的普通节点。

**START（入口）**：从「图开始」连到第一个要执行的节点，执行时就会先跑这个节点。

```python
from langgraph.graph import START
# 第一个执行的节点是 node_start
graph.add_edge(START, "node_start")
```

**END（终止）**：从某个节点连到 END，表示该节点执行完后流程结束，没有后续节点。

```python
from langgraph.graph import END
# node_a 执行完后，没有后续节点了
graph.add_edge("node_a", END)
```

### 1.3 节点缓存（Node Caching）

LangGraph 支持基于节点输入的缓存，通过 `CachePolicy(key_func, ttl)` 配置；编译时指定缓存实现（如 `InMemoryCache()`）。缓存命中时直接返回结果，未命中时执行节点并写入缓存。

![节点缓存：key_func 与 ttl 配置示意](images/24/24-1-3-1.jpeg)

**图中要点说明：**

- **启用缓存的两步**：① 在**编译图**（或指定入口点）时传入缓存实现（如 `InMemoryCache()`）；② 为需要缓存的节点指定 **CachePolicy**（每个节点可单独配置策略）。
- **key_func**：根据**节点输入**生成缓存键的函数。输入经过 `key_func` 得到唯一键；若两次输入的键相同，则命中同一缓存、直接返回上次结果，不再执行节点。
- **ttl**：缓存的**存活时间（秒）**。超过 ttl 后该条缓存失效，下次会重新执行节点并写入新结果。若不设置 ttl，缓存**永不过期**（进程内一直有效，直到清空或重启）。

【案例源码】`案例与源码-3-LangGraph框架/04-node/Node_Cache.py`

[Node_Cache.py](案例与源码-3-LangGraph框架/04-node/Node_Cache.py ":include :type=code")

### 1.4 错误处理与重试机制

在 `add_node` 时传入 `retry_policy`（`RetryPolicy`，含 `max_attempts`、`retry_on` 等），可对节点配置重试；默认 `retry_on` 对多数异常重试，`ValueError`、`TypeError` 等不在重试列表中，也可自定义 `retry_on`。

**定义重试策略**：用 `RetryPolicy` 指定「最多重试几次、间隔多久、只对哪些异常重试」等；再在 `add_node` 时把该策略传给对应节点。用 `RetryPolicy` 指定「最多重试几次、间隔多久、只对哪些异常重试」等；再在 `add_node` 时把该策略传给对应节点。

```python
# 重试策略：最多 3 次，初始间隔 1 秒，带退避与抖动，仅对网络类异常重试
retry_policy = RetryPolicy(
    max_attempts=3,              # 最大重试次数（含首次共 3 次）
    initial_interval=1,           # 首次重试前等待 1 秒
    jitter=True,                  # 加随机抖动，避免大量请求同时重试（重试风暴）
    backoff_factor=2,             # 每次重试间隔翻倍：1s → 2s → 4s …
    retry_on=[RequestException, Timeout]  # 只对这类异常重试，其它异常直接抛出
)
graph.add_node("process", process_node, retry_policy=retry_policy)
```

- **max_attempts**：总共尝试执行的次数（含第一次）。
- **initial_interval** / **backoff_factor**：重试间隔从 `initial_interval` 秒开始，每次按 `backoff_factor` 倍增。
- **jitter**：在间隔上加随机偏移，避免同时重试压垮服务。
- **retry_on**：列表或可调用对象，指定「哪些异常才重试」；未列出的异常（如 `ValueError`）不重试，直接抛出。

【案例源码】`案例与源码-3-LangGraph框架/04-node/Node_ExpErrRetry.py`

[Node_ExpErrRetry.py](案例与源码-3-LangGraph框架/04-node/Node_ExpErrRetry.py ":include :type=code")

---

## 2、Graph API 之 Edge（边）

### 2.1 简介

**知识出处**：[LangGraph 官方文档 - Graph API > Edges](https://docs.langchain.com/oss/python/langgraph/graph-api#edges)

Edge 定义节点之间的连接与执行顺序；一个节点可有多个出边，多节点可指向同一节点。

可以这样理解：**边就是流程的「连接线」**——从哪个节点到哪个节点、先执行谁后执行谁，都由边来决定。

边的两大基础类型是 **普通边**（固定从 A 到 B）和 **条件边**（根据当前状态决定下一步走哪条路）；它们的核心作用都是 **实现复杂的路由逻辑**，让工作流既能顺序执行，也能按条件分支、汇聚。

类型包括：普通边、条件边、入口点、条件入口点（后两种见下文）。

![添加边的代码示意](images/24/24-2-1-1.jpeg)

**图中要点说明：**

- 左侧代码用多条 `add_edge` 把「固定边」串成一条线：`START → input → process → output → END`，执行顺序即按此路径依次跑。
- 写完后需调用 `graph.compile()` 编译图，框架会校验结构是否正确（若某条边指向的节点尚未用 `add_node` 添加，会报错）。
- 右侧流程图即该图编译后的直观表示：`__start__`、`input`、`process`、`output`、`__end__` 按箭头顺序执行；最后通过 `app.invoke(初始状态)` 运行图。

![Edge 关键类型示意](images/24/24-2-1-2.jpeg)

**四种边/入口的用法小结（对应上图）：**

- **普通边（Normal Edges）**：不判断条件，直接「从当前节点到下一个节点」，无分支。
- **条件边（Conditional Edges）**：先执行一个路由函数，根据当前状态/返回值决定下一步走到哪个（或哪些）节点，实现分支。
- **入口点（Entry Point）**：指定「用户输入进来后，图从哪个节点开始跑」，即图的起点。
- **条件入口点（Conditional Entry Point）**：用户输入进来后，先调一个函数，根据结果决定从哪个（或哪些）节点开始执行，起点可动态选择。

### 2.2 Normal Edges（普通边）

普通边表示无条件地从当前节点跳转到下一节点。

【案例源码】`案例与源码-3-LangGraph框架/05-edge/Edge_Normal.py`

[Edge_Normal.py](案例与源码-3-LangGraph框架/05-edge/Edge_Normal.py ":include :type=code")

### 2.3 Conditional Edges（条件边）

使用 `add_conditional_edges(节点名, 路由函数, 映射)`，根据状态选择性路由到不同节点。当流程不是「固定从 A 到 B」、而要**根据当前状态选择**下一步走哪条边（甚至结束）时，就用条件边。

**基本用法**：从某节点出发，用「路由函数」决定下一跳。路由函数接收当前 `state`，返回值即下一个节点的名字（或节点名列表；列表时多个节点会在下一步并行执行）。

```python
graph.add_conditional_edges("node_a", routing_function)
```

**带映射的写法**：若路由函数返回的是布尔、枚举或少量离散值，可用字典把「返回值」映射到「节点名」，语义更清晰。

```python
# 返回 True 去 node_b，返回 False 去 node_c
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

![条件边根据状态动态路由](images/24/24-2-3-1.jpeg)

**图中要点说明：**

- 流程从 START 进入后，先经过 **route func**（路由函数）；路由函数根据当前状态做判断，从多条**条件边**中选一条。
- 图中示例：满足 condition_1 走 node 1、condition_2 走 node 2、condition_3 走 node 3，即「同一出口、多路分支、只走其一」；三个节点执行完后均可再汇聚到 END。
- 这就是「根据状态动态路由」：下一步走哪个节点由运行时状态决定，而不是写死的顺序。

【案例源码】`案例与源码-3-LangGraph框架/05-edge/Edge_Conditional.py`、`Edge_ConditionalV2.py`

[Edge_Conditional.py](案例与源码-3-LangGraph框架/05-edge/Edge_Conditional.py ":include :type=code")
[Edge_ConditionalV2.py](案例与源码-3-LangGraph框架/05-edge/Edge_ConditionalV2.py ":include :type=code")

### 2.4 Entry Point 与 Conditional Entry Point

- **入口点**：`set_entry_point(node_id)` 或 `add_edge(START, node_id)`。
- **条件入口点**：`add_conditional_edges(START, route_function, mapping)`，根据输入从不同节点开始。

**入口点（Entry Point）**：图启动时第一个执行的节点。用「从 START 连到该节点」或 `set_entry_point` 指定即可。

```python
from langgraph.graph import START
graph.add_edge(START, "node_a")   # 图从 node_a 开始执行
```

**条件入口点（Conditional Entry Point）**：图从哪个节点开始不固定，由路由函数根据**初始状态/输入**决定。用法与条件边相同，只是源节点改为 `START`；路由函数接收当前 state，返回值（或映射字典中的键）对应要执行的第一个节点。

```python
from langgraph.graph import START
# 路由函数根据 state 返回下一节点名，或返回可映射的值
graph.add_conditional_edges(START, routing_function)
# 或带映射：返回 True 从 node_b 开始，返回 False 从 node_c 开始
graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
```

【案例源码】`案例与源码-3-LangGraph框架/05-edge/Edge_EntryPoint.py`、`Edge_ConditionalEntryPoint.py`

[Edge_EntryPoint.py](案例与源码-3-LangGraph框架/05-edge/Edge_EntryPoint.py ":include :type=code")
[Edge_ConditionalEntryPoint.py](案例与源码-3-LangGraph框架/05-edge/Edge_ConditionalEntryPoint.py ":include :type=code")

---

## 3、Send、Command 与 Runtime 上下文

### 3.1 总体概述

Send 和 Command 用于**高级工作流控制**：动态决定下一节点、是否更新状态、是否并行多路执行等。

### 3.2 Send（多路并行与 Map-Reduce）

从条件边返回 `Sequence[Send]`，可为每个 Send 指定目标节点和传入状态，LangGraph 会并行执行，常用于 Map-Reduce（拆分任务 → 并行执行 → 汇总）。Send 接受两个参数：节点名称、传递给该节点的状态。

**为何需要 Send？**

默认情况下边和节点是事先写死的，且共享同一份 State。但有些场景下「下一步有几条边、每条边传什么状态」要**运行时才确定**，例如：前一个节点产出一份列表，你想对**列表里每一项**都跑同一个下游节点，且**每一项带自己的那一份状态**（条数事先不知道）。这时条件边的路由函数可返回 **多个 `Send`**，每个 `Send` 指定「去哪个节点」和「传给该节点的状态」，框架会据此**动态开出多路分支**，并行执行。

**示例**：`state["subjects"]` 里有多个主题，对每个主题都去节点 `generate_joke` 生成笑话，每路传入自己的 `{"subject": s}`。

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

`node_a` 执行完后会调用 `continue_to_jokes`；返回的多个 `Send` 会形成多路并行，每路都进入 `generate_joke` 并带上各自的 `subject`。

![Map-Reduce：拆分 → 并行 → 汇总](images/24/24-3-2-1.jpeg)

**Send 的参数**：`Send(node, arg)` —— `node` 为目标节点名称（字符串），`arg` 为传给该节点的状态或消息（任意类型，通常为 dict）。每个 `Send` 会触发一次该节点的执行，且**只**收到这份 `arg`，即「每路自带私有状态」。

**Map-Reduce 式写法示例**：上游节点产出任务列表 `state["tasks"]`，路由函数为每个任务生成一个 `Send`，发往同一节点 `process_task`；再通过普通边把所有 `process_task` 汇聚到 `reduce_results`（扇出 → 并行 → 扇入）。

```python
def route_tasks(state: MapReduceState) -> list[Send]:
    sends = []
    for idx, task in enumerate(state["tasks"]):
        send = Send("process_task", {"task_id": idx, "task_name": task})
        sends.append(send)
    return sends

graph.add_conditional_edges("generate_tasks", route_tasks)  # 上游完成后调用 route_tasks，按返回的 Send 列表分发
graph.add_edge("process_task", "reduce_results")            # 所有 process_task 都完成后，才执行汇总节点
```

**三条关键规则：**

- ① 路由函数返回的 `list[Send]` 中，**每个 Send 触发一次目标节点的独立执行**，且各自携带自己的 `arg`（私有状态），多路**并行**。
- ② `add_conditional_edges("generate_tasks", route_tasks)` 表示 `generate_tasks` 执行完后调用 `route_tasks`，用其返回的 Send 列表完成**任务分发**。
- ③ `add_edge("process_task", "reduce_results")` 表示**所有** `process_task` 实例都执行完毕后，才会进入 `reduce_results` 做汇总。

【案例源码】`案例与源码-3-LangGraph框架/06-specialApi/SendDemo.py`

[SendDemo.py](案例与源码-3-LangGraph框架/06-specialApi/SendDemo.py ":include :type=code")

### 3.3 Command（状态更新与流程控制）

Command 可同时**更新状态**和**指定下一节点**（或 END），常用于人机闭环与多智能体交接。

与条件边的区别：条件边只做路由；Command 在路由的同时更新状态。

**两点约束**：Command **每次只能路由到一个节点**（不能像 Send 那样一次开出多路）；并且可以在同一步里**更新当前 State**，相当于「先改状态，再决定往哪跳」。

**基本用法**：节点函数返回 `Command(update=..., goto=...)`。`update` 为要对图状态做的修改（字典，会按 reducer 合并进全局状态）；`goto` 为下一节点名或 `END`（流程会跳转到该节点，相当于动态指定了一条边）。

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},   # 状态更新：把 state["foo"] 设为 "bar"
        goto="my_other_node"     # 流程控制：下一步执行 my_other_node
    )
```

**根据条件动态返回 Command**：在节点内根据 `state` 判断，再返回不同的 `Command`，可同时改状态和选下一跳，适合「先算一把、再决定交给谁」的场景（如人机闭环、转交专家节点）。

```python
# 在节点函数中返回 Command 来实现动态路由
def agent_node(state: State) -> Command:
    if need_help(state):
        # 需要帮助：转交 expert_agent，并更新 messages
        return Command(goto="expert_agent", update={"messages": state["messages"] + [new_message]})
    else:
        # 不需要：直接结束
        return Command(goto=END)
```

【案例源码】`案例与源码-3-LangGraph框架/06-specialApi/CommandDemo.py`

[CommandDemo.py](案例与源码-3-LangGraph框架/06-specialApi/CommandDemo.py ":include :type=code")

### 3.4 Runtime 运行时上下文

通过 `context_schema` 将**不属于图状态**的配置（如模型名、数据库连接、API 密钥）传入节点，节点通过 `runtime.context` 访问；实现配置与状态分离、类型安全、依赖统一管理。

**通俗理解**：图的 State 用来存「业务数据在节点间怎么流转」；而模型名、API Key、数据库连接等属于**运行环境/全局配置**，不适合塞进 State。Runtime 上下文就是专门放这类配置的：建图时用 `context_schema` 声明「有哪些字段」，调用 `invoke` 时用 `context` 传入具体值，节点里用 `runtime.context.xxx` 按需读取。

**三步用法：**

1. **定义 context_schema**：用 `@dataclass` 定义一个类，描述运行时上下文里有哪些字段（如 `llm_provider`），可带默认值。
2. **建图时挂上 schema，调用时传入 context**：`StateGraph(State, context_schema=ContextSchema)`；执行时 `graph.invoke(inputs, context={"llm_provider": "anthropic"})`，传入的值会覆盖 schema 中的默认值，并做类型校验。
3. **在节点中访问**：节点函数增加参数 `runtime: Runtime[ContextSchema]`，通过 `runtime.context.llm_provider` 等访问，类型安全。

```python
from dataclasses import dataclass

@dataclass
class ContextSchema:
    llm_provider: str = "openai"

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)  # 从运行时上下文取配置
    # ... 节点逻辑
    return state

graph = StateGraph(State, context_schema=ContextSchema)
# 执行时指定上下文信息，会覆盖 ContextSchema 中的默认值
graph.invoke(inputs, context={"llm_provider": "DeepSeek-R1-Online-0120"})
```

【案例源码】`案例与源码-3-LangGraph框架/06-specialApi/RuntimeContextDemo.py`

[RuntimeContextDemo.py](案例与源码-3-LangGraph框架/06-specialApi/RuntimeContextDemo.py ":include :type=code")

---

**本章小结：**

- **Node**：图的执行单元，可绑定任意函数；START/END、`set_entry_point` / `set_finish_point`；**CachePolicy**（key_func、ttl）、**RetryPolicy**（max_attempts、retry_on）；案例见 `04-node/`。
- **Edge**：普通边、条件边、入口点、条件入口点；`add_conditional_edges` 实现分支与动态入口；案例见 `05-edge/`。
- **Send / Command / Runtime**：**Send** 用于 Map-Reduce 式多路并行（条件边返回 `Sequence[Send]`）；**Command** 在节点内同时更新状态并指定下一跳（或 END）；**Runtime** 通过 `context_schema` 将配置与状态分离；案例见 `06-specialApi/`。

**建议下一步：**
 在本地按顺序运行 Node、Edge、Send/Command/Runtime 案例，并尝试修改条件边路由、Command 的 update/goto、Runtime 的 context；若需子图、多智能体或持久化，可继续学习后续 LangGraph 进阶章节。
