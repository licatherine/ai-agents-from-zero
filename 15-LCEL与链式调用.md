# 15 - LCEL 与链式调用

---

**本章课程目标：**

- 理解 **Runnable** 的定位与「统一调用方式」的意义，掌握 **LCEL**（LangChain 表达式语言）及管道符 `|` 的用法。
- 掌握 **Chain** 的典型结构（提示词模板 + 大模型 + 输出解析器），理解链本身也是 Runnable、可继续组合。
- 能使用 **RunnableSequence**、**RunnableBranch**、**RunnableParallel**、**RunnableLambda** 等组合方式，会阅读并编写顺序链、分支链、串行链、并行链与函数链的示例代码。

**前置知识建议：** 已学习 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)，了解 LangChain 的六大模块与基本调用方式；建议已学 [第 11 章 Model I/O](11-Model-I-O与模型接入.md)、[第 13 章 提示词与消息模板](13-提示词与消息模板.md)、[第 14 章 输出解析器](14-输出解析器.md)，便于理解「提示 → 模型 → 解析」在链中的位置。

**学习建议：** 先动手跑通一条 `prompt | model | parser` 顺序链，再依次尝试分支链、串行链、并行链与函数链；链式调用是后续 [第 16 章 记忆与对话历史](16-记忆与对话历史（含Redis基础）.md)、[第 17 章 Tools 工具调用](17-Tools工具调用.md) 以及 LangGraph 的基础。

---

## 1、Runnable 与统一调用方式

### 1.1 前置知识点：抽象基类（ABC）

若你第一次接触「抽象基类」，可先建立如下概念，便于理解 Runnable 的定位。

- **是什么**：抽象基类是一种「只约定能做什么、不亲自实现具体怎么做」的类。它规定子类必须实现哪些方法（如 `invoke`、`stream`），但不（或只部分）实现这些方法的具体逻辑。
- **为什么有用**：有了统一约定，代码可以「面向接口编程」——只要对象是某种抽象基类的子类，就可以放心调用约定好的方法，而不必关心它具体是提示模板、模型还是解析器。这样组件之间易于替换和组合。
- **在 Python 中**：通过 `abc` 模块的 `ABC` 和 `@abstractmethod` 定义抽象基类；子类继承并实现所有抽象方法后，才能被正常实例化。
- **在本章中**：**Runnable** 就是这样一个抽象基类；[Prompt 模板](13-提示词与消息模板.md)、[LLM](11-Model-I-O与模型接入.md)、[Output Parser](14-输出解析器.md)、以及用 `|` 拼出来的链，都是「实现了 Runnable 接口」的具体类型，因此都可以用同一套 `invoke`、`stream`、`batch` 等方式调用。
- **LangChain 中常见的抽象基类**（均位于 `langchain_core`）：**Runnable**（可执行组件的统一接口，本章核心）；**BaseChatModel** / **BaseLLM**（[聊天模型](11-Model-I-O与模型接入.md)的基类）；**BasePromptTemplate**、**BaseChatPromptTemplate**（[提示词模板](13-提示词与消息模板.md)的基类）；**BaseOutputParser**（[输出解析器](14-输出解析器.md)的基类）。这些基类大多实现了 Runnable 接口，因此既能各自约定「模型该长什么样」「解析器该长什么样」，又能被 LCEL 用 `|` 统一组合。

### 1.2 什么是 Runnable

**Runnable** 是 LangChain 中的**抽象基类**（定义在 `langchain_core.runnables`），为所有「可执行」的组件提供**统一的操作接口**。

- **定位**：LangChain 核心抽象接口，统一组件调用方式。
- **目标**：无论组件是提示模板、模型、解析器还是整条链，都支持同一套方法（如 `invoke`、`stream`、`batch`），并支持 LCEL 组合。
- **核心理念**：一切可执行的对象都应有统一的调用方式，从而具备「即插即用」的能力。

![Runnable 在 LangChain 中的定位：统一可执行组件的接口](images/15/15-1-2-1.jpeg)

> **说明**：上图为 Runnable 的定位示意。Prompt、Model、Parser、Chain 等只要实现 Runnable 接口，就可以用相同方式调用（如 `invoke`）并用管道符 `|` 组合成链。

### 1.3 为什么需要统一调用方式

若没有统一接口，各组件调用方式不一致，组合时需要**手动记住并适配**每种组件的专属方法；使用 Runnable 之后，则**统一用 `invoke` 等一套方法**。下面按组件类型做「使用前 / 使用后」的对比，便于建立直观认识。

**1. 提示词模板（Prompt）**

- **使用前**：有的模板用 `.format(**variables)` 渲染，返回的是普通字符串，你要自己区分「这是字符串」还是「消息列表」，再传给模型。
- **使用后**：用 `prompt.invoke({"topic": "AI"})`，返回的是 LangChain 内部统一的「可传给模型」的格式（如 `PromptValue`），直接可以交给下一步。

> **可以这样理解**：LangChain 里的 Prompt 模板（如 `ChatPromptTemplate`）本质上是 **Runnable 的子类**（或实现了 Runnable 接口）。所以它既有自己「业务层面」的方法（如 `.format()`），也具备 Runnable 抽象基类规定的**公共方法**（如 `.invoke()`、`.stream()`）。用 `invoke` 时走的是统一接口，返回 `PromptValue` 等标准类型，便于和模型、解析器用 `|` 串联；需要时仍可用 `format` 做简单渲染。模型、解析器、工具同理：都是 Runnable，既有各自专属方法，也统一支持 `invoke` 等。

```python
# 使用前：写法不统一，且返回类型各异
msg = prompt.format(topic="AI")           # 或 prompt.invoke(...)，依实现而定

# 使用后：统一用 invoke，返回类型一致，便于串联
prompt_value = prompt.invoke({"topic": "AI"})
```

**2. 模型（LLM / Chat Model）**

- **使用前**：有的封装用 `model.generate(...)`，有的用 `model(...)`，入参可能是字符串或消息列表，需要按各库文档分别查。
- **使用后**：一律 `model.invoke(输入)`，输入可以是上一步的 `PromptValue` 或消息列表，输出是统一的 `AIMessage` 等，便于交给解析器。

```python
# 使用前：不同库/封装调用方式不一
response = model.generate(prompt_value)   # 或 model(prompt_value)，因实现而异

# 使用后：统一入口
ai_message = model.invoke(prompt_value)
```

**3. 输出解析器（Output Parser）**

- **使用前**：解析器常用 `.parse(text)` 或 `.parse_with_prompt(...)`，你要从模型返回对象里先取出「文本」再传进去，链式组合时要手写中间步骤。
- **使用后**：直接用 `parser.invoke(ai_message)`，解析器会从消息里取内容并解析，和上一步的 `AIMessage` 无缝衔接。

```python
# 使用前：先取文本再解析，接口不统一
text = ai_message.content
result = parser.parse(text)

# 使用后：一步到位，入参即上一步输出
result = parser.invoke(ai_message)
```

**4. 工具（Tool）**

- **使用前**：工具常见 `.run(input)` 或 `.invoke(input)`，和「链」的写法不一致，接到链里要单独写一层适配。
- **使用后**：工具也实现 Runnable，用 `tool.invoke(input)`，和 prompt、model、parser 用法一致，可直接用 `|` 参与组合。工具定义与 [Agent](21-Agent智能体.md) 用法见 [第 17 章](17-Tools工具调用.md)。

```python
# 使用前：可能是 .run()，和链的 invoke 不统一
output = some_tool.run("查询天气")

# 使用后：与链上其他组件一致
output = some_tool.invoke("查询天气")
```

**小结**：使用 Runnable 之后，所有组件都通过 **invoke**（以及 stream、batch 等）调用，例如：

```python
prompt_value = prompt.invoke({"topic": "AI"})
ai_message  = model.invoke(prompt_value)
result      = parser.invoke(ai_message)
chain.invoke({"question": "你好"})   # 整条链同样用 invoke
```

本质是：**接口统一让组件具备了「即插即用」的能力**，便于用管道符 `|` 串联和替换，而不必再记「提示词用 format、解析器用 parse、工具用 run」等差异。

### 1.4 Runnable 接口与核心方法

实现 **Runnable** 接口的对象表示「可以执行的数据流节点」，既可以是**单个组件**，也可以是**整条链**或**复合结构**。具体包括：

- **单个组件**：如 [Prompt 模板](13-提示词与消息模板.md)、[Model](11-Model-I-O与模型接入.md)、[Output Parser](14-输出解析器.md)
- **顺序流程**：如 prompt → model → parser 串联而成的一条链（即 [第 13 章](13-提示词与消息模板.md) + [第 11 章](11-Model-I-O与模型接入.md) + [第 14 章](14-输出解析器.md) 的组合）
- **复合结构**：并行、多路、多输入多输出的组合（如 RunnableParallel、RunnableBranch）

只要实现了 Runnable 接口，就可以像函数一样用 **invoke()** 调用，或用管道符 **|** 与其他 Runnable 组合成新链。

**Runnable 接口中定义的常用方法如下：**

- <strong>invoke(input)</strong>：同步执行，处理单个输入；最常用，适用于交互式或单次调用。
- <strong>batch(inputs)</strong>：批量执行，一次处理多个输入；适用于批量任务，提升吞吐与效率。
- <strong>stream(input)</strong>：流式执行，逐步返回结果；典型场景为大模型逐字/逐 token 输出，需实时展示生成内容时使用。
- <strong>ainvoke(input)</strong>：异步执行，处理单个输入；用于高并发、非阻塞 I/O 场景。

此外还有 <strong>astream(input)</strong>（异步流式）、<strong>abatch(inputs)</strong>（异步批量）等变体，与上述方法语义一致，仅改为异步调用，便于在 asyncio 或高并发框架中与其他异步逻辑配合使用。

---

## 2、LCEL 是什么

### 2.1 定义与一句话

<strong>LCEL</strong>（LangChain Expression Language，LangChain 表达式语言）是专门用于<strong>组合 Runnable 组件</strong>的声明式语法，其<strong>核心操作符是管道符 `|`</strong>。

**一句话**：通过 LCEL（管道符 `|`、RunnableSequence、RunnableParallel 等）快速将多个 Runnable 拼接成复杂工作流，支持顺序、条件分支、并行执行等。

**典型写法示例**：

```python
chain = prompt | model | output_parser
result = chain.invoke({"question": "什么是 LangChain？"})
```

核心思想：用 `|` 把多个 Runnable 像拼积木一样组合起来，数据从左到右依次流过。

### 2.2 可组合性

LCEL 强调**可组合性**：将多个组件按特定顺序或分支组合成一条「链」（Pipeline），以完成复杂任务。链本身也是 Runnable，可以继续被组合。

---

## 3、Chain 结构

- 使用 LCEL 创建的 Runnable 我们称为**「链」（Chain）**；链本身也是 Runnable，可以继续用 `|` 或 RunnableParallel 等组合。
- **Chain 的典型结构**由三部分组成：
  1. **[提示词模板](13-提示词与消息模板.md)**（Prompt）
  2. **[大模型](11-Model-I-O与模型接入.md)**（LLM / Chat Model）
  3. **[结果结构化解析器](14-输出解析器.md)**（Output Parser，可选）

**管道运算符 `|`** 是 LCEL 最具特色的语法：多个 Runnable 通过 `|` 串联，形成清晰的数据处理链。典型链在代码中写出来就是 **`prompt | model | parser`**：从左到右依次是提示词模板、管道符、模型、管道符、解析器；数据流为「用户输入 → 经 Prompt 渲染 → 交给 Model → 模型输出 → 经 Parser 解析 → 得到最终结构化结果」。

---

## 4、链式调用基础用法与案例

下面按类型介绍几种常用链的用法，并给出对应案例源码路径与核心代码说明。

### 4.0 几种链的对比与如何选择

| 类型       | 典型写法 / 类名                                      | 执行方式                             | 输入 → 输出                            | 典型场景                                                        |
| ---------- | ---------------------------------------------------- | ------------------------------------ | -------------------------------------- | --------------------------------------------------------------- |
| **顺序链** | `prompt \| model \| parser`（RunnableSequence）      | 一步接一步，前一步输出作为下一步输入 | 单输入 → 单输出                        | 单轮问答、一次「提示 → 模型 → 解析」的完整流程                  |
| **分支链** | `RunnableBranch((条件1, 链1), (条件2, 链2), 默认链)` | 按条件只走其中一条子链               | 单输入 → 单输出（来自被选中的那一支）  | 多语言/多策略路由（如按「日语」「韩语」选不同翻译链）、意图分流 |
| **串行链** | 多条子链用 `\|` 或 lambda 串联                       | 多步顺序执行，每一步可再次调用模型   | 单输入 → 单输出（最后一步的结果）      | 多轮模型调用（如先总结再翻译、先检索再生成）                    |
| **并行链** | `RunnableParallel({"key1": 链1, "key2": 链2})`       | 多条子链同时跑，结果汇总为 dict      | 单输入 → 多输出（按 key 聚合）         | 同一问题多语言/多模型并行、多路推理或评估                       |
| **函数链** | `RunnableLambda(函数)` 或函数直接参与 `\|`           | 在链中插入自定义 Python 逻辑         | 取决于函数（常做格式转换、打印、过滤） | 中间结果处理、日志、字段映射、简单条件判断                      |

**如何选择：**

- **只有一条直线流程**（提示 → 模型 → 解析，无分支、无并行）→ 用 **顺序链**（`prompt | model | parser`）。
- **需要根据输入走不同子链**（如按语言/意图选不同提示或模型）→ 用 **分支链**（RunnableBranch）。
- **需要多步依次调用模型**（如先 A 再 B，B 依赖 A 的输出）→ 用 **串行链**（多段 `|` 或 lambda 串联）。
- **同一输入要同时跑多条链并汇总**（如中英文各答一遍、多模型投票）→ 用 **并行链**（RunnableParallel）。
- **链中要插入自定义逻辑**（转换格式、打日志、简单计算）→ 用 **函数链**（RunnableLambda 或可调用对象）。

**使用场景速览：**

- **顺序链**：单轮 QA、表单解析、一次性的「用户输入 → 模型 → 结构化输出」。
- **分支链**：多语言翻译路由、客服意图分流（咨询/投诉/转人工）、按主题选不同 prompt 或模型。
- **串行链**：先检索再生成（[RAG](19-RAG检索增强生成.md) 简化版）、先总结再翻译、多步推理链。
- **并行链**：同一问题中英双答、多模型并行取最优或投票、多路检索结果合并。
- **函数链**：在链中做字段重命名、过滤无效输出、打印调试、把上一步输出转成下一步需要的 dict 结构。

---

### 4.1 RunnableSequence（顺序链）

**顺序链**即最常见的「Prompt → Model → Parser」一条线执行，数据依次经过每个节点。

**RunnableSequence**（可运行序列）按顺序「链接」多个可运行对象：前一个对象的输出作为后一个对象的输入。LCEL 重载了管道符 **`|`**，用两个 Runnable 即可创建 RunnableSequence，因此下面两种写法等价：

- 使用管道符：`chain = runnable1 | runnable2`
- 使用显式构造函数：`chain = RunnableSequence([runnable1, runnable2])`

在典型顺序链中即为：`chain = prompt | model | parser`，对应「提示模板 → 模型 → 输出解析器」的数据流。

【案例源码】`案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableSequenceDemo.py`

[LCEL_RunnableSequenceDemo.py](案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableSequenceDemo.py ":include :type=code")

---

### 4.2 RunnableBranch（分支链）

**RunnableBranch** 实现**条件分支**：根据输入决定走哪一条子链，类似 if-else if-else。

- 初始化时传入若干 `(条件, Runnable)` 对和一个**默认分支**。
- 执行时对输入依次求值条件，**第一个为 True 的条件**对应的 Runnable 会在该输入上运行；若无一为 True，则运行默认分支。

典型用法：根据用户输入中的关键词（如「日语」「韩语」）选择不同翻译提示词与子链。

【案例源码】`案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableBranchDemo.py`

[LCEL_RunnableBranchDemo.py](案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableBranchDemo.py ":include :type=code")

---

### 4.3 RunnableSerializable / 串行链（多步串联）

当需要**多次调用大模型**、把多个步骤串联起来时，可以把多条子链用 `|` 或 lambda 串成一条「串行链」：前一步的输出作为后一步的输入。

例如：先让模型用中文介绍某主题，再把该介绍作为输入交给另一条链翻译成英文。

【案例源码】`案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableSerializableDemo.py`

[LCEL_RunnableSerializableDemo.py](案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableSerializableDemo.py ":include :type=code")

---

### 4.4 RunnableParallel（并行链）

**并行链**指**同时运行多条子链**，待全部完成后汇总结果。

适用场景举例：

- 同一问题用中英文各答一遍并聚合
- 多个模型同时跑同一问题取最优或综合
- 多路径推理、多模态（如图片 + 文本）并行处理

【案例源码】`案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableParallelDemo.py`

[LCEL_RunnableParallelDemo.py](案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableParallelDemo.py ":include :type=code")

> **延伸**：并行链的图结构可配合 `get_graph().print_ascii()` 查看，为后续学习 LangGraph 做铺垫。

---

### 4.5 RunnableLambda（函数链）

**RunnableLambda** 将**普通 Python 函数**包装成 Runnable，从而可以放入 LCEL 链中，与其他组件用 `|` 连接。

- 作用：把自定义逻辑（如打印中间结果、数据格式转换）变成链中的一个节点。
- 用法：用 `RunnableLambda(函数)` 或直接把函数放在 `|` 之间（LangChain 会自动包装）。

【案例源码】`案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableLambdaDemo.py`

[LCEL_RunnableLambdaDemo.py](案例与源码-4-LangGraph框架/06-lcel/LCEL_RunnableLambdaDemo.py ":include :type=code")

---

**本章小结：**

- **Runnable** 是 LangChain 中「可执行组件」的统一接口，通过 `invoke`、`stream`、`batch` 等方法调用；一切可执行对象具备统一调用方式，便于用管道符 `|` 串联和替换。
- **LCEL**（LangChain 表达式语言）用管道符 `|` 将多个 Runnable 组合成链，支持顺序、分支、并行与函数节点；链本身也是 Runnable，可继续组合。
- **Chain 典型结构**为「[提示词模板](13-提示词与消息模板.md) + [大模型](11-Model-I-O与模型接入.md) + [输出解析器](14-输出解析器.md)」；**顺序链**（RunnableSequence）、**分支链**（RunnableBranch）、**串行链**（多步 `|` 串联）、**并行链**（RunnableParallel）、**函数链**（RunnableLambda）是五种常见组合方式，可按业务选择或组合使用。

**建议下一步：** 在本地依次运行顺序链、分支链、串行链、并行链与函数链案例，理解数据在链中的流向；接着学习 [第 16 章 记忆与对话历史](16-记忆与对话历史（含Redis基础）.md)，实现多轮连贯对话；再学 [第 17 章 Tools 工具调用](17-Tools工具调用.md)、[第 21 章 Agent 智能体](21-Agent智能体.md)，让链具备调用外部工具与自主决策能力。
