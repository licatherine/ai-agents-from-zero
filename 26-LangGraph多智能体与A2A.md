# 26 - LangGraph 多智能体与 A2A

---

**本章课程目标：**

- 理解 **A2A（Agent-to-Agent）协议**与 **MCP（Model Context Protocol）** 的本质区别：前者侧重代理协作，后者侧重工具访问；能用自己的话说明二者在「修车厂」等类比中的角色。
- 掌握**多智能体架构**的常见形态：单智能体、Network、Supervisor、Supervisor as tools、Hierarchical、Custom；能根据场景做大致选型。
- 会阅读并运行 **Supervisor** 与 **Handoff** 案例代码（含 V0.3 / V1.0 与 Handoff 工具），理解主管协调子 Agent、控制权交接与状态传递的用法。
- 了解 **Agent Skills** 的定位：提示词的规范化与工程化落地，可复用、可组合的技能单元。

**前置知识建议：** 已学习 [第 22 章 LangGraph 概述与快速入门](22-LangGraph概述与快速入门.md)、[第 23 章 图与状态](23-LangGraphAPI：图与状态.md)、[第 24 章 节点、边与进阶](24-LangGraphAPI：节点、边与进阶.md)、[第 25 章 流式、持久化、时间回溯与子图](25-LangGraphAPI：流式、持久化、时间回溯与子图.md)，掌握图、State、流式与持久化；建议已学 [第 17 章 工具调用](17-Tools工具调用.md)、[第 21 章 Agent 智能体](21-Agent智能体.md)。

**学习建议：** 先通读「A2A 与 MCP 的区别」和「多智能体架构形态」建立概念，再按顺序学习单智能体示例、Supervisor（V0.3 / V1.0）、Handoff；Agent Skills 可作为扩展阅读。案例源码位于 `案例与源码-3-LangGraph框架/08-multi_agent`。

---

## 1、A2A 协议与多智能体架构概览

### 1.1 概述

当前 AI 架构中，两个重要协议在重塑智能系统构建方式：**Google 的 Agent-to-Agent 协议（A2A）** 与 **Model Context Protocol（MCP）**。二者共同指向一个趋势：从**确定性编程**转向**自主协作系统**。

### 1.2 协议的本质区别：工具 vs 代理

- **MCP（Model Context Protocol）**  
  关注**工具访问**：定义大模型如何与各种工具、数据、资源交互的标准方式。可理解为：让 AI 能像程序员调用函数一样使用各类能力。

- **A2A（Agent-to-Agent Protocol）**  
  关注**代理协作**：定义智能体之间如何发现、通信、协作，使不同 AI 系统能像人类团队一样协同完成任务。

**类比：**

- **MCP** 像「工具车间」：让工人（模型）知道每个工具（API、函数）的位置和用法，但不负责工人之间的分工与协作。
- **A2A** 像「会议室」：让不同专业代理能坐在一起，理解彼此专长并协调如何共同完成复杂任务。

**修车厂例子：** MCP 让维修工知道如何使用千斤顶、扳手等工具（人 invoke 工具）；A2A 让客户能与维修工沟通（「车有异响」），并让维修工之间或与配件商代理协作（「发左轮照片」「漏液持续多久了？」）。

### 1.3 多智能体架构（Multi-Agent Architecture）

在 LangChain 体系中，**LangChain** 主要负责与大语言模型的交互，**LangGraph** 负责复杂流程调度。二者结合可实现**多智能体架构**：不是让一个大模型「包打天下」，而是由**多个专精的 Agent 协作**完成更复杂的任务。

- **单智能体（Single-Agent）**：一个 LLM + 一组工具；LLM 自己决定是否调工具、完成所有逻辑。适合简单对话助手、单一领域（天气、翻译、知识库 QA 等）。
- **多智能体（Multi-Agent）**：多个 Agent 节点组成一张图（Graph），通过消息传递、条件跳转（Command/Send）与记忆（Memory）协作；适合任务复杂、需要分工与流程编排的场景。

**口语对照（帮助记忆）：**

- 单智能体 ≈ **一个「大脑」包圆**：同一大模型（同一套提示词 + 工具）负责整条链路上的决策。
- 多智能体 ≈ **多个角色分工**：多个 Agent（可以是同一模型的不同人设与工具，也可以是不同实例）各司其职，经 Supervisor、Handoff、Network 等方式**协作**，而不是一个节点里硬写所有逻辑。

**相对「单智能体一把梭」，多智能体常见好处：**

- **解耦复杂任务**：每个 Agent 只深耕自己的子领域（如订票 / 客服 / 数据分析），提示词与工具更简单、易维护。
- **可扩展**：新业务线可**增量**加 Agent 或子图，少动原有编排。
- **更可控**：结合 LangGraph 的 **HITL（人在回路）**、**时间回溯（Time Travel）** 等，便于在关键步骤暂停、改状态或重跑分支。

**概念辨析：只有「大模型 + 工具」算不算智能体？和 [第 17 章 Tools 工具调用](17-Tools工具调用.md) 是什么关系？**

- **可以这么用，而且很常见。** 第 17 章讲的是：工具如何定义、模型如何输出 `tool_calls`、你的代码如何真正执行并把 **ToolMessage** 喂回模型——这是**工具调用能力**本身。
- **算不算「智能体」要看有没有「决策闭环」。** 若只是**单次**「模型说调工具 → 你执行一次 → 模型收结果写回复」，有人称为 **Tool-augmented LLM（带工具的模型）**；若在 [第 21 章 Agent 智能体](21-Agent智能体.md) 那种 **ReAct / Agent 循环**里，模型可以**多轮**决定「再调不调用、调哪个、何时结束」，通常就称为 **Agent（智能体）**。业界口径不完全统一，但教程里可以把：**一个 LLM 作为唯一决策主体 + 工具 + 多步编排** 理解为 **单智能体**；下面的 `LangGraphAgent.py` 即这种形态。
- **与多智能体的分界**：单智能体是**一个**决策主体搞定（或尝试搞定）任务；多智能体是**多个**这样的主体（或多个专用子图）在图里分工、路由、交接。**先掌握 17 章工具 + 21 章单 Agent，再读本章多 Agent**，顺序最顺。

【案例源码】`案例与源码-3-LangGraph框架/08-multi_agent/LangGraphAgent.py`：一个 Agent 绑定天气工具，用户问天气时由该 Agent 调用工具并回复。

[LangGraphAgent.py](案例与源码-3-LangGraph框架/08-multi_agent/LangGraphAgent.py ":include :type=code")

![多智能体架构示意：多个 Agent 组成图并协作](images/26/26-1-3-1.gif)

### 1.4 多智能体的常见连接方式

- **Network（网络型）**：多个 Agent 地位平等、彼此都可通信，也都能决定下一步找谁协作；通俗地说，像一个**没有总指挥的小组会**。适合多视角协作、并行搜索与汇总、研究讨论等场景。  
  例子：用户问“新能源车市场前景如何？”，Agent A 查政策，Agent B 查技术趋势，Agent C 查竞争对手；最后三者交换信息并汇总，给出一份综合分析。

- **Supervisor（监督者型）**：由一个主控 Agent（Supervisor）统一调度其他 Agent，子 Agent 先向主管汇报，再由主管决定任务继续交给谁；通俗地说，像**项目经理分派任务**。适合企业助手（IT/HR/财务）、智能客服按领域分配专家等场景。  
  例子：用户问“帮我报销差旅费”，Supervisor 会把任务路由给财务 Agent；用户问“我的邮箱密码忘了”，Supervisor 则把任务转给 IT Agent。

- **Supervisor as tools**：本质上仍是主管模式，只是主管把不同「子智能体」当成工具调用，像在一个总控大脑里挂了多个**专业插件**；由主管这个 LLM 决定调哪个子智能体以及传什么参数。适合希望保留统一控制入口、但又想复用多个专业子 Agent 的场景。  
  例子：主 LLM 在回答问题时，按需调用 `法律Agent()` 或 `翻译Agent()`；此时子 Agent 对主 LLM 来说，更像可调用的“高级工具”。

- **Hierarchical（层级型）**：包含多层 Supervisor，顶层先拆大任务，再交给下层主管继续细分，最后才到具体执行 Agent；通俗地说，像**公司分层组织架构**。适合大型任务拆解、复杂流程编排、跨团队协作等场景。  
  例子：任务是“写一份智能家居市场调研报告”，顶层 Supervisor 先把任务拆成「市场」「技术」「用户调研」三块；其中市场 Supervisor 再管理多个 Agent 去查政策、竞争对手和数据，技术 Supervisor 再管理硬件与软件趋势分析 Agent，最后由顶层统一汇总成完整报告。

- **Custom（自定义）**：不拘泥于固定模板，直接使用 LangGraph 的节点、边、条件路由、HITL 等能力，按业务自由拼装执行流；通俗地说，就是**自己搭一套最贴合业务的协作机制**。它最灵活，但也最依赖开发者对流程的设计能力。  
  例子：企业内部 Copilot 可设计为“用户输入 → Supervisor 判断 → 财务 / IT / HR Agent 处理”；其中某些场景还允许 Agent 之间继续协作（如 IT + 安全），最终高风险结果再交给人类审批（HITL）。

**换一个角度理解：按“协作方式”给多智能体分型**

上面讲的是常见**连接结构**（Network、Supervisor、Hierarchical 等）；如果换个视角，也可以按“任务是怎么被分工与推进的”来理解多智能体。下面这几类更偏**执行模式**，和前面的结构型分类并不冲突：

- **路由型**：先由一个分发者（Dispatcher / Router）判断问题属于哪类，再把任务交给最合适的 Agent。适合智能客服、多领域问答、企业内部 AI 助手的统一入口。
- **协作型**：多个 Agent 并行处理不同子任务，最后把结果合并。适合旅游规划、信息汇总、行业研究、报告素材收集等场景。
- **辩论型**：常见写法是 `Proposer + Critic/Judge`，一个 Agent 先给方案，另一个负责挑错、审查、评分或纠偏。适合代码生成、法律/合同审查、需要较强校验的内容生成。
- **分阶段型**：像一条 Pipeline，不同 Agent 负责不同阶段，前一阶段输出作为后一阶段输入。适合 ETL、报告生成、数据处理流水线等任务。
- **人机混合型**：关键步骤引入 **HITL（人在回路）**，必要时允许人工审批、回退、重试，甚至配合时间回溯。适合高风险决策、审批流、企业内部敏感操作。

可以把它们理解成：

- 前面的 **Network / Supervisor / Hierarchical / Custom** 更像是在回答：**“这些 Agent 之间怎么连？”**
- 这里的 **路由型 / 协作型 / 辩论型 / 分阶段型 / 人机混合型** 更像是在回答：**“这些 Agent 是按什么方式一起干活？”**

如果要做一个企业内部 AI 助手，常见组合方式可能是：

- **入口用路由型**：先判断用户问的是 IT、HR 还是财务问题。
- **中间用协作型**：例如 IT 场景里并行查询知识库、日志系统、监控信息，再合并结果。
- **关键内容加辩论型**：例如 HR 回复或合规文案，先生成，再交给 Critic 做语气与合规检查。
- **高风险结果接人机混合型**：最终敏感回答、审批动作或外部发送前，交给人工确认。

---

## 2、多智能体案例：Supervisor 与 Handoff

### 2.1 Supervisor（主管）架构

**Supervisor** 模式由一个**中央主管智能体**协调所有子智能体：主管控制通信流与任务委派，根据当前上下文与任务需求决定调用哪个子 Agent，类似企业中的「项目经理」——管理者接收任务、分解并委派给各工作者 Agent，最后整合结果。

LangGraph 提供 **langgraph-supervisor** 库（`pip install langgraph-supervisor`），可快速搭建 Supervisor 图。

![Supervisor 架构：主管协调多个子 Agent](images/26/26-2-1-1.gif)

**流程说明（概念）：** 用户输入 → Supervisor 解析需求 → 按需调用 flight_assistant / hotel_assistant → 子 Agent 调用工具并回报 → Supervisor 汇总并回复用户。

<img src="images/26/26-2-1-2.gif" alt="Supervisor 执行流程示意" width="40%" />

**老版本示例（V0.3，基于 create_react_agent）**  
【案例源码】`案例与源码-3-LangGraph框架/08-multi_agent/SupervisorV0.3.py`

[SupervisorV0.3.py](案例与源码-3-LangGraph框架/08-multi_agent/SupervisorV0.3.py ":include :type=code")

**新版本示例（V1.0，基于 create_agent）**  
【案例源码】`案例与源码-3-LangGraph框架/08-multi_agent/SupervisorV1.0.py`

[SupervisorV1.0.py](案例与源码-3-LangGraph框架/08-multi_agent/SupervisorV1.0.py ":include :type=code")

### 2.2 Handoff（交接）

**Handoff** 指一个智能体将**控制权与状态**交接给另一个智能体，需包含：**目的地**（下一个 Agent）与**传递给下一 Agent 的 State**。Supervisor 通常使用 `create_handoff_tool` 等移交工具；也可以**自定义 Handoff 工具**，在父图中用 Command + Send 将任务与状态交给指定 Agent。

【案例源码】`案例与源码-3-LangGraph框架/08-multi_agent/SupervisorHandoff.py`

[SupervisorHandoff.py](案例与源码-3-LangGraph框架/08-multi_agent/SupervisorHandoff.py ":include :type=code")

---

## 3、Agent Skills（智能体技能）简介

**参考链接：**

- https://agentskills.io/what-are-skills
- https://developers.openai.com/codex/skills/

**是什么：**  
可以把 **Agent** 想象成厨师，**Agent Skills** 则是厨师掌握的烹饪技法（刀工、炒、蒸、烤）与厨房工具（锅、铲、烤箱）。Agent 决定「做什么、用什么、按什么顺序」；Skills 是完成具体工序的**能力单元与流程**。二者结合才能完成复杂任务。

![Agent 与 Skills 的关系：Agent 决策，Skills 提供可复用能力](images/26/26-3-1-1.jpeg)

**类比示例（技能步骤）：** 如炒菜可拆成备料、热锅、下料、调味、装盘等步骤，每个步骤可对应一组规范化操作（类似「技能」）。下图为人做豆浆/炸油条等技能流程示例。

![技能示例：人做豆浆或炸油条等流程](images/26/26-3-1-2.jpeg)

![技能示例：炒菜步骤 2](images/26/26-3-1-3.jpeg)

![技能示例：炒菜步骤 3](images/26/26-3-1-4.jpeg)

![技能示例：炒菜步骤 4](images/26/26-3-1-5.jpeg)

![技能示例：炒菜步骤 5](images/26/26-3-1-6.jpeg)

![技能示例：炒菜步骤 6](images/26/26-3-1-7.jpeg)

![技能示例：炒菜步骤 7](images/26/26-3-1-8.jpeg)

**官方说法：** 常提到「渐进式披露」与分层架构——从简单到复杂，按需暴露能力。

![Skill 架构：简单形态](images/26/26-3-1-8.jpeg)

![Skill 架构：复杂形态](images/26/26-3-1-9.jpeg)

**一句话总结：** Agent Skills 可理解为**提示词的规范化与工程化落地**，类似提示词版本的「Maven 结构」——可复用、可组合、可管理的技能单元与依赖关系。

---

**本章小结：**

- **A2A 与 MCP**：MCP 侧重工具访问（模型如何调用工具、数据与资源），A2A 侧重代理协作（智能体之间如何发现、通信与协同）；二者互补，共同支撑从确定性编程到自主协作系统的演进。
- **多智能体架构**：常见形态包括单 Agent、Network、Supervisor、Supervisor as tools、Hierarchical、Custom；选型时需考虑任务复杂度、是否需要中心调度、是否需层级拆解等。
- **Supervisor**：由中央主管 Agent 接收任务、委派子 Agent、汇总结果；可使用 langgraph-supervisor 或自建图实现；案例见 SupervisorV0.3、SupervisorV1.0。
- **Handoff**：控制权与状态在 Agent 间交接，需明确目的地与传递的 State；可通过 create_handoff_tool 或自定义 Command + Send 实现，案例见 SupervisorHandoff。
- **Agent Skills**：将提示词与能力规范化为可复用、可组合的「技能」单元，便于工程化落地与维护。

**建议下一步：** 在本地运行 `案例与源码-3-LangGraph框架/08-multi_agent` 下的 LangGraphAgent、SupervisorV1.0、SupervisorHandoff，结合 [第 21 章 Agent 智能体](21-Agent智能体.md)、[第 25 章 流式、持久化、时间回溯与子图](25-LangGraphAPI：流式、持久化、时间回溯与子图.md) 巩固图、状态与多轮对话；若需将多智能体与业务系统深度集成，可继续查阅 LangGraph 官方 Multi-Agent 与 A2A 相关文档。
