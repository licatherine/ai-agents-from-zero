# 16 - 记忆与对话历史（含 Redis 基础）

---

**本章课程目标：**

- 理解**记忆（Memory）**在对话中的作用与必要性，掌握「读历史 → 拼入提示 → 调模型 → 写回历史」的实现思路。
- 了解 **RunnableWithMessageHistory** 与 **BaseChatMessageHistory** 的定位与用法，会选用内存版（InMemoryChatMessageHistory）实现进程内多轮对话。
- 掌握使用 **Redis**（Redis Stack）持久化对话历史的方案，会配置环境、运行 Redis 版案例并理解 session_id 与持久化写入。

**前置知识建议：** 已学习 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)、[第 15 章 LCEL 与链式调用](15-LCEL与链式调用.md)，了解[链](15-LCEL与链式调用.md)的构建与调用方式；建议已学 [第 13 章 提示词与消息模板](13-提示词与消息模板.md)、[第 14 章 输出解析器](14-输出解析器.md)。

**学习建议：** 先运行「无记忆」演示体会问题，再跑通内存版带历史对话，最后在本地启动 Redis Stack 并运行 Redis 持久化案例；复杂场景可后续结合 LangGraph 的 persistence 学习。

---

## 1、为什么需要记忆

**现象**：若只用「[Prompt](13-提示词与消息模板.md) + [Model](11-Model-I-O与模型接入.md) + [Parser](14-输出解析器.md)」且不保存历史，每次请求相互独立，模型无法知道上一轮说过什么。例如先问「我叫张三」，再问「我叫什么？」，模型会回答「我不知道你是谁」。

![无记忆时两轮对话相互独立，模型无法利用上一轮信息](images/16/16-1-1-1.jpeg)

**需求**：聊天系统需要**记忆组件**存储和管理对话的上下文，使 AI 能够根据历史内容给出连贯、个性化的回复，即「前言可以搭后语」。

**官方文档**（短期记忆）：https://docs.langchain.com/oss/python/langchain/short-term-memory

![LangChain 官方文档中关于短期记忆的说明](images/16/16-1-1-2.jpeg)

> **说明**：记忆是让 AI 记住先前互动、从反馈中学习并适应用户偏好的能力，在涉及多轮交互的复杂任务中非常重要。

---

## 2、记忆是什么与能干什么

记忆是聊天系统中**存储和管理对话上下文**的组件：让 AI 能「记住」之前的对话，在多轮中提供上下文，使回复**连贯、可延续**（前言搭后语），并可根据历史做简单个性化。

---

## 3、「我不知道」演示：无记忆时的行为

下面示例演示：同一程序内连续两次调用链，第一次告诉模型「我叫张三」，第二次问「你知道我是谁吗？」——若不注入历史，模型会回答「我不知道」。

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_IDontKnow.py`

[Memory_IDontKnow.py](案例与源码-4-LangGraph框架/07-memory/Memory_IDontKnow.py ":include :type=code")

---

## 4、实现原理

要实现记忆，需要**额外模块**保存与模型对话的上下文，并在每次请求时：

1. **读**：从记忆组件读取历史对话。
2. **写**：在链执行完毕后，将本轮用户输入与模型回复写入记忆组件。
3. **用**：在下一次请求前，把历史消息与当前用户输入一起放入提示词，再交给模型，模型即可基于完整上下文生成回复。

因此，一个记忆组件至少需要三种能力：**读取历史**、**写入历史**、**存储消息**。在 LangChain 中，负责该能力的模块统称为 **Memory（记忆）**，用于存储用户与模型交互的历史信息。

![记忆在链中的位置：读历史 → 拼入 Prompt → 调模型 → 写回历史](images/16/16-4-1-1.jpeg)

> **说明**：上图概括了带记忆的调用流程——链执行前从记忆组件读取历史并和用户输入一起填入提示；执行后将本轮输入与输出写回记忆组件。

---

## 5、实现类介绍：0.3 与 1.0+

**ConversationChain** 是 LangChain 早期用于简化对话的类，内部集成内存与固定提示模板，适合快速写简单对话，但存在：

- 灵活性不足，难以支持复杂对话流程
- 与新 API、现代聊天模型（如带工具调用的模型）兼容性差
- 与 LCEL / Runnable 的模块化设计理念不一致

**RunnableWithMessageHistory** 是 LangChain 推荐的替代方案（0.3.x 起稳定，1.0 中保留），优势包括：

- **模块化**：可自由组合提示模板、模型与记忆逻辑
- **灵活**：支持自定义历史存储（内存、Redis 等）与复杂流程
- **兼容**：与 LCEL 及现代聊天模型无缝集成

**官方建议**：

- **简单聊天**：使用 **BaseChatMessageHistory** 与 **RunnableWithMessageHistory** 配合。BaseChatMessageHistory 是保存聊天消息历史的抽象基类，常用实现包括 InMemoryChatMessageHistory（内存）、RedisChatMessageHistory（Redis）等。
- **复杂场景**：可考虑 LangGraph 的 persistence（Checkpointer + 记忆中间件等）。

**BaseChatMessageHistory 简介**：

- **属性**：`messages: List[BaseMessage]`，只读，用于读取历史消息。
- **方法**：`add_message(msg)`、`add_messages(msgs)`、`clear()`；具体存储由实现类完成。

**图 1 说明**：图中为 LangChain 源码结构示意。`BaseChatMessageHistory` 是**抽象基类（ABC）**，只定义「消息历史」的接口（如读、写、清空），不能直接实例化；`InMemoryChatMessageHistory` 是**具体实现类**，继承该接口并把消息存在进程内存里。理解「抽象类定义规范、实现类负责具体存储」即可。

![BaseChatMessageHistory 与常用实现类](images/16/16-5-1-1.jpeg)

**图 2 说明**：图中为 IDE 里「选择 BaseChatMessageHistory 的实现」时的列表示意。LangChain 提供了多种实现（如内存、文件、Redis、Elasticsearch、DynamoDB 等），对应不同存储后端；选型时根据是否需要持久化、是否多实例等需求，从列表中选择合适实现即可。本课程常用的是内存版（InMemory）和 Redis 版。

![常用消息历史组件及特性对比](images/16/16-5-1-2.jpeg)

**消息历史组件选型参考**：按存储方式可选用不同实现，选型时可根据「是否需持久化、是否多进程/多实例」等需求选择。

| 组件名称                          | 特性                                                              |
| :-------------------------------- | :---------------------------------------------------------------- |
| `InMemoryChatMessageHistory`      | 基于内存存储的聊天消息历史组件，进程内有效，重启即丢失。          |
| `FileChatMessageHistory`          | 基于文件存储的聊天消息历史组件，可持久化到本地文件。              |
| `RedisChatMessageHistory`         | 基于 Redis 存储的聊天消息历史组件，支持持久化与多实例共享。       |
| `ElasticsearchChatMessageHistory` | 基于 Elasticsearch 存储的聊天消息历史组件，适合与 ES 技术栈集成。 |

---

## 6、案例代码

### 6.1 内存版（进程内，重启即丢失）

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_RunnableWithMessageHistory.py`

[Memory_RunnableWithMessageHistory.py](案例与源码-4-LangGraph框架/07-memory/Memory_RunnableWithMessageHistory.py ":include :type=code")

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_RunnableWithMessageHistoryV2.py`

[Memory_RunnableWithMessageHistoryV2.py](案例与源码-4-LangGraph框架/07-memory/Memory_RunnableWithMessageHistoryV2.py ":include :type=code")

**直接使用 InMemoryChatMessageHistory API**：下面案例不通过 RunnableWithMessageHistory，而是手动创建 `InMemoryChatMessageHistory`、用 `add_user_message` / `add_message` 维护历史，每次将 `history.messages` 传给 `llm.invoke`。适合需要细粒度控制「何时读、何时写」的场景；多数多轮对话仍推荐用上面两种「链 + RunnableWithMessageHistory」的写法。

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_InMemoryChatMessageHistory.py`

[Memory_InMemoryChatMessageHistory.py](案例与源码-4-LangGraph框架/07-memory/Memory_InMemoryChatMessageHistory.py ":include :type=code")

### 6.2 持久化：Redis 存储

本节将对话历史保存到 Redis，实现持久化记忆，重启后仍可恢复。以下按「设计要求 → Redis 简介 → Docker 启动 → 命令与查看 → 环境验证 → 案例代码」组织。

#### 6.2.1 设计要求与参考文档

- **设计要求**：将用户与大模型的对话保存到 **Redis**，实现持久化记忆，重启后仍可恢复历史。
- **官方文档**：https://python.langchain.ac.cn/docs/integrations/memory/redis_chat_message_history/
- **本课程案例**：源码默认使用**原生 Redis**（`redis://localhost:6379`，Docker 示例为 `redis` 官方镜像），未使用 Redis Stack 特有功能；RedisChatMessageHistory 仅依赖 Redis 的 list/string 等基础结构，**原生 Redis 即可满足**。若你已安装 Redis Stack，也可使用（协议兼容），通过环境变量 `REDIS_URL` 指定端口即可（如 `redis://localhost:26379`）。

#### 6.2.2 Redis Stack 简介

本小节为拓展阅读。案例只需**原生 Redis**；若对 Redis 生态感兴趣，可了解 Redis Stack。

- **是什么**：Redis Stack = 原生 Redis + 搜索 + 图 + 时间序列 + JSON + 概率结构 + 可视化与开发框架支持。
- **核心组件**（了解即可）：RediSearch（全文/向量搜索）、RedisJSON（JSON 存储与查询）、RedisGraph（图查询）、RedisBloom（概率结构）等。
- **Python**：LangChain 1.0+ 使用 Redis 作为后端时，建议 `pip install redis==5.3.1`（原生 Redis 与 Redis Stack 通用）。

![Redis Stack 与原生 Redis 的关系](images/16/16-6-2-2-1.jpeg)

Redis Stack 是 Redis Labs 推出的**「增强版 Redis」**，**不是 Redis 的替代品**，而是在原生 Redis 基础上的功能扩展包，专为构建现代实时应用而设计。本课程对话历史案例不依赖其增强能力。

| 功能维度 | 原生 Redis                 | Redis Stack 增强功能                                  |
| :------- | :------------------------- | :---------------------------------------------------- |
| 数据结构 | 字符串、列表、集合、哈希等 | 增加 JSON、图、时间序列、概率结构等高级类型           |
| 查询能力 | 仅限键值查询               | 支持全文搜索、向量搜索、图查询、JSON 查询             |
| 使用场景 | 缓存、消息队列、计数器等   | 实时推荐、时序分析、知识图谱、文档数据库、AI 向量检索 |
| 开发体验 | 命令行操作，需手动拼装逻辑 | 提供 RedisInsight 和对象映射库，开发效率更高          |

**Docker 安装 Redis Stack**（推荐）：本机需已安装 Docker 或 Docker Desktop。

- **两种镜像区别**：`redis/redis-stack-server` 仅包含 Redis 服务，**没有** RedisInsight Web 界面；若需要浏览器打开管理界面，请使用 `redis/redis-stack`（包含 Redis + RedisInsight）。以下命令均使用 `redis/redis-stack`，以便用浏览器查看键与数据。

拉取并启动（带 RedisInsight，可打开 http://localhost:8001 ）：

```bash
docker run -d --name redis-stack -p 26379:6379 -p 8001:8001 redis/redis-stack
```

- **端口说明**：`26379:6379` 表示宿主机 26379 映射到容器内 Redis 的 6379，本机连接时使用 `redis://localhost:26379`；`8001:8001` 为 RedisInsight Web 管理界面，浏览器访问 http://localhost:8001 可查看键、执行命令等。
- **若报错「8001 address already in use」**：说明本机 8001 已被占用。可改用其他宿主机端口映射 RedisInsight，例如 8002：先执行 `docker rm -f redis-stack`（若容器存在），再执行 `docker run -d --name redis-stack -p 26379:6379 -p 8002:8001 redis/redis-stack`，然后通过 **http://localhost:8002** 访问 RedisInsight。Redis 仍为 `localhost:26379`，脚本无需修改。
- **验证是否启动成功**：
  - 本机若已安装 `redis-cli`：执行 `redis-cli -p 26379 PING`，返回 `PONG` 即表示 Redis 正常。
  - 或执行 `docker ps` 查看容器 `redis-stack` 状态为 `Up`；进入容器内执行 `redis-cli PING` 也可（Docker Desktop 中选中该容器 → Exec 标签页 → 输入 `redis-cli` 后输入 `PING`）。
- **运行本课程 Redis Stack 示例**：安装并启动后，在项目目录执行 `python3 Memory_RedisStackChatMessageHistory.py`（默认连接 `localhost:26379`），无需改代码。
- **停止与删除**（按需）：`docker stop redis-stack` 停止容器；`docker rm redis-stack` 删除容器（需先 stop）。若需保留数据，可使用 Docker 卷挂载，此处不展开。

#### 6.2.3 Docker 启动与宿主机连接

脚本（如 `Memory_RedisChatMessageHistory.py`）在**本机（宿主机）**运行，连接的是 `localhost:6379`。Redis 若跑在 Docker 容器内，必须做**端口映射**，宿主机才能访问。

- **检查端口映射**：执行 `docker port <容器名>`（如 `docker port docker-redis-1`），或 `docker ps` 查看该容器的 **PORTS** 列。出现 `0.0.0.0:6379->6379/tcp` 表示本机 6379 已映射到容器 6379。
- **未映射或希望使用本机 6379**：按顺序执行以下命令，用 `-p 6379:6379` 启动 Redis 容器：

```bash
docker stop docker-redis-1
docker rm docker-redis-1
docker run -d --name docker-redis-1 -p 6379:6379 redis
```

启动后执行 `docker port docker-redis-1` 确认输出为 `6379/tcp -> 0.0.0.0:6379`，再在本机运行脚本即可。

- **使用 Redis Stack 且映射到其他端口**（如 `-p 26379:6379`）：设置环境变量后运行脚本，例如 `export REDIS_URL=redis://localhost:26379`，或将 `REDIS_URL=redis://localhost:26379` 写入项目 `.env`。

#### 6.2.4 Redis 基本命令与查看对话历史

在 Docker 容器的 Exec 终端或本机 `redis-cli` 中，先执行 `redis-cli` 进入客户端（提示符变为 `127.0.0.1:6379>`），再使用以下常用命令。

| 命令                 | 说明                       | 示例                             |
| :------------------- | :------------------------- | :------------------------------- |
| `PING`               | 检测服务是否存活           | `PING` → 返回 `PONG`             |
| `SET key value`      | 设置字符串键值             | `SET mykey "hello"`              |
| `GET key`            | 获取字符串值               | `GET mykey`                      |
| `DEL key [key ...]`  | 删除一个或多个键           | `DEL mykey`                      |
| `KEYS pattern`       | 按模式查找键（慎用于生产） | `KEYS *`、`KEYS message_store:*` |
| `EXISTS key`         | 判断键是否存在             | `EXISTS mykey` → 1 或 0          |
| `TTL key`            | 查看键的剩余过期时间（秒） | `TTL mykey`，-1 表示永不过期     |
| `EXPIRE key seconds` | 设置键的过期时间           | `EXPIRE mykey 3600`              |
| `DBSIZE`             | 当前数据库的键数量         | `DBSIZE`                         |
| `FLUSHDB`            | 清空当前数据库（慎用）     | `FLUSHDB`                        |

对话历史案例中，LangChain 按 `session_id` 写入键，键名为 `message_store:<session_id>`。运行过案例（如 `session_id=user-001`）后，可在 `redis-cli` 中按下面步骤查看：

```
127.0.0.1:6379> KEYS *
1) "message_store:user-001"

127.0.0.1:6379> TYPE message_store:user-001
list

127.0.0.1:6379> LRANGE message_store:user-001 0 -1
1) "{\"type\": \"ai\", \"data\": {\"content\": \"你执行的是 Redis 的连通性测试命令...\"}}"
...
```

键类型为 list，每条元素为一则 JSON 消息（`type` 为 `human` 或 `ai`，`data.content` 为内容）。若 `DBSIZE` 为 1，可先 `KEYS *` 得到键名，再用 `TYPE` 确认类型，最后用 `LRANGE 键名 0 -1` 查看全部消息。

#### 6.2.5 环境验证

运行以下脚本确认 Redis 与 Python 包可用。

【案例源码】`案例与源码-4-LangGraph框架/07-memory/RedisEnvCheck.py`

[RedisEnvCheck.py](案例与源码-4-LangGraph框架/07-memory/RedisEnvCheck.py ":include :type=code")

#### 6.2.6 案例：Redis 对话历史

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_RedisChatMessageHistory.py`

[Memory_RedisChatMessageHistory.py](案例与源码-4-LangGraph框架/07-memory/Memory_RedisChatMessageHistory.py ":include :type=code")

#### 6.2.7 案例：使用 Redis Stack（可选）

与 6.2.6 逻辑相同，仅默认连接 **Redis Stack** 常用端口 `26379`，便于使用 RedisInsight 查看键与对话数据。需先启动 Redis Stack 容器，例如：

```bash
docker run -d --name redis-stack -p 26379:6379 -p 8001:8001 redis/redis-stack
```

![RedisInsight 界面示意](images/16/16-6-2-7-1.png)

【案例源码】`案例与源码-4-LangGraph框架/07-memory/Memory_RedisStackChatMessageHistory.py`

[Memory_RedisStackChatMessageHistory.py](案例与源码-4-LangGraph框架/07-memory/Memory_RedisStackChatMessageHistory.py ":include :type=code")

---

**本章小结：**

- **为什么需要记忆**：无记忆时每次请求独立，模型无法利用上一轮内容；多轮对话需要记忆组件存储上下文，实现「前言搭后语」。
- **实现原理**：[链](15-LCEL与链式调用.md)执行前从记忆组件**读取**历史并拼入[提示](13-提示词与消息模板.md)，执行后将本轮输入与输出**写回**记忆组件；记忆组件需具备读、写与存储能力。
- **推荐实现**：使用 **RunnableWithMessageHistory** 搭配 **BaseChatMessageHistory** 的实现类；**InMemoryChatMessageHistory** 适合进程内、重启即丢；**RedisChatMessageHistory** 配合 Redis（或 Redis Stack）可实现持久化，需配置 `redis` 包（如 5.3.1）与 `session_id` 区分会话。

**建议下一步：** 在本地跑通内存版与 Redis 版带历史对话案例，理解 `session_id` 与历史读写时机；接着学习 [第 17 章 Tools 工具调用](17-Tools工具调用.md)，让模型具备调用外部接口的能力；再学 [第 21 章 Agent 智能体](21-Agent智能体.md)，将记忆与工具结合实现多轮决策。
