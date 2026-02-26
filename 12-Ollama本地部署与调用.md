# 12 - Ollama 本地部署与调用

---

**本章课程目标：**

- 理解 **Ollama** 是什么、能做什么，以及如何获取程序与模型。
- 掌握 Ollama 的**安装与配置**（含自定义路径、模型存储目录）、**常用命令**及**安装与验证模型**的步骤。
- 学会使用 **LangChain**（`langchain-ollama` 的 `ChatOllama`）调用本地 Ollama 服务，无需 API Key，适合本地开发与离线使用。

**前置知识建议：** 已学习 [第 9 章 - LangChain 概述与架构](9-LangChain概述与架构.md)、[第 10 章 - LangChain 快速上手与 HelloWorld](10-LangChain快速上手与HelloWorld.md)、[第 11 章 - Model I/O 与模型接入](11-Model-I-O与模型接入.md)，了解 LangChain 的定位与模型调用方式。

---

## 1、Ollama 介绍与资源

本章介绍如何使用 **Ollama** 在本地运行大模型，并配合 **LangChain** 进行调用。与云端 API 相比，Ollama 适合本地开发、离线体验和隐私场景。

**LangChain 与 Ollama 集成文档**：https://docs.langchain.com/oss/python/integrations/chat/ollama

### 1.1 什么是 Ollama

Ollama 就是让你**在自己电脑上跑大模型**的一个免费、开源小工具。装好之后，输入一条命令就能下载并运行各种开源模型（比如 LLaMA、Qwen、DeepSeek），不用自己折腾显卡、环境，它会把模型和配置都打包好。适合自己学着玩、做小项目，或者没网的时候在本地用，对新手也比较友好。

![Ollama 产品介绍：本地运行大模型](images/12/12-1-1-1.png)

### 1.2 能做什么、怎么用

**能做什么**：在本地运行开源大模型；用命令行或网页与模型对话；把 Ollama 当作**本地 API 服务**，给 LangChain、Coze 等应用当「后端」，让它们调你电脑上的模型。

**怎么用**：安装完成后，用命令行**拉取模型**（如 `ollama pull qwen:4b`）、**启动对话**（如 `ollama run qwen:4b`），或**开启本地 API** 供其他程序调用。具体命令见 1.4 及第 3 节。

![Ollama 产品定位与能力](images/12/12-1-2-1.jpeg)

![Ollama 基本使用方式概览](images/12/12-1-2-2.jpeg)

### 1.3 从哪里下载（程序 + 模型）

- **下载 Ollama 程序**
  - 官网 / 下载页：https://ollama.com （入口）、https://ollama.com/download （直接下安装包），支持 Windows、macOS、Linux。
  - Docker：用容器部署时到 **Docker Hub** 拉 Ollama 镜像即可。
- **下载模型**
  - **Ollama Hub / 模型库**：在官网或命令行里选模型、下模型（如 `llama3`、`qwen:4b`、`deepseek-r1:14b`），执行 `ollama run 模型名` 时会自动从这里拉取。

![Ollama 下载入口与支持平台](images/12/12-1-3-1.jpeg)

### 1.4 安装后基本使用（命令行）

安装完成后，在终端执行：`ollama pull <模型名>` 拉取模型，`ollama run <模型名>` 进入对话；需要给 LangChain 等调用时，保持 Ollama 运行即可，默认会提供本地 API。详细步骤见 **第 2 节 安装与配置** 及后续小节。

---

## 2、安装与配置

### 2.1 自定义 Ollama 安装路径与模型存储目录

若希望将 Ollama 或模型文件安装到非默认目录（例如 D 盘或大容量磁盘），可先自定义安装路径，再设置**模型存储目录**。**Windows 下建议不要装到 C 盘**，以免程序与模型占满系统盘；安装时或安装前将安装路径与模型目录设到 D 盘等其它盘符即可。

![自定义 Ollama 安装路径的配置步骤](images/12/12-2-1-1.gif)

### 2.2 设置模型存储目录（环境变量）

手动创建用于存放模型的目录（如 `D:\devSoft\Ollama\models`），然后新建系统环境变量：

- **变量名**：`OLLAMA_MODELS`
- **变量值**：`D:\devSoft\Ollama\models`（按你的实际路径填写）

这样 Ollama 拉取的模型会保存到该目录，便于管理和迁移。

![设置 OLLAMA_MODELS 环境变量指定模型存储路径](images/12/12-2-2-1.jpeg)

### 2.3 复制或迁移已有模型目录

若之前已在其他位置下载过模型，可将整个模型目录复制到上述 `OLLAMA_MODELS` 路径下，避免重复下载。

![将已有模型目录复制到 OLLAMA_MODELS 路径](images/12/12-2-3-1.jpeg)

---

## 3、常用命令

安装完成后，在终端中可使用以下命令（Windows 需在安装后打开新的终端以使环境变量生效）：

| 命令                                  | 说明                                        |
| ------------------------------------- | ------------------------------------------- |
| `ollama pull llama3`                  | 下载指定模型（如 llama3）。                 |
| `ollama run llama3`                   | 启动并进入该模型的交互对话。                |
| `ollama list`                         | 列出本机已下载的所有模型。                  |
| `ollama rm llama3`                    | 删除指定模型以释放磁盘空间。                |
| `ollama cp llama3 my-llama3`          | 本地复制或重命名模型。                      |
| `ollama show llama3`                  | 查看模型详细信息（参数、大小等）。          |
| `ollama create my-model -f Modelfile` | 使用 Modelfile 构建自定义模型。             |
| `ollama serve`                        | 启动后台服务，供 API 调用（通常自动启动）。 |
| `ollama ps`                           | 查看当前正在运行的模型进程。                |
| `ollama stop llama3`                  | 停止正在运行的指定模型。                    |

---

## 4、安装与验证模型（以千问、DeepSeek 为例）

### 4.1 验证 Ollama 是否安装成功

- 查看版本：在终端执行 `ollama --version`。
- 检查服务端口：Ollama 默认使用端口 **11434**。
  - Windows：`netstat -ano | findstr 11434`
  - macOS：`lsof -i :11434` 或 `netstat -an | grep 11434`
  - Linux：`lsof -i :11434` 或 `netstat -tlnp | grep 11434`

> **lsof 和 netstat 的区别**：**lsof**（list open files）列出打开的文件，在 Unix/macOS 里网络连接也算一种文件，所以 `lsof -i :11434` 能直接看到**哪个进程**（PID、进程名）占用了该端口，适合回答「谁在用这个端口」。**netstat**（network statistics）显示网络连接、监听端口等**连接状态**（如 LISTEN、ESTABLISHED），偏重「端口是否在监听、有哪些连接」；要看进程通常需加参数（如 Linux 的 `-p`）。查 Ollama 是否在跑时，用 **lsof -i :11434** 最直接；两者二选一即可。

![验证 Ollama 安装与服务状态](images/12/12-4-1-1.jpeg)

### 4.2 以通义千问、DeepSeek 为例运行模型

执行 `ollama run <模型名>` 时，**若本地还没有该模型，Ollama 会先自动拉取（pull）再启动对话**，无需先单独执行 `ollama pull`；若已拉取过则直接进入对话。

- **千问**（示例 4B 尺寸）：
  ```bash
  ollama run qwen:4b
  ```
- **DeepSeek R1**（示例 14B）：
  ```bash
  ollama run deepseek-r1:14b
  ```

进入对话后，可直接输入问题与模型交互。

![在 Ollama 中运行千问与 DeepSeek 模型](images/12/12-4-2-1.jpeg)

### 4.3 退出交互对话

在对话界面输入 `/bye` 或使用快捷键（如 Ctrl+D）即可退出当前模型对话。

![退出 Ollama 模型交互对话](images/12/12-4-3-1.jpeg)

---

## 5、LangChain 整合 Ollama 调用本地大模型

在本地用 Ollama 跑通模型后，即可在 Python 中用 LangChain 通过 HTTP 调用本地 Ollama 服务，无需 API Key，适合本地开发与调试。

### 5.1 环境要求与依赖

- 确保已安装**最新版 Ollama**，并已通过 `ollama run <模型名>` 或后台服务拉取过至少一个模型。
- 安装 LangChain 的 Ollama 集成包与（可选）官方 Ollama Python 包：

```bash
pip install -qU langchain-ollama
pip install -U ollama
```

### 5.2 示例代码：使用 ChatOllama

以下示例使用 `langchain_ollama` 的 `ChatOllama`，连接本机默认的 Ollama 服务（`http://localhost:11434`），并调用已拉取的模型（如 `qwen:4b` 或 `llama3`）。

【案例源码】`案例与源码-4-LangGraph框架/03-ollama/LangChain_Ollama.py`

[LangChain_Ollama.py](案例与源码-4-LangGraph框架/03-ollama/LangChain_Ollama.py ":include :type=code")

若需与 **Prompt 模板、Chain、Agent** 等结合，只需将上面的 `llm` 传入对应组件即可，与使用 `ChatOpenAI` 的方式一致。

---

**本章小结：**

- **Ollama** 用于在本地一键运行开源大模型；安装后通过 `ollama pull`、`ollama run` 等命令拉取与运行模型，默认提供本地 API（端口 11434）。
- **安装与配置**：可自定义安装路径与模型存储目录（环境变量 `OLLAMA_MODELS`），便于管理与迁移。
- **LangChain 整合**：通过 **langchain-ollama** 的 `ChatOllama` 在 LangChain 中调用本地模型，无需 API Key，与 `ChatOpenAI` 用法一致，可接入 Prompt、Chain、Agent 等组件。

**建议下一步：** 学习 [第 13 章 - 提示词与消息模板](13-提示词与消息模板.md)、[第 14 章 - 输出解析器](14-输出解析器.md)，与第 11 章的 Model I/O 三件套形成完整的「输入 → 模型 → 输出解析」闭环。
