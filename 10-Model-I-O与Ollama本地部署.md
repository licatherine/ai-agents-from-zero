# 10 - Model I/O 与 Ollama 本地部署

---

**本章课程目标：**

- 理解 LangChain 的 **Model I/O** 模块：输入提示（Prompt）、调用模型（Model）、输出解析（Parser）三件套。
- 掌握在 LangChain 中接入 **OpenAI 兼容接口**、**DeepSeek**、**通义千问** 等大模型的方式与选型。
- 学会使用 **Ollama** 在本地部署与运行大模型，并用 **LangChain** 调用本地模型。

**前置知识建议：** 已学习第 9 章「LangChain 概述与快速上手」，了解 LangChain 的定位与基本用法；具备 Python 环境与 API 调用基础。

---

## 1、Model I/O 大模型接口

### 1.1 官方文档与概述

Model I/O 是 LangChain 中与大模型交互的**核心模块**，官方文档可从这里入手：

- **Model I/O 总览**：https://docs.langchain.com/oss/python/langchain/models  
- **Chat 集成**：https://docs.langchain.com/oss/python/integrations/chat  

后续接入具体厂商时，可再查阅对应集成文档（如 OpenAI、DeepSeek、Ollama 等）。

### 1.2 Model I/O 是什么

**Model I/O** 的含义是：**标准化各类大模型的输入与输出**。它主要包含三部分：

- **输入模板**：把原始数据格式化成模型可接受的输入（如提示词模板）。
- **模型本身**：通过统一接口调用不同的大语言模型。
- **格式化输出**：从模型返回中提取信息，并按预定格式（如 JSON、结构化文本）输出。

![Model I/O 模块结构示意：输入、模型、输出三部分](images/10/10-1-2-1.jpeg)

> **说明**：上图为 Model I/O 的整体结构——左侧为**输入**（如 Prompt Template），中间为**模型调用**（LLM/Chat Model），右侧为**输出解析**（Output Parser）。三者组合即可完成「问→模型推理→结构化结果」的完整流程。

### 1.3 I/O 三件套

可以把 Model I/O 理解为三个固定环节，对应 LangChain 里的三类组件。

#### 1.3.1 输入提示（Format）

对应 **Prompts Template（提示词模板）**。作用包括：

- 将原始数据格式化成模型可处理的形式。
- 通过模板（如带占位符的字符串）插入变量，拼成完整问题再送入模型。

即：**管理大模型的输入**，避免在代码里手写拼接字符串。

#### 1.3.2 调用模型（Predict）

对应 **Models**。作用包括：

- 使用**统一接口**调用不同的大语言模型（OpenAI、DeepSeek、通义、Ollama 等）。
- 接收已格式化的问题，进行**预测或生成**，返回模型输出。

即：**屏蔽各厂商 API 差异**，用同一套写法切换模型。

#### 1.3.3 输出解析（Parse）

对应 **Output Parser**。作用包括：

- 从模型的**原始输出**中提取信息。
- 按**预先约定**的格式规范化结果，例如解析成 JSON、列表或自定义结构。

即：**把「模型说的话」变成「程序好用的数据结构」**。

#### 一句话小结

**Model I/O = 输入（Format）→ 处理（Predict）→ 输出（Parse）**，对应提示模板、模型调用、输出解析三步。

---

### 1.4 调用模型与模型分类

#### 1.4.1 模型在应用中的位置

一个 AI 应用的核心往往就是它所依赖的大语言模型。LangChain **不提供**任何 LLM，而是通过**第三方集成**把各平台模型接入到你的应用中，例如：OpenAI、Anthropic、Hugging Face、LlaMA、阿里通义、ChatGLM 等。

- **模型接口参考**：https://reference.langchain.com/python/langchain_core/language_models/

#### 1.4.2 LangChain 中的模型分类

LangChain 将大语言模型按用途分为多种类型，**实际开发中最常用的是「聊天对话模型」**（Chat Model），用于多轮对话、系统角色与用户消息等场景。

![LangChain 中模型分类：LLM、Chat、Embedding 等](images/10/10-1-4-1.gif)

> **说明**：图中一般会区分 **LLM**（纯文本补全）、**Chat Model**（多轮对话）、**Embedding**（向量化）等。本课程重点使用 **Chat Model**，与第 9 章的 `init_chat_model`、`ChatOpenAI` 等对应。

---

### 1.5 模型参数与返回

#### 1.5.1 标准化参数

在构建聊天模型（如 `init_chat_model`）时，LangChain 定义了一批**标准化参数**，便于在不同模型间保持相近的配置方式。

- **官方说明**：https://docs.langchain.com/oss/python/langchain/models#parameters  

常见参数包括：`model`、`temperature`、`max_tokens`、`api_key`、`base_url` 等。

![聊天模型常用标准化参数一览](images/10/10-1-5-1.jpeg)

![标准化参数的适用范围说明](images/10/10-1-5-2.jpeg)

> **说明**：上述标准化参数**并非对所有模型都生效**。通常仅对 LangChain 官方集成包（如 `langchain-openai`、`langchain-anthropic`）中的模型有完整支持；`langchain-community` 中的第三方模型可能不遵守这些规则，需以具体文档为准。

#### 1.5.2 示例：参数在代码中的用法

下面是一个在代码中设置模型参数的简单示例，便于理解「参数 → 模型行为」的对应关系。

![示例：在代码中设置模型参数](images/10/10-1-5-3.jpeg)

#### 1.5.3 模型返回：Message 组件

调用聊天模型后，返回的是**消息对象**，而不是裸字符串。在 LangChain 中，最常见的是 **AIMessage**，表示模型生成的一条回复。

![调用模型后返回的 AIMessage 结构](images/10/10-1-5-4.jpeg)

所有消息类型（如 `AIMessage`、`HumanMessage`、`SystemMessage`）一般都具有以下属性：

- **type**：消息类型（如 `ai`、`human`、`system`）。
- **content**：消息正文（通常是我们需要的文本内容）。
- **response_metadata**：模型返回的元数据（如 token 使用量、模型名等）。

![Message 的通用属性：type、content、response_metadata](images/10/10-1-5-5.jpeg)

在代码中通常通过 `response.content` 获取模型生成的文本，通过 `response.response_metadata` 获取额外信息。

---

### 1.6 接入大模型

LangChain 支持通过不同集成包接入多种大模型，以下按「OpenAI 兼容」「DeepSeek」「通义千问」「智谱」分别说明，并给出选型与示例。

**集成总览**：https://docs.langchain.com/oss/python/integrations/providers/overview  

#### 1.6.1 接入 OpenAI（及兼容接口）

- **文档**：  
  - https://docs.langchain.com/oss/python/integrations/providers/openai  
  - https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/

**openai.OpenAI 与 langchain_openai.ChatOpenAI 的区别与选型：**

| 对比项       | openai.OpenAI                 | langchain_openai.ChatOpenAI     |
| ------------ | ----------------------------- | ------------------------------- |
| **所属生态** | OpenAI 官方 Python SDK（`openai`） | LangChain 生态（`langchain-openai`） |
| **定位**     | 底层、纯粹的 API 调用         | LangChain 的「模型适配器」       |
| **特点**     | 轻量、直接、贴近原生 API      | 可与 Prompt、Chain、Agent、Memory 等组件无缝配合 |
| **选型建议** | 仅需简单调用、不打算用 LangChain 时 | 需要链式编排、多组件协作时       |

**结论**：两者无绝对优劣，按需求选择——**简单调用用官方 SDK**，**复杂工作流用 LangChain 的 ChatOpenAI**。

**示例一：使用 OpenAI 官方 SDK（如对接 DeepSeek 兼容接口）**

```python
# 需先安装：pip install openai
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello，你是谁？"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```

**示例二：使用 LangChain ChatOpenAI（如对接通义千问兼容接口）**

```python
# 需先安装：pip install langchain_openai
from langchain_openai import ChatOpenAI
import os

chatLLM = ChatOpenAI(
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 可按需更换，模型列表见阿里云文档
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
]

response = chatLLM.invoke(messages)
print(response.content)
```

**示例三：使用 init_chat_model 统一入口（LangChain 1.0 推荐）**

```python
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com"
)

print(model.invoke("你是谁").content)
```

#### 1.6.2 接入 DeepSeek

- **文档**：  
  - https://docs.langchain.com/oss/python/integrations/providers/deepseek  
  - https://docs.langchain.com/oss/python/integrations/chat/deepseek  

DeepSeek 提供官方 LangChain 集成包 `langchain-deepseek`，无需手动填写 `base_url`（已在 SDK 内封装）。

```python
import os
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("deepseek-api"),
)

print(model.invoke("你是谁？"))
```

#### 1.6.3 接入通义千问（阿里云百炼）

- **控制台与 API 文档**：https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654  

通义千问可通过 **OpenAI 兼容模式** 使用，即采用上面 1.6.1 中的 `ChatOpenAI` + `base_url` + 阿里云 API Key 的方式即可。

#### 1.6.4 接入智谱科技（课后拓展）

- **文档**：https://docs.langchain.com/oss/python/integrations/chat/zhipuai  

智谱 GLM 系列模型可通过 `langchain-zhipuai` 或官方文档中的方式接入，可作为课后练习自行尝试。

---

## 2、Ollama 本地大模型部署

本章介绍如何使用 **Ollama** 在本地运行大模型，并配合 **LangChain** 进行调用。与云端 API 相比，Ollama 适合本地开发、离线体验和隐私场景。

**LangChain 与 Ollama 集成文档**：https://docs.langchain.com/oss/python/integrations/chat/ollama  

### 2.1 Ollama 介绍与资源

#### 2.1.1 什么是 Ollama

Ollama 是一款用于在**本地**运行大语言模型的工具，支持一键拉取、运行和管理开源模型（如 LLaMA、Qwen、DeepSeek 等），无需自行配置推理引擎，对个人学习与原型开发非常友好。

![Ollama 产品介绍：本地运行大模型](images/10/10-2-1-1.png)

![Ollama 是什么：本地 LLM 运行工具](images/10/10-2-1-2.jpeg)

#### 2.1.2 官网与镜像

- **官网**：https://ollama.com  

![Ollama 官网入口](images/10/10-2-1-3.jpeg)

- **Docker Hub**：可拉取 Ollama 镜像，用于容器化部署。  
- **Ollama Hub / 模型库**：在官网或命令行中可浏览、拉取各类模型（如 `llama3`、`qwen:4b`、`deepseek-r1:14b` 等）。

#### 2.1.3 能做什么（产品定位）

Ollama 的典型用途包括：在本地运行开源大模型、通过命令行或 API 与模型对话、为 Coze、LangChain 等应用提供本地模型后端。

![Ollama 产品定位与能力](images/10/10-2-1-4.jpeg)

#### 2.1.4 去哪下载与安装

- **下载页**：https://ollama.com/download  

支持 Windows、macOS、Linux，按系统选择安装包即可。

![Ollama 下载入口与支持平台](images/10/10-2-1-5.jpeg)

![Ollama 官方下载页面](images/10/10-2-1-6.jpeg)

#### 2.1.5 怎么玩（基本使用方式）

安装完成后，可通过命令行拉取模型、启动对话，或开启本地 API 供其他程序（如 LangChain）调用。

![Ollama 基本使用方式概览](images/10/10-2-1-7.jpeg)

---

### 2.2 安装与配置

#### 2.2.1 自定义 Ollama 安装路径与模型存储目录

若希望将 Ollama 或模型文件安装到非默认目录（例如 D 盘或大容量磁盘），可先自定义安装路径，再设置**模型存储目录**。

![自定义 Ollama 安装路径的配置步骤](images/10/10-2-2-1.gif)

#### 2.2.2 设置模型存储目录（环境变量）

手动创建用于存放模型的目录（如 `D:\devSoft\Ollama\models`），然后新建系统环境变量：

- **变量名**：`OLLAMA_MODELS`  
- **变量值**：`D:\devSoft\Ollama\models`（按你的实际路径填写）

这样 Ollama 拉取的模型会保存到该目录，便于管理和迁移。

![设置 OLLAMA_MODELS 环境变量指定模型存储路径](images/10/10-2-2-2.jpeg)

#### 2.2.3 复制或迁移已有模型目录

若之前已在其他位置下载过模型，可将整个模型目录复制到上述 `OLLAMA_MODELS` 路径下，避免重复下载。

![将已有模型目录复制到 OLLAMA_MODELS 路径](images/10/10-2-2-3.jpeg)

---

### 2.3 常用命令

安装完成后，在终端中可使用以下命令（Windows 需在安装后打开新的终端以使环境变量生效）：

| 命令 | 说明 |
|------|------|
| `ollama pull llama3` | 下载指定模型（如 llama3）。 |
| `ollama run llama3` | 启动并进入该模型的交互对话。 |
| `ollama list` | 列出本机已下载的所有模型。 |
| `ollama rm llama3` | 删除指定模型以释放磁盘空间。 |
| `ollama cp llama3 my-llama3` | 本地复制或重命名模型。 |
| `ollama show llama3` | 查看模型详细信息（参数、大小等）。 |
| `ollama create my-model -f Modelfile` | 使用 Modelfile 构建自定义模型。 |
| `ollama serve` | 启动后台服务，供 API 调用（通常自动启动）。 |
| `ollama ps` | 查看当前正在运行的模型进程。 |
| `ollama stop llama3` | 停止正在运行的指定模型。 |

---

### 2.4 安装与验证模型（以千问、DeepSeek 为例）

#### 2.4.1 验证 Ollama 是否安装成功

- 查看版本：在终端执行 `ollama --version`。  
- 检查服务端口：Ollama 默认使用端口 **11434**。  
  - Windows：`netstat -ano | findstr 11434`  
  - Linux：`ps -ef | grep 8080` 或检查 11434 端口占用。

![验证 Ollama 安装与服务状态](images/10/10-2-4-1.jpeg)

#### 2.4.2 以通义千问、DeepSeek 为例运行模型

- **千问**（示例 4B 尺寸）：  
  ```bash
  ollama run qwen:4b
  ```  
- **DeepSeek R1**（示例 14B）：  
  ```bash
  ollama run deepseek-r1:14b
  ```  

进入对话后，可直接输入问题与模型交互。

![在 Ollama 中运行千问与 DeepSeek 模型](images/10/10-2-4-2.jpeg)

#### 2.4.3 退出交互对话

在对话界面输入 `/bye` 或使用快捷键（如 Ctrl+D）即可退出当前模型对话。

![退出 Ollama 模型交互对话](images/10/10-2-4-3.jpeg)

---

### 2.5 LangChain 整合 Ollama 调用本地大模型

在本地用 Ollama 跑通模型后，即可在 Python 中用 LangChain 通过 HTTP 调用本地 Ollama 服务，无需 API Key，适合本地开发与调试。

#### 2.5.1 环境要求与依赖

- 确保已安装**最新版 Ollama**，并已通过 `ollama run <模型名>` 或后台服务拉取过至少一个模型。  
- 安装 LangChain 的 Ollama 集成包与（可选）官方 Ollama Python 包：

```bash
pip install -qU langchain-ollama
pip install -U ollama
```

#### 2.5.2 示例代码：使用 ChatOllama

以下示例使用 `langchain_ollama` 的 `ChatOllama`，连接本机默认的 Ollama 服务（`http://localhost:11434`），并调用已拉取的模型（如 `qwen:4b` 或 `llama3`）。

```python
from langchain_ollama import ChatOllama

# 使用本机 Ollama 服务，模型名需与 ollama list 中的名称一致
llm = ChatOllama(
    model="qwen:4b",       # 或 "llama3"、"deepseek-r1:14b" 等
    base_url="http://localhost:11434",  # 默认即本机
    temperature=0.7,
)

response = llm.invoke("你好，请用一句话介绍你自己。")
print(response.content)
```

若需与 **Prompt 模板、Chain、Agent** 等结合，只需将上面的 `llm` 传入对应组件即可，与使用 `ChatOpenAI` 的方式一致。

---

**本章小结：**

- **Model I/O** 负责输入（Prompt 模板）、模型调用、输出解析三部分，是 LangChain 与各类大模型打交道的标准方式。  
- 接入云端模型时，可根据需求选择 **OpenAI SDK** 或 **LangChain ChatOpenAI / init_chat_model**；DeepSeek、通义千问等均有对应文档与示例。  
- **Ollama** 用于在本地一键运行开源大模型；通过 **langchain-ollama** 的 `ChatOllama` 即可在 LangChain 中调用本地模型，便于学习与离线使用。  

下一章将深入 **提示词（Prompt）与输出解析（Output Parser）** 的用法，与本章的 Model I/O 三件套形成完整闭环。
