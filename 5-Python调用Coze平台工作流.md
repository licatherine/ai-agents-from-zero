# 5 - Python 调用 Coze 平台工作流

本章偏**实操部署**：学会用 API 和 Python 调用你在 Coze 上已搭建好的工作流，把「扣子里的工作流」变成「代码里可调用的服务」。整体流程与 [第 4 章 - Python 调用 Dify 工作流](4-Python调用Dify平台工作流.md) 类似，只是平台接口和 SDK 不同。

---

**本章课程目标：**

- 知道调用前需在 Coze 中**发布 API**，并在「API 调试」里拿到 **workflow_id**、**app_id** 和 **API Key（token）**。
- 理解请求方式：POST、URL（如 `api.coze.cn/v1/workflow/stream_run`）、请求头与请求体（**parameters** 对应工作流输入变量）；能看懂调试结果中的 PING（心跳）、Message（携带 content）、Done（结束）含义。
- 能用 **cozepy** 官方 SDK 在本地跑通流式调用：安装 `cozepy`、配置 token 与 `COZE_CN_BASE_URL`、在 `stream()` 中传入 **parameters**，并会处理 MESSAGE / ERROR / INTERRUPT 等事件。

**学习建议：** 按文档顺序做即可：发布 API → 进入调试拿到 workflow_id、app_id、API Key → 在调试里用 curl 或界面跑通 → 安装 cozepy、拷贝示例代码并补全 **parameters** 后本地运行。国内使用需选 **api.coze.cn**（示例中已用 `COZE_CN_BASE_URL`）。

---

## 1. 发布 API

Coze 的 API 功能需要通过应用发布功能启用。

![](images/5/5-1-1.png)

## 2. 调试

### 2.1 入口

发布成功后，在工作流画布页面可以看到 API 调试入口。

![](images/5/5-2-1-1.png)

### 2.2 查看工作流 ID 和应用 ID

通过上述入口进入 API playground，选择右侧的 Shell 请求方式。

界面中会显示**工作流 ID**（字段名 workflow_id）和**应用 ID**（字段名 app_id）的取值。

![](images/5/5-2-2-1.png)

### 2.3 授权 API Key

左侧窗口向下滑动，可以看到 token（即 API Key），点击「授权」按钮。

![](images/5/5-2-3-1.png)

点击「授权」后，将自动生成并填充 API Key。

![](images/5/5-2-3-2.png)

右侧的 Shell 命令窗口会同步更新。

### 2.4 添加参数

请求体中的 **parameters** 对象用于向工作流传递参数。

该对象的每个属性对应于工作流的一个输入变量。

![](images/5/5-2-4-1.png)

### 2.5 运行

可以直接点击 Shell 命令窗口右上角的「运行」按钮。

![](images/5/5-2-5-1.png)

### 2.6 完整命令

```sh
curl -X POST 'https://api.coze.cn/v1/workflow/stream_run' \
-H "Authorization: Bearer {api_key}" \
-H "Content-Type: application/json" \
-d '{
  "workflow_id": "{workflow_id}",
  "app_id": "{app_id}",
  "parameters": {
    "link": "https://www.bilibili.com/video/BV1H48CzUEhj/?spm_id_from=333.337.search-card.all.click&vd_source=88c7b17e5559e5e21cf45e7e873d1459" #希望复刻的视频链接
  }
}'
```

> **说明**：上述 `link` 为示例参数（希望复刻的视频链接），请按实际工作流输入替换。JSON 标准不支持 `#` 注释，请求体中请勿写入注释。

### 2.7 运行结果

![](images/5/5-2-7-1.png)

右下角可以看到运行结果。

- **200**：HTTP 状态码，表示请求成功、通信正常。
- **PING**：心跳响应，用于维持连接。
- **Message**：携带数据的响应类型。

![](images/5/5-2-7-2.png)

此处的 **content** 字段即为工作流在该步骤的输出内容。

**Done** 为最后一个响应，表示工作流执行完毕，通常出现在所有 Message 之后。

![](images/5/5-2-7-3.png)

> **可这样记：** 流式返回时，**PING** 是保活心跳，**Message** 才是真正带 `content` 的数据，**Done** 表示流结束。写代码时一般只关心 Message 和 Done（或错误类型），心跳可忽略。

## 3. 通过 Python 代码调用工作流

### 3.1 官方提供的 Python 示例代码

Coze 平台提供了用于调用工作流 API 的 Python 示例代码。

![](images/5/5-3-1-1.png)

### 3.2 示例源码

```python
"""
This example describes how to use the workflow interface to stream chat.
"""

import os
# Our official coze sdk for Python [cozepy](https://github.com/coze-dev/coze-py)
from cozepy import COZE_CN_BASE_URL

# Get an access_token through personal access token or oauth.
coze_api_token = '{API_KEY}'
# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
coze_api_base = COZE_CN_BASE_URL

from cozepy import Coze, TokenAuth, Stream, WorkflowEvent, WorkflowEventType  # noqa

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# Create a workflow instance in Coze, copy the last number from the web link as the workflow's ID.
workflow_id = '{WORKFLOW_ID}'


# The stream interface will return an iterator of WorkflowEvent. Developers should iterate
# through this iterator to obtain WorkflowEvent and handle them separately according to
# the type of WorkflowEvent.
def handle_workflow_iterator(stream: Stream[WorkflowEvent]):
    for event in stream:
        if event.event == WorkflowEventType.MESSAGE:
            print("got message", event.message)
        elif event.event == WorkflowEventType.ERROR:
            print("got error", event.error)
        elif event.event == WorkflowEventType.INTERRUPT:
            handle_workflow_iterator(
                coze.workflows.runs.resume(
                    workflow_id=workflow_id,
                    event_id=event.interrupt.interrupt_data.event_id,
                    resume_data="hey",
                    interrupt_type=event.interrupt.interrupt_data.type,
                )
            )


handle_workflow_iterator(
    coze.workflows.runs.stream(
        workflow_id=workflow_id
    )
)
```

将源码粘贴到 PyCharm。

### 3.3 在本地 PyCharm 中运行前，需要先安装 cozepy：

```sh
pip install cozepy
```

代码粘贴并运行后，若缺少参数会报错：

![](images/5/5-3-3-1.png)

需要补充代码：

![](images/5/5-3-3-2.png)

**处理方式**：在从 Coze 平台拷贝的代码基础上，为 `stream()` 调用增加 **parameters** 参数：

```python
handle_workflow_iterator(
    coze.workflows.runs.stream(
        workflow_id=workflow_id,
        parameters={
            "link": "https://www.bilibili.com/video/BV1S2421P788/?share_source=copy_web&vd_source=8d04b2c1b7fd20888b03c20e99f26dc0"   # 替换成实际需要的链接
        }
    )
)
```

### 3.4 最终代码

```python
"""
This example describes how to use the workflow interface to stream chat.
"""

import os
# Our official coze sdk for Python [cozepy](https://github.com/coze-dev/coze-py)
from cozepy import COZE_CN_BASE_URL

# Get an access_token through personal access token or oauth.
coze_api_token = 'cztei_hXYOqnustyYyhrSuGFl4tgcxJ9E2KjYLPnHvcEcoWRwWvujWU0sPqka8xyQ1wsCyi'
# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
coze_api_base = COZE_CN_BASE_URL

from cozepy import Coze, TokenAuth, Stream, WorkflowEvent, WorkflowEventType  # noqa

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# Create a workflow instance in Coze, copy the last number from the web link as the workflow's ID.
workflow_id = '7537267958432858127'


# The stream interface will return an iterator of WorkflowEvent. Developers should iterate
# through this iterator to obtain WorkflowEvent and handle them separately according to
# the type of WorkflowEvent.
def handle_workflow_iterator(stream: Stream[WorkflowEvent]):
    for event in stream:
        if event.event == WorkflowEventType.MESSAGE:
            print("got message", event.message)
        elif event.event == WorkflowEventType.ERROR:
            print("got error", event.error)
        elif event.event == WorkflowEventType.INTERRUPT:
            handle_workflow_iterator(
                coze.workflows.runs.resume(
                    workflow_id=workflow_id,
                    event_id=event.interrupt.interrupt_data.event_id,
                    resume_data="hey",
                    interrupt_type=event.interrupt.interrupt_data.type,
                )
            )


handle_workflow_iterator(
    coze.workflows.runs.stream(
        workflow_id=workflow_id,
        parameters={
            "link": "https://www.bilibili.com/video/BV1S2421P788/?share_source=copy_web&vd_source=8d04b2c1b7fd20888b03c20e99f26dc0"   # 替换成实际需要的链接
        }
    )
)
```

### 3.5 运行

运行结果如下：

![](images/5/5-3-5-1.png)

## 4. 通过平台运行

![](images/5/5-4-1.png)

---

**章节思考题：**

1. `workflow_id`、`app_id`、API Key 在 Coze 工作流调用里分别承担什么作用？

   **答案：** `workflow_id` 用来指定具体执行哪个工作流，`app_id` 用来标识所属应用或 Bot 上下文，API Key 负责身份认证和调用授权。三者分别对应目标对象、应用归属和安全校验。

2. `parameters` 为什么是调用 Coze 工作流时最容易出错的一层？

   **答案：** 因为 `parameters` 必须和平台里定义的入参名称、类型、结构严格对齐，一旦字段名错、类型不对或层级不匹配，工作流就会拿不到值或直接报错。它往往不是“看起来差不多就行”，而是“必须完全一致”。

3. 为什么流式事件里要单独处理 ERROR 和 INTERRUPT，而不是只打印正文？

   **答案：** 因为 ERROR 和 INTERRUPT 不是普通正文，而是决定这次执行是否真的成功完成的关键信号。只打印正文会让你误以为任务结束了，实际可能已经中断、等待人工确认或直接失败。

4. 如果你要把 Coze 工作流接入一个 Python 服务端接口，你会如何设计“参数接收 -> 调用工作流 -> 返回结果”这条链路？

   **答案：** 我会把链路设计成三步：先在接口层校验并接收业务参数，再在服务层把参数映射成 Coze 工作流需要的 `parameters` 发起调用，最后统一解析正常结果、异常事件和中断状态，返回给前端一致的响应结构。

5. 当调用结果不符合预期时，你会如何利用平台调试信息和代码日志一起定位问题？

   **答案：** 先在平台调试页确认每个入参是否进到了正确节点，再看节点执行结果和报错位置；同时在代码侧记录请求体、事件流和最终返回。把平台内视角和代码外视角对起来，问题通常很快就能定位到是参数映射、流程逻辑还是接口调用层。

**本章小结：**

- **调用前提**：Coze 工作流要先发布 API，并确认 `workflow_id`、`app_id` 和 API Key 都已拿到。
- **调用核心**：`cozepy` 帮你封装了请求与事件流处理，但真正最关键的仍然是 `parameters` 是否和工作流输入变量一一对齐，以及 `workflow_id`、`app_id` 是否指向了正确对象。
- **事件流不要只看正文**：Coze 的流式执行除了正常输出，还可能出现 ERROR、INTERRUPT 等事件。学这章时，重点不是“能打印出一段文本”，而是能正确理解一次工作流执行到底是成功完成、等待干预，还是中途失败。
- **调试重点**：与 Dify 类似，平台调试和代码调用要相互印证；一旦结果异常，优先看参数映射、事件流、平台侧调试输出和本地日志是否能互相对上。
- 从掌握结果看，学完本章后，你至少应该：知道 Coze 工作流在被代码调用前，需要先发布 API，并拿到 `workflow_id`、`app_id`、API Key；能理解 Coze 流式调用里的几类常见事件，以及 `parameters` 与工作流输入变量之间的对应关系；能用 `cozepy` 跑通一次最小调用，并从事件流里区分正常输出、错误和中断。

**建议下一步：** 如果你想继续走 Coze 本地化和私有化路线，可以看 [第 6 章 Coze 的 Windows 平台部署](6-Coze的Windows平台部署.md)；如果你想从平台调用进一步过渡到代码框架主线，也可以回到 [第 9 章 LangChain 概述与架构](9-LangChain概述与架构.md) 开始进入 LangChain 学习。
