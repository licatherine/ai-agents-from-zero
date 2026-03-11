# 19 - RAG 检索增强生成

---

**本章课程目标：**

- 理解 **RAG（检索增强生成）** 是什么、为何需要、与微调等方案的关系，以及标准流程（索引 + 检索两阶段）。
- 掌握 LangChain 中 **文档加载器**、**文本分割器**、**向量存储与检索** 等 RAG 组件的用法（向量化与向量库基础见 [第 18 章](18.md)）。
- 完成「文档加载 → 分割 → 向量化存储 → 相似检索 → 与大模型结合回答」的完整 RAG 案例（如基于 alibaba-java.docx 的智能运维助手）。

**前置知识建议：** 已学习 [第 18 章 - 向量化、向量数据库与 Embedding 实战](18.md)，掌握向量、向量库、Embedding 与 Redis 存取的用法；已学习 [第 9 章](9-LangChain概述与架构.md)、[第 10 章](10-LangChain快速上手与HelloWorld.md)。对 [1-3 RAG、微调、续训与智能体](1-3-RAG、微调、续训与智能体.md) 有初步了解更佳。

**学习建议：** 先建立「先检索、再生成」的直觉，再按「索引阶段」与「检索阶段」理解流程；然后动手跑通文档加载、分割与综合案例，把第 18 章的 Embedding 与向量库串联起来。

---

## 1、RAG 概述

### 1.1 是什么

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种在生成回答前，先从外部知识库**检索**相关材料，再把这些材料作为上下文交给大模型**生成**答案的技术。大模型的知识受限于训练数据，若希望它使用**领域知识或私有数据**，通常可以：

- 使用 **RAG**（本节重点）；
- 使用**微调**；
- **RAG + 微调** 结合。

用更直白的话说：**RAG 就是「先从你的数据里找出和问题相关的内容，把这些内容塞进提示里，再一起发给大模型」**。这样模型在回答时手头有真实材料可参考，更容易答得准、少瞎编（即降低「幻觉」）。和「直接问模型」相比，RAG 相当于先查资料再作答；和「用你的数据微调模型」相比，RAG 不改模型参数，只改每次提问时附带的「参考资料」，更适合知识经常更新或不想动模型的场景。三者可以组合使用：例如用 RAG 补实时文档，用微调统一话术或格式。

**为何需要 RAG — 「幻觉」问题：**  
大模型可能「已读乱回」「已读不回」或「似是而非」。RAG 通过「先查资料再回答」，用真实文档约束生成，减轻幻觉、并便于引用来源。

**核心设计理念（一句话）：**  
RAG 就像给 AI 大模型装上了「实时百科大脑」：先查资料再回答，让 AI 摆脱单纯依赖训练数据的「知识遗忘和幻觉回复」困境。  
**通俗比喻：** 类似考试时允许查小抄——先检索再作答。

### 1.2 能干什么

通过引入**外部知识源**增强 LLM 的产出：在生成前从指定知识库检索相关内容，再基于这些内容生成更准确、更贴合上下文的回答，尤其适合企业文档、产品手册、最新信息等场景。

### 1.3 怎么玩：索引 + 检索两阶段

RAG 流程可拆成两个阶段：**索引（Index）** 和 **检索（Retrieval）**。

![RAG 整体流程：索引阶段与检索阶段](images/18/image144.png)

**索引阶段：** 把原始文档加载、分割、向量化后写入向量数据库，建立「文档片段 ↔ 向量」的索引。（向量化与向量库用法见 [第 18 章](18.md)。）

具体来说，做**向量检索**时，索引阶段一般会：先对文档做清洗与整理，必要时补上**元数据**；再把文档**切分成较小片段（分块）**；用嵌入模型把这些片段转成向量；最后写入**向量数据库**。这样检索阶段才能「按相似度找片段」。索引通常可以**离线**完成，不用等用户——例如用定时任务（如每周一次）对公司文档做一次重索引，或由单独的程序专门负责建库；这样主应用只负责「查」。但若你的场景里用户会**上传自己的文档**并希望马上能被 LLM 用到，索引就要**在线**执行，并接到主应用里，在用户上传后立刻完成分块、向量化与写入。

> **元数据（metadata）**：与文档或片段**绑定的附加信息**，本身不是正文内容。例如来源文件路径（`source`）、页码（`page`）、标题、作者、日期等。在 RAG 里：**正文（page_content）** 会参与向量化与相似度检索；**元数据**一般不参与向量计算，用来做**过滤**（如只查某本书）、**标注来源**（如回答里注明「见《XX》第 3 页」）或展示。LangChain 的 `Document` 即由 `page_content`（正文）和 `metadata`（字典）组成。

下图把上述「索引阶段」拆成一条流水线：**原始文档（Document）** 先进入 **文本分割器（Text Splitter）** 切成多段 **片段（Segments）**；每个片段再经 **嵌入模型（Embedding Model）** 转成 **向量（Embeddings）**；最后把这些片段与对应向量一并存入 **向量库（Embedding Store）**，供检索阶段按相似度查询。

![索引阶段细节示意](images/18/image146.jpeg)

**检索阶段：** 用户提问 → 问题向量化 → 在向量库中做相似性检索 → 取回相关片段 → 与问题一起组成上下文交给大模型 → 生成最终答案。

![检索阶段：查询向量化、相似检索、上下文组装与生成](images/18/image147.jpeg)

---

## 2、RAG 文本处理核心知识

### 2.1 LangChain 组件与标准流程

LangChain 为 RAG 提供了文档加载、分割、嵌入、向量存储、检索等组件，可按下述标准流程搭建应用。这些组件在 RAG 里的大致分工如下：

| 组件               | 作用                                                                                  | 常用类/说明                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **文档加载器**     | 从各种格式（PDF、Word、Markdown、文件等）把文档读进来，变成 LangChain 里的 Document。 | `Document`（文档组件）、`UnstructuredPDFLoader`（PDF 加载器）、`UnstructuredFileLoader`（文件加载器）、`UnstructuredMarkdownLoader`（Markdown 加载器） 等。 |
| **文档分割器**     | 把大文档切成较小片段（分块），便于后续嵌入和检索，也避免超出模型上下文长度。          | 常用 **RecursiveCharacterTextSplitter**（按字符递归切分，尽量保持语义完整）。                                                                               |
| **文本嵌入模型**   | 把文本转成数值向量（Embedding），便于用相似度比较语义。                               | 如 `OpenAIEmbeddings`、`HuggingFaceEmbeddings`；本教程多用阿里百炼等（见第 18 章）。                                                                        |
| **向量数据库组件** | 把生成的向量（及元数据）存进向量库，并支持按相似度快速检索。                          | 抽象为 **VectorStore**，具体有 Chroma、Pinecone、FAISS、Redis 等不同实现类。                                                                                |
| **文本检索器**     | 用户提问时，把问题向量化后在向量库里做相似检索，返回最相关的文档片段。                | 常用 **VectorStoreRetriever**，背后连到某个 VectorStore。                                                                                                   |

**RAG 标准流程简述（面试题）：**

1. **准备阶段**：用**文档加载器**加载各类格式文档，得到 LangChain 的 **Document** 对象。
2. **分割**：用**文本分割器**按规则将文档切分为适当大小的片段。
3. **向量化与存储**：用**嵌入模型**将片段转为向量，通过**向量数据库**写入存储（见 [第 18 章](18.md)）。
4. **使用阶段**：用户提问 → 问题向量化 → **检索器**在向量库中做相似性检索，返回相关片段。
5. **生成**：将检索到的片段填入**提示词模板**作为上下文，与问题一起发给大模型，由模型「阅读—理解—整合—生成」后返回答案。
6. **总结**：RAG 的核心价值是让生成模型在**检索到的外部知识**上再做一次加工，输出更连贯、准确且可引用的回答。

![RAG 从检索到生成的完整数据流](images/18/image149.gif)

### 2.2 文档加载器（Document Loaders）

- **官方文档**：https://docs.langchain.com/oss/python/integrations/document_loaders

文档加载器负责把**各种格式**的文档（TXT、PDF、Word、Markdown、JSON、CSV 等）转成 **Document** 对象。各加载器用法不同，但都继承自 **BaseLoader**，统一提供 **load()** 方法；**load()** 返回 **Document 列表**。

**常用加载器与支持格式**：不同加载器对应不同数据源或文件类型，例如 **CSVLoader** 读 CSV，**JSONLoader** 读 JSON，**BSHTMLLoader** 读 HTML；**Unstructured** 系列可支持多种常见文件类型（详见官方文档）；**DoclingLoader**、**PolarisAIDataInsightLoader** 等也可处理多种格式。按你的文件类型选对应加载器即可。

<strong>统一接口</strong>：所有文档加载器都实现 <strong>BaseLoader</strong> 接口，从 Slack、Notion、Google Drive 或本地文件等读入后，统一转成 LangChain 的 Document，便于后续统一处理。常用方法有 <strong>load()</strong>（一次性加载全部文档）和 <strong>lazy_load()</strong>（流式加载，适合大文件或大批量数据）。

**继承关系**：所有加载器都从最顶层的 **BaseLoader** 派生。下图把这种关系画成树状：例如 **UnstructuredBaseLoader** 负责「非结构化文件」一类，其下再有 **UnstructuredFileLoader**，再派生出 HTML、Markdown、Excel 等具体加载器；**BasePDFLoader** 专门管 PDF，下面有 **PyPDFLoader** 等；还有 **JSONLoader** 等与 BaseLoader 直接相连。同一分支下的类共享基类，便于扩展；实际使用时按文件格式选对具体加载器即可。

![文档加载器继承关系示意](images/18/image152.jpeg)

![文档加载器与 Document 的关系](images/18/image153.jpeg)

**Document 类**：无论从何种来源加载，最终都解析为 **Document**，主要属性：

- **page_content**：文档内容（字符串）。
- **metadata**：元数据（字典），如 source、文件名等。

示例（加载 TXT 后）：  
`[Document(metadata={'source': 'assets/sample.txt'}, page_content='LangChain 是一个用于构建基于大语言模型（LLM）应用的开发框架……')]`

**文档加载器案例：** 下面按格式列出同目录下的加载示例。运行后终端会打印出 **Document 列表**：每个元素是一个 Document，包含 <strong>page_content</strong>（文档正文字符串）和 <strong>metadata</strong>（如 <code>source</code> 文件路径等）；后续分割、向量化都基于这一结构。

- **TXT（纯文本）**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadTxtDemo.py`

[RagLoadTxtDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadTxtDemo.py ":include :type=code")

- **PDF**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadPdfDemo.py`

[RagLoadPdfDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadPdfDemo.py ":include :type=code")

- **Word（.docx）**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadDocDemo.py`

[RagLoadDocDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadDocDemo.py ":include :type=code")

- **Markdown**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadMarkdownDemo.py`

[RagLoadMarkdownDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadMarkdownDemo.py ":include :type=code")

- **JSON**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadJsonDemo.py`

[RagLoadJsonDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadJsonDemo.py ":include :type=code")

- **CSV**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadCSVDemo.py`

[RagLoadCSVDemo.py](案例与源码-4-LangGraph框架/10-rag/docloads/RagLoadCSVDemo.py ":include :type=code")

### 2.3 文本分割器（Text Splitters）

- **官方文档**：https://docs.langchain.com/oss/python/integrations/splitters

**为什么要分割：**  
文档往往很长，一方面大模型有上下文与 Token 限制，无法「一口吞」；另一方面整篇编码成本高且易超限，切成片段再向量化更利于检索与成本控制。

LangChain 提供多种分割策略，多数继承自 **TextSplitter**，常用方法：

- <strong>split_text()</strong>：将字符串分割成字符串列表。
- <strong>split_documents()</strong>：将 Document 列表分割成更小的 Document 列表。
- <strong>create_documents()</strong>：由字符串列表创建 Document 列表。

**常用分割器与适用场景**：按内容或需求选一种即可；本教程案例多用 **RecursiveCharacterTextSplitter**。

| 分割器                             | 作用                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **RecursiveCharacterTextSplitter** | 递归按字符切分：先尝试按段落、句子、词等大单位切，必要时再按单字符，尽量让每块语义完整。通用场景首选。 |
| **CharacterTextSplitter**          | 按你指定的分隔符（如换行、空格、逗号）切分，简单可控。                                                 |
| **MarkdownHeaderTextSplitter**     | 按 Markdown 标题（如 `##`）切分，适合 Markdown 文档，每块对应一个标题下的内容。                        |
| **HTMLHeaderTextSplitter**         | 按 HTML 标题（如 `<h1>`、`<h2>`）切分，适合 HTML 文档。                                                |
| **PythonCodeTextSplitter**         | 按 Python 代码结构（函数、类等）切分，适合代码类文本。                                                 |
| **TokenTextSplitter**              | 按 **token 数量**切分，便于严格控制每块不超模型 token 上限（第二常用）。                               |

**RecursiveCharacterTextSplitter（递归字符文本切分器）** 常用参数：

| 参数                   | 含义               | 说明                                                                                                                                                    |
| ---------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **chunk_size**         | 单块最大长度       | 每块最多允许多长，由 **length_function** 决定怎么算（默认按字符数）。设多大要结合大模型上下文上限，例如 GPT-3.5 约 3000 token，对应字符数需按模型换算。 |
| **chunk_overlap**      | 块与块的重叠长度   | 相邻两块之间重叠多少字符，用来保留上下文、避免把一句话拦腰截断。必须小于 chunk_size，常见设为 chunk_size 的 10%～ 20%。（常用）                         |
| **separators**         | 递归切分用的分隔符 | 按优先级的一组分隔符（如先换行、再句号、再空格）。先按第一个分隔符切，某块仍超长再按下一个切，都切不到合适大小时会强制按长度切。可按领域自定义分隔符。  |
| **length_function**    | 长度怎么算         | 默认用 `len` 即按字符数；可改为按 **token** 数（如用 tiktoken），更贴合模型输入限制。                                                                   |
| **keep_separator**     | 是否保留分隔符     | 默认 `False`，切完后分隔符丢掉；设为 `True` 则把分隔符留在块末尾，便于保留原文结构。                                                                    |
| **is_separator_regex** | 分隔符是否当正则用 | 默认 `False`，分隔符当普通字符串匹配；设为 `True` 则按正则表达式解析，适合复杂切分规则。                                                                |

入门时重点用 **chunk_size**、**chunk_overlap**、**length_function** 即可；其余用默认值。

**分割纯文本：**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveTextSplitter.py`

[RecursiveTextSplitter.py](案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveTextSplitter.py ":include :type=code")

**分割纯文本（V2：用代码验证「切完再拼回去 = 原文」，可选）**  
与上面相同：都用 split_text 切同一段文本。区别在于：(1) V2 用手动 `[Document(page_content=text) for text in texts]` 转 Document，而上面用 `create_documents`；(2) **V2 多了一段「把切出的块按 chunk_overlap 剔除重叠再拼成一条字符串」的代码**，并打印「拼接结果是否等于原文」。跑一遍就能直观看到：有重叠的切分不会丢内容，拼回去和原文一致。入门以 RecursiveTextSplitter 为主即可，V2 用于理解重叠与完整性。

【案例源码】`案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveTextSplitterV2.py`

[RecursiveTextSplitterV2.py](案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveTextSplitterV2.py ":include :type=code")

**分割 Document 对象（先加载再分割）：**

【案例源码】`案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveDocumentSplitter.py`

[RecursiveDocumentSplitter.py](案例与源码-4-LangGraph框架/10-rag/textsplit/RecursiveDocumentSplitter.py ":include :type=code")

---

## 3、RAG 综合案例：智能运维助手

### 3.1 需求说明

某系统涉及后续自动化维护，需要根据**响应码**让大模型辅助运维人员定位问题。希望实现：**AI 智能运维助手**，根据提供的错误编码给出异常解释，辅助运维人员更好地定位与维护系统。

**技术选型：** LangChain + 阿里百炼嵌入模型（如 text-embedding-v3）+ 向量数据库 RedisStack + 大模型（如 DeepSeek/通义）实现 RAG。

**知识库文档：** 使用《alibaba-java.docx》等作为错误码说明文档，经「加载 → 分割 → 向量化 → 存入 Redis」建索引，用户提问时先检索再生成。

### 3.2 Before：未使用 RAG 的局限

若不使用 RAG，直接问大模型「00000 是什么意思」，模型没有你的私有错误码文档，容易产生歧义或瞎编。因此需要先**从文档中检索**再回答。

### 3.3 After：使用 RAG 的完整流程

【案例源码】`案例与源码-4-LangGraph框架/10-rag/EmbeddingRagLLM.py`

流程概括：  
① 用 **Docx2txtLoader** 加载 `alibaba-java.docx`；  
② 用 **CharacterTextSplitter**（或 RecursiveCharacterTextSplitter）分割；  
③ 用 **DashScopeEmbeddings** 向量化，并写入 **Redis** 向量库；  
④ 用 **as_retriever()** 做相似性检索；  
⑤ 用 **PromptTemplate** 将「检索到的上下文 + 用户问题」拼成提示词；  
⑥ 大模型根据上下文生成答案。

[EmbeddingRagLLM.py](案例与源码-4-LangGraph框架/10-rag/EmbeddingRagLLM.py ":include :type=code")

运行前请确保：Redis 已启动（端口与 `redis_url` 一致）、已配置 `aliQwen-api` 等环境变量，且 `alibaba-java.docx` 放在脚本可访问的路径（如 10-rag 目录下）。

---

**本章小结：**

- **RAG**：**先检索、再生成**。索引阶段：文档加载 → 分割 → 向量化 → 写入向量库；检索阶段：问题向量化 → 相似检索 → 上下文 + 问题 → 大模型生成。可有效减轻幻觉、利用私有文档。
- **LangChain 组件**：文档加载器（如 TextLoader、Docx2txtLoader）统一返回 Document 列表；文本分割器（如 RecursiveCharacterTextSplitter）控制 chunk_size、chunk_overlap；向量存储与检索器与 [第 18 章](18.md) 的 Embedding 配合完成存与查。
- **综合案例**：通过 `EmbeddingRagLLM.py` 串起「加载 docx → 分割 → 向量化存 Redis → 检索 → 提示词模板 → LLM」的完整 RAG 流程，实现基于错误码文档的智能运维问答。

**建议下一步：** 在本地配置好 Redis 与阿里百炼 API Key，跑通 10-rag 下的文档加载、分割与 `EmbeddingRagLLM.py`；再根据业务文档调整分割参数与检索数量 k。若需工具调用、多步推理，可继续学习 [第 17 章 - Tools 工具调用](17-Tools工具调用.md) 或 LangGraph 相关章节。
