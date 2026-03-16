"""
【案例】LangChain + Ollama 本地大模型对话

对应教程章节：第 12 章 - Ollama 本地部署与调用 → 5、LangChain 整合 Ollama 调用本地大模型

知识点速览：
用 Python 调用本机运行的 Ollama 大模型（如 qwen:4b），通过 langchain_ollama 的 ChatOllama 发问、拿回答，
无需 API Key，适合本地开发与离线使用。使用前需：① pip install langchain-ollama（及 ollama）；② 本机安装并启动 Ollama，
拉取模型（如 ollama run qwen:4b）；Ollama 默认在 http://localhost:11434 提供服务。
"""

from langchain_ollama import ChatOllama

# ---------- 第一步：创建“聊天客户端” ----------
# ChatOllama 是 LangChain 里专门用来和 Ollama 对话的类，可理解为「连接本地模型的客户端」
# 参数说明：base_url 为 Ollama 服务地址（本机默认 http://localhost:11434）；model 为已拉取的模型名；reasoning 控制是否开启深度思考模式
model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen:4b",
    reasoning=False,
)

# ---------- 第二步：发一条消息并打印回复 ----------
# invoke(问题) 会把你写的问题发给上面的模型，并返回一个“消息对象”
# 这个对象里包含模型生成的文字等内容，直接 print 会显示摘要信息
response = model.invoke("什么是LangChain，100字以内回答")
print(response)

"""
【可选】如果你只想看到模型回复的纯文字，可以这样取：
print(response.content)
"""

