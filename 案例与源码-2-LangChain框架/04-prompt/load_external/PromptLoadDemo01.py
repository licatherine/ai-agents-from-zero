"""
【案例】从 JSON 文件加载提示词模板

对应教程章节：第 13 章 - 提示词与消息模板 → 8、从文件加载提示词（JSON / YAML）

知识点速览：
一、为什么要把提示词放在文件里？
  - 提示词写死在代码里，改一次就要改代码、重新部署；长提示词也会让代码难以阅读。
  - 存成 JSON（或 YAML）后，可以在不改代码的情况下修改提示词，方便版本管理、多人协作和 A/B 测试。

二、load_prompt 做什么？
  - load_prompt("文件路径") 会读取 JSON/YAML 文件，并根据内容创建一个「提示词模板」对象。
  - 得到的对象和用 PromptTemplate(template=..., input_variables=...) 创建的类似，可以调用 .format(变量名=值) 填充占位符。

三、JSON 文件要包含哪些字段？
  - _type：固定写 "prompt"，表示这是 LangChain 的 Prompt 模板。
  - input_variables：列表，列出模板里所有占位符的名字，如 ["name", "what"]。
  - template：字符串，提示词正文，占位符用 {name}、{what} 表示。
  缺一不可，否则 load_prompt 可能报错或行为异常。

四、运行前注意：请在 load_external 目录下执行本脚本，或使用 prompt.json 的完整路径，否则会找不到文件。
"""

# 从 langchain_core 引入 load_prompt，用于从 JSON/YAML 加载模板
from langchain_core.prompts import load_prompt

# 从当前目录（或指定路径）加载 prompt.json，得到与 PromptTemplate 用法相同的模板对象
# encoding="utf-8" 保证中文等字符正常显示
template = load_prompt("prompt.json", encoding="utf-8")

# 用 .format() 填入占位符变量，得到最终字符串（与 1.6 节 PromptTemplate.format 用法一致）
print(template.format(name="张三", what="搞笑的"))
# 输出示例：请张三讲一个搞笑的的故事
