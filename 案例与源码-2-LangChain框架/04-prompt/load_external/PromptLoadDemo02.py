"""
【案例】从 YAML 文件加载提示词模板

对应教程章节：第 13 章 - 提示词与消息模板 → 8、从文件加载提示词（JSON / YAML）

知识点速览：
一、和 JSON 有什么不同？
  - 用法完全一样：都是 load_prompt("文件路径", encoding="utf-8")，得到的对象同样用 .format(...) 填参。
  - 区别只在文件格式：YAML 更易读、支持注释，JSON 更通用、便于程序生成。按团队习惯选一种即可。

二、YAML 文件要包含哪些字段？
  - 与 JSON 一致：_type（"prompt"）、input_variables（占位符名列表）、template（提示词字符串）。
  - 写法示例见同目录下的 prompt.yaml。

三、运行前注意：请在 load_external 目录下执行本脚本，或使用 prompt.yaml 的完整路径。
"""

import warnings

warnings.filterwarnings(
    "ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14"
)

# 从 YAML 加载提示词模板，API 与 load_prompt("prompt.json") 一致
from langchain_core.prompts import load_prompt

template = load_prompt("prompt.yaml", encoding="utf-8")
print(template.format(name="年轻人", what="滑稽"))
# 输出示例：请年轻人讲一个滑稽的故事
