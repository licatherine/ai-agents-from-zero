# 环境检查脚本：用于确认当前 Python 环境中 LangChain 相关包的版本与安装路径，
# 便于排查「装错版本」或「虚拟环境不一致」等问题。无需 API Key，可直接运行。

import langchain          # 核心框架（Chain、Agent、Memory 等）
import langchain_community  # 社区扩展（部分集成、第三方工具等）
import sys                # 获取 Python 解释器信息

# LangChain 核心包版本号（需与教程/文档要求的版本区间一致）
print("langchainVersion:  " + langchain.__version__)
# 社区扩展包版本号
print("langchain_communityVersion:  " + langchain_community.__version__)
# LangChain 实际安装路径（可确认是否来自当前虚拟环境）
print("langchainfile:" + langchain.__file__)

# 当前 Python 解释器版本（如 3.10.x），用于确认运行环境
print(sys.version)

# 【输出示例】
# langchainVersion:  1.2.9
# langchain_communityVersion:  0.4.1
# langchainfile:/Users/tools/PyCharmMiscProject/python100/.venv/lib/python3.10/site-packages/langchain/__init__.py
# 3.10.19 (main, Oct  9 2025, 15:25:03) [Clang 17.0.0 (clang-1700.6.3.2)]