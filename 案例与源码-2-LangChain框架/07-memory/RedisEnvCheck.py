"""
【案例】Redis 环境校验：确认 redis 包已安装且版本符合 LangChain 建议（如 5.3.1）

对应教程章节：第 16 章 - 记忆与对话历史 → 6、案例代码 → 6.2 持久化：Redis 存储 → 环境验证

知识点速览：
- 使用 RedisChatMessageHistory 前需安装 redis 包；LangChain 1.0+ 建议 pip install redis==5.3.1。
- 本脚本仅做导入与版本打印，用于在跑 Redis 对话历史案例前确认环境正常；Redis 服务需单独启动（如 Docker: redis/redis-stack-server，端口 6379 或 26379）。
"""

# 建议：pip install redis==5.3.1

# 尝试导入 redis 包
import redis

# 验证包版本（无报错即为导入成功）
print(redis.__version__)

# 极简 redis 导入测试脚本
try:
    # 导入 redis 包
    import redis

    print("✅ redis 包导入成功！")
    print(f"✅ redis 包版本：{redis.__version__}")
except ModuleNotFoundError:
    print("❌ 未找到 redis 包，请先安装！")
except Exception as e:
    print(f"❌ redis 包导入异常：{e}")

"""
5.3.1
✅ redis 包导入成功！
✅ redis 包版本：5.3.1
"""

"""
【输出示例】
5.3.1
✅ redis 包导入成功！
✅ redis 包版本：5.3.1
"""

