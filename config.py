import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 向量数据库路径
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
# 上传文件存储路径
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
# 记忆数据库路径 (SQLite)
MEMORY_DB_PATH = os.path.join(DATA_DIR, "memory.sqlite")

# 确保目录存在
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)