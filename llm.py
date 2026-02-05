from langchain_community.chat_models.tongyi import ChatTongyi

from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from config import DASHSCOPE_API_KEY
def get_llm(streaming: bool = True, enable_search: bool = False) -> ChatTongyi:
    """
    获取 LLM 实例
    :param enable_search: 是否开启 Qwen 内置联网
    """
    return ChatTongyi(
        model="qwen-plus", # 或者 qwen-max
        streaming=streaming,
        api_key=DASHSCOPE_API_KEY,
        model_kwargs={
            "enable_search": enable_search
        }
    )
def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )