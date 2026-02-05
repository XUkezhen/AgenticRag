import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
# 保持使用异步 SQLite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages

from llm import get_llm, get_embeddings
from config import CHROMA_DIR

# ✅ 新增：一个简单粗暴但好用的 Token 计数器
def count_tokens_simple(messages: List[BaseMessage]) -> int:
    # 粗略估算：把所有消息的字数加起来
    # Qwen 的 Token 效率很高，通常 1 个汉字 < 1 Token
    # 这里我们按 1 汉字 = 1 Token 算，留足了安全余量
    total_text = "".join([m.content for m in messages])
    return len(total_text)

# 1. 定义状态
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str
    # Multi-role pipeline state (kept out of chat history)
    plan: str
    critique: str


# 2. Plan node (question decomposition / answer outline)
async def plan_node(state: State, config: RunnableConfig):
    llm = get_llm(streaming=False, enable_search=False)

    last_msg = state["messages"][-1]
    query = last_msg.content

    planner_prompt = [
        SystemMessage(
            content=(
                "你是 PlannerAgent（规划智能体）。请为回答用户问题生成一个简短可执行的计划。\n"
                "输出（纯文本即可）：\n"
                "1) 回答大纲\n"
                "2) 检索提示（关键词/实体/可能的限定条件）\n"
                "3) 若需要澄清，请给出 1-3 个追问问题"
            )
        ),
        HumanMessage(content=query),
    ]

    resp = await llm.ainvoke(planner_prompt)
    return {"plan": resp.content}


# 2. 检索节点 (长期记忆提取)
def retrieve_node(state: State, config: RunnableConfig):
    session_id = config["configurable"].get("thread_id")
    # 获取用户最新的一条消息
    last_msg = state["messages"][-1]
    query = last_msg.content

    embeddings = get_embeddings()
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="chat_docs"
        )
        # --- 上下文工程步骤 A：精准检索 ---
        # 只取最相关的 4 个片段，避免信息过载
        results = vectorstore.similarity_search(
            query, k=4, filter={"session_id": session_id}
        )
        context_text = "\n\n".join([
            f"--- 资料 (来源: {d.metadata.get('source', 'unknown')}) ---\n{d.page_content}"
            for d in results
        ])
    except Exception:
        context_text = ""
    return {"context": context_text}


# 3. Critique node (self-check / missing evidence)
async def critique_node(state: State, config: RunnableConfig):
    llm = get_llm(streaming=False, enable_search=False)

    last_msg = state["messages"][-1]
    query = last_msg.content
    plan = state.get("plan", "")
    context = state.get("context", "")

    critic_prompt = [
        SystemMessage(
            content=(
                "你是 CriticAgent（审校智能体）。请检查：计划 + 检索到的资料是否足以支撑回答。\n"
                "请输出：\n"
                "- 缺口/缺失证据（要点列表）\n"
                "- 可能的幻觉风险/易错点\n"
                "- 是否需要向用户追问澄清（如需要，给出 1-3 个问题）"
            )
        ),
        HumanMessage(content=f"用户问题：\n{query}\n\n计划：\n{plan}\n\n检索到的资料：\n{context}"),
    ]

    resp = await llm.ainvoke(critic_prompt)
    return {"critique": resp.content}


# 4. 生成节点 (核心上下文工程发生地)
async def generate_node(state: State, config: RunnableConfig):
    # 获取配置
    use_web = config["configurable"].get("use_web", False)
    context = state.get("context", "")
    plan = state.get("plan", "")
    critique = state.get("critique", "")

    # 获取 LLM
    llm = get_llm(streaming=True, enable_search=use_web)

    # --- 上下文工程步骤 B：历史记录裁剪 (Sliding Window) ---
    # 策略：即使数据库存了1000条，我们只把最近的 Max Tokens 发给模型
    # Qwen-Plus 窗口很大，但为了省钱和速度，我们限制在 20k token (约 1.5万汉字)
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=20000,
        strategy="last",  # 保留“最后”的对话
        token_counter=count_tokens_simple,  # 使用 Qwen 自己的 token 计算器
        include_system=False,  # 系统提示词单独加，不参与裁剪
        start_on="human"  # 保证对话是以“人”开始的，防止截断导致逻辑错乱
    )

    # --- 上下文工程步骤 C：提示词组装 (Prompt Engineering) ---
    # 清晰地定义 AI 的角色、能力边界和参考资料
    system_prompt_text = (
        "你是一个专业的 AI 智能助手。"
        "\n请遵循以下原则："
        "\n1. 优先回答用户的具体问题。"
        "\n2. 如果提供了【参考资料】，请优先基于资料回答，并告知用户依据。"
    )

    if use_web:
        system_prompt_text += "\n3. (当前已开启联网模式) 如果资料库信息不足，请利用你的联网能力搜索最新信息。"
    else:
        system_prompt_text += "\n3. (当前未开启联网) 请基于你的训练知识库或上传的文件回答。"

    if context:
        system_prompt_text += f"\n\n### 📎 参考资料 (RAG检索结果) ###\n{context}\n################################"

    if plan:
        system_prompt_text += f"\n\n### PlannerAgent Plan ###\n{plan}\n########################"

    if critique:
        system_prompt_text += f"\n\n### CriticAgent Notes ###\n{critique}\n########################"

    # 最终组合：系统提示 + 裁剪后的历史
    final_messages = [SystemMessage(content=system_prompt_text)] + trimmed_history

    # 调用模型
    response = await llm.ainvoke(final_messages)

    # 返回结果 (注意：这里返回的 response 会被自动存入 SQLite 长期保存)
    return {"messages": [response]}


# 5. 构建图
workflow = StateGraph(State)
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("critique", critique_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "critique")
workflow.add_edge("critique", "generate")
workflow.add_edge("generate", END)

# 导出 builder
graph_builder = workflow
