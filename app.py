import os
import uuid
import json
import shutil
import aiosqlite
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import UPLOAD_DIR, MEMORY_DB_PATH
from graph import graph_builder
from ingest import ingest_file

# --- 会话元数据管理 (简单的 JSON 文件存储名字) ---
META_FILE = "data/sessions.json"


def load_meta():
    if not os.path.exists(META_FILE): return {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(data):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：连接数据库
    async with aiosqlite.connect(MEMORY_DB_PATH) as conn:
        # 初始化 Checkpointer
        checkpointer = AsyncSqliteSaver(conn)
        # 初始化表结构 (LangGraph 需要)
        await checkpointer.setup()
        # 编译 Graph
        app.state.graph = graph_builder.compile(checkpointer=checkpointer)
        yield
    # 关闭时自动断开连接


app = FastAPI(title="AI Chat Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 数据模型 ---
class RenameReq(BaseModel):
    name: str


class ChatReq(BaseModel):
    query: str
    session_id: str
    use_web: bool = False


def sse_pack(event_type: str, data: str):
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


# --- 接口 ---

@app.get("/sessions")
async def list_sessions():
    """获取所有会话列表"""
    meta = load_meta()
    # 转换为列表返回，按时间倒序（这里简单返回）
    sessions = [{"id": k, "name": v.get("name", "未命名会话")} for k, v in meta.items()]
    return {"sessions": sessions}


@app.post("/session/new")
async def new_session():
    """创建新会话"""
    sid = str(uuid.uuid4())
    meta = load_meta()
    meta[sid] = {"name": "新对话 " + sid[:4]}
    save_meta(meta)
    return {"session_id": sid, "name": meta[sid]["name"]}


@app.post("/session/{session_id}/rename")
async def rename_session(session_id: str, req: RenameReq):
    """重命名会话"""
    meta = load_meta()
    if session_id not in meta:
        meta[session_id] = {}  # 容错
    meta[session_id]["name"] = req.name
    save_meta(meta)
    return {"status": "ok", "name": req.name}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """获取指定会话的历史消息"""
    # 从 Graph 的 state 中读取 memory
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await app.state.graph.aget_state(config)

    if not snapshot.values:
        return {"messages": []}

    # 格式化消息给前端
    formatted_msgs = []
    for msg in snapshot.values["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "ai"
        # 过滤掉 SystemMessage
        if role == "user" or isinstance(msg, AIMessage):
            formatted_msgs.append({
                "role": role,
                "content": msg.content
            })
    return {"messages": formatted_msgs}


@app.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        session_id: str = Form(...)
):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(ingest_file, file_path, session_id)
    return {"message": "File uploaded", "filename": file.filename}


@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    async def event_generator():
        config = {
            "configurable": {
                "thread_id": req.session_id,
                "use_web": req.use_web
            }
        }
        inputs = {"messages": [HumanMessage(content=req.query)]}

        # 使用 app.state.graph (已绑定 DB)
        async for event in app.state.graph.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            if kind == "on_chain_start" and event["name"] == "plan":
                yield sse_pack("status", "正在规划问题...")
            elif kind == "on_chain_start" and event["name"] == "retrieve":
                yield sse_pack("status", "正在检索知识库...")
            elif kind == "on_chain_start" and event["name"] == "critique":
                yield sse_pack("status", "正在自检与补强...")
            elif kind == "on_chain_start" and event["name"] == "generate":
                status = "正在联网搜索..." if req.use_web else "正在思考..."
                yield sse_pack("status", status)
            elif kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield sse_pack("token", content)
        yield sse_pack("done", "[DONE]")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
