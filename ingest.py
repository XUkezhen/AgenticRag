import os
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rapidocr_onnxruntime import RapidOCR  # 如果你要支持图片/扫描件

from llm import get_embeddings
from config import CHROMA_DIR


# --- 辅助函数：提取图片文字 ---
def extract_text_from_image(img_path):
    try:
        engine = RapidOCR()
        result, _ = engine(img_path)
        if not result: return ""
        return "\n".join([line[1] for line in result])
    except Exception as e:
        print(f"[OCR Error] {e}")
        return ""


async def ingest_file(file_path: str, session_id: str):
    """
    处理单个文件上传，带进度条和分批写入
    """
    print(f"--- [开始处理] {os.path.basename(file_path)} (Session: {session_id}) ---")

    # 1. 加载文档
    docs = []
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(file_path)
            docs = [Document(page_content=text, metadata={"source": file_path})]
        else:
            print(f"[WARN] 不支持的文件格式: {file_path}")
            return
    except Exception as e:
        print(f"[Load Error] 加载文件失败: {e}")
        return

    if not docs:
        print("[Warn] 文件内容为空或解析失败")
        return

    print(f"-> 文档加载完成，共 {len(docs)} 页/部分，正在切分...")

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 稍微改小一点，提高检索精度
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 3. 注入 Metadata
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
        chunk.metadata["source"] = os.path.basename(file_path)

    total_chunks = len(chunks)
    print(f"-> 切分完成，共生成 {total_chunks} 个切片。准备开始 Embedding 入库...")

    # 4. 初始化向量库
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="chat_docs"
    )

    # 5. 🔥 核心优化：分批写入 + 进度打印
    # 每次处理 50 个切片，避免 API 超时或数据库卡死
    BATCH_SIZE = 50

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]

        try:
            # 写入 Chroma (这一步会调用阿里云 API)
            vectorstore.add_documents(batch)

            # 计算进度
            progress = min(i + BATCH_SIZE, total_chunks)
            percent = (progress / total_chunks) * 100
            print(f"   [写入中] {progress}/{total_chunks} ({percent:.1f}%) ...")

            # 💡 稍微休息一下，防止触发阿里云 API 的 QPS 限制（每秒请求过多会被封）
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ [Error] 批次 {i} 写入失败: {e}")
            # 可以选择 continue 跳过错误批次，或者 break
            continue

    print(f"✅ [完成] 文件 {os.path.basename(file_path)} 全部处理完毕！")