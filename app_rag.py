"""
多模态 RAG 知识库系统 - Web 界面
基于 Streamlit 构建，结合 ChromaDB 知识库实现双重检索
"""

import os
import cv2
import json
import tempfile
import streamlit as st
from datetime import datetime

# 页面配置
st.set_page_config(page_title="多模态 RAG 知识库系统", page_icon="🧠", layout="wide")

# 导入核心功能
from multimodal_rag_qa import (
    detect_objects,
    visualize_detections,
    format_detection_summary,
    ask_qwen,
    DASHSCOPE_API_KEY,
    YOLO_MODEL_PATH,
    OUTPUT_DIR,
)

# 导入知识库模块
from knowledge_base import (
    init_knowledge_base,
    add_to_knowledge_base,
    search_by_vector,
    search_by_text,
    search_hybrid,
    get_knowledge_base_stats,
    get_image_embedding,
)

# 配置
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化 Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "current_api_key" not in st.session_state:
    st.session_state.current_api_key = DASHSCOPE_API_KEY

if "kb_initialized" not in st.session_state:
    st.session_state.kb_initialized = False


def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def process_image_rag(uploaded_file, question, api_key, use_knowledge_base=True):
    """
    RAG 流程：YOLO检测 + 知识库检索 + LLM生成

    参数:
        uploaded_file: 上传的文件
        question: 用户问题
        api_key: API Key
        use_knowledge_base: 是否使用知识库

    返回:
        dict: 处理结果
    """
    # 保存上传的图片
    temp_image_path = save_uploaded_file(uploaded_file)

    # 设置 API Key
    import dashscope

    dashscope.api_key = api_key

    try:
        # ========== 1. 实时 YOLO 检测 ==========
        realtime_detections = detect_objects(temp_image_path)

        # 生成标注图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"rag_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        annotated_path = visualize_detections(
            temp_image_path, realtime_detections, output_path
        )

        # ========== 2. 知识库检索 ==========
        kb_results = {"vector_search": None, "text_search": None, "combined": []}

        if use_knowledge_base:
            # 2.1 向量相似搜索
            kb_results["vector_search"] = search_by_vector(temp_image_path, top_k=3)

            # 2.2 文本语义搜索
            kb_results["text_search"] = search_by_text(question, top_k=3)

            # 2.3 融合结果
            kb_results["combined"] = []

            if kb_results["vector_search"] and kb_results["vector_search"].get("ids"):
                for img_id, meta, dist in zip(
                    kb_results["vector_search"]["ids"][0],
                    kb_results["vector_search"]["metadatas"][0],
                    kb_results["vector_search"]["distances"][0],
                ):
                    kb_results["combined"].append(
                        {
                            "id": img_id,
                            "metadata": meta,
                            "distance": dist,
                            "type": "vector",
                        }
                    )

            if kb_results["text_search"] and kb_results["text_search"].get("ids"):
                for img_id, meta, dist in zip(
                    kb_results["text_search"]["ids"][0],
                    kb_results["text_search"]["metadatas"][0],
                    kb_results["text_search"]["distances"][0],
                ):
                    existing_ids = [r["id"] for r in kb_results["combined"]]
                    if img_id not in existing_ids:
                        kb_results["combined"].append(
                            {
                                "id": img_id,
                                "metadata": meta,
                                "distance": dist,
                                "type": "text",
                            }
                        )

        # ========== 3. 构建上下文 ==========
        # 实时检测结果
        context_parts = []
        context_parts.append("【实时检测结果】")
        context_parts.append(format_detection_summary(realtime_detections))

        # 知识库结果
        if use_knowledge_base and kb_results["combined"]:
            context_parts.append("\n【知识库相关图片】")
            context_parts.append(f"找到 {len(kb_results['combined'])} 张相关图片:")

            for i, result in enumerate(kb_results["combined"][:3], 1):
                meta = result["metadata"]
                context_parts.append(f"\n相关图片 {i}:")
                context_parts.append(f"  - 路径: {meta.get('image_path', 'N/A')}")
                context_parts.append(f"  - 包含: {meta.get('classes', 'N/A')}")
                context_parts.append(f"  - 匹配方式: {result['type']}")

        full_context = "\n".join(context_parts)

        # ========== 4. 调用 LLM ==========
        prompt = f"""你是一个图像分析助手。

{full_context}

用户问题：{question}

请根据实时检测结果和知识库信息回答用户的问题。如果知识库中有相关信息，请参考。回答要友好自然。"""

        answer = ask_qwen_with_context(prompt)

        # 保存到历史记录
        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": uploaded_file.name,
            "question": question,
            "answer": answer,
            "detections": realtime_detections,
            "kb_results": kb_results["combined"],
            "annotated_image": annotated_path,
        }
        st.session_state.history.insert(0, history_item)

        # 限制历史记录数量
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]

        return {
            "detections": realtime_detections,
            "answer": answer,
            "annotated_image_path": annotated_path,
            "kb_results": kb_results,
            "context": full_context,
        }

    finally:
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def ask_qwen_with_context(prompt):
    """
    调用通义千问 API（自定义 prompt 版本）
    """
    import dashscope

    try:
        response = dashscope.Generation.call(
            model="qwen-max", prompt=prompt, result_format="message"
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"API 调用失败: {response.code} - {response.message}"

    except Exception as e:
        return f"调用出错: {str(e)}"


# ==================== 页面内容 ====================

# 侧边栏 - 设置
with st.sidebar:
    st.title("🧠 知识库 RAG 系统")

    # 初始化知识库
    if not st.session_state.kb_initialized:
        try:
            init_knowledge_base()
            st.session_state.kb_initialized = True
        except Exception as e:
            st.warning(f"知识库未初始化: {e}")

    # 知识库统计
    st.divider()
    st.subheader("📚 知识库状态")
    try:
        kb_stats = get_knowledge_base_stats()
        st.metric("图片数量", kb_stats["total_images"])
        if kb_stats["classes"]:
            st.write("**物体类别:**")
            for cls, count in sorted(kb_stats["classes"].items(), key=lambda x: -x[1])[
                :5
            ]:
                st.write(f"  - {cls}: {count}")
    except Exception as e:
        st.info("知识库为空")

    st.divider()

    # API Key 配置
    api_key_input = st.text_input(
        "通义千问 API Key",
        type="password",
        value=st.session_state.current_api_key
        if st.session_state.current_api_key != "your-api-key-here"
        else "",
        help="可在阿里云 dashscope 控制台获取",
    )

    if api_key_input:
        st.session_state.current_api_key = api_key_input

    # RAG 开关
    use_kb = st.toggle("启用知识库检索", value=True)
    if use_kb:
        st.caption("将同时检索 ChromaDB 知识库中的历史图片")

    st.divider()

    # 清空历史
    if st.button("🗑️ 清空历史记录"):
        st.session_state.history = []
        st.rerun()

# 主界面
st.title("🧠 多模态 RAG 知识库系统")
st.markdown("结合 **YOLO 实时检测** + **ChromaDB 知识库** 的智能问答系统")

# 说明
with st.expander("ℹ️ 关于 RAG 知识库系统"):
    st.markdown("""
    ### 工作流程
    
    ```
    用户上传图片 → YOLO实时检测 → ChromaDB知识库检索 → 融合上下文 → 通义千问回答
    ```
    
    ### 知识库优势
    
    - **双重检索**: 向量相似搜索 + 文本语义搜索
    - **历史参考**: 参考知识库中相似图片的信息
    - **持续学习**: 可不断向知识库添加新图片
    """)

# 布局：左侧输入，右侧结果
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "支持 JPG、PNG 格式", type=["jpg", "jpeg", "png"], help="上传你要分析的图片"
    )

    st.subheader("💬 提问")
    question = st.text_area(
        "描述你的问题", value="这张图片里有什么？", height=80, help="描述你想知道的内容"
    )

    # 分析按钮
    analyze_btn = st.button(
        "🚀 开始 RAG 分析", type="primary", use_container_width=True
    )

# 处理逻辑
if analyze_btn and uploaded_file and question:
    if not st.session_state.current_api_key:
        st.error("请先配置 API Key")
    else:
        with st.spinner("🔍 RAG 检索中..."):
            try:
                result = process_image_rag(
                    uploaded_file, question, st.session_state.current_api_key, use_kb
                )

                with col2:
                    st.subheader("📊 分析结果")

                    # 显示标注图片
                    st.image(
                        result["annotated_image_path"],
                        caption="检测结果",
                        use_container_width=True,
                    )

                    # 检测统计
                    detections = result["detections"]
                    if detections:
                        class_count = {}
                        for det in detections:
                            cls = det["class"]
                            class_count[cls] = class_count.get(cls, 0) + 1

                        st.write("**检测到的物体：**")
                        for cls, count in sorted(class_count.items()):
                            st.write(f"- {cls}: {count}个")

                    # 知识库检索结果
                    if use_kb and result["kb_results"]["combined"]:
                        st.subheader("📚 知识库检索结果")
                        st.write(
                            f"找到 {len(result['kb_results']['combined'])} 张相关图片:"
                        )

                        for i, kb_item in enumerate(
                            result["kb_results"]["combined"][:3], 1
                        ):
                            meta = kb_item["metadata"]
                            with st.expander(f"相关图片 {i} ({kb_item['type']})"):
                                st.write(f"**路径:** {meta.get('image_path', 'N/A')}")
                                st.write(f"**包含:** {meta.get('classes', 'N/A')}")
                                st.write(f"**距离:** {kb_item['distance']:.4f}")

                    # LLM 回答
                    st.subheader("💡 RAG 智能回答")
                    st.markdown(f"```\n{result['answer']}\n```")

                    # 显示上下文（调试用）
                    with st.expander("🔧 查看检索上下文"):
                        st.text(result["context"])

            except Exception as e:
                st.error(f"分析失败: {str(e)}")

elif analyze_btn:
    if not uploaded_file:
        st.warning("请先上传图片")
    if not question:
        st.warning("请输入问题")

# 历史记录展示
st.divider()
st.subheader("📜 历史记录")

if st.session_state.history:
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"{item['timestamp']} - {item['image_name']}"):
            col_h1, col_h2 = st.columns([1, 1])

            with col_h1:
                st.markdown(f"**问题：** {item['question']}")
                st.markdown(f"**回答：** {item['answer']}")
                if item.get("kb_results"):
                    st.caption(f"知识库检索: {len(item['kb_results'])} 条")

            with col_h2:
                if os.path.exists(item["annotated_image"]):
                    st.image(
                        item["annotated_image"],
                        caption="检测结果",
                        use_container_width=True,
                    )
else:
    st.info("暂无历史记录")

# 页脚
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🧠 多模态 RAG 知识库系统 | 基于 YOLO + ChromaDB + 通义千问"
    "</div>",
    unsafe_allow_html=True,
)
