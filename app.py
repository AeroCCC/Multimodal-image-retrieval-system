"""
多模态智能问答系统 - Web 界面
基于 Streamlit 构建
"""

import os
import cv2
import json
import tempfile
import streamlit as st
from datetime import datetime

# 页面配置
st.set_page_config(page_title="多模态智能问答系统", page_icon="🤖", layout="wide")

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

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化 Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "current_api_key" not in st.session_state:
    st.session_state.current_api_key = DASHSCOPE_API_KEY

if "question" not in st.session_state:
    st.session_state.question = "这张图片里有什么车？"


def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def process_image_streamlit(uploaded_file, question, api_key):
    """
    Streamlit 版本的图片处理流程
    """
    # 保存上传的图片
    temp_image_path = save_uploaded_file(uploaded_file)

    # 设置 API Key
    import dashscope

    dashscope.api_key = api_key

    try:
        # 1. YOLO 检测
        detections = detect_objects(temp_image_path)

        # 2. 生成标注图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"web_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        annotated_path = visualize_detections(temp_image_path, detections, output_path)

        # 3. 调用通义千问
        answer = ask_qwen(question, detections, uploaded_file.name)

        # 保存到历史记录
        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": uploaded_file.name,
            "question": question,
            "answer": answer,
            "detections": detections,
            "annotated_image": annotated_path,
        }
        st.session_state.history.insert(0, history_item)

        # 限制历史记录数量
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]

        return {
            "detections": detections,
            "answer": answer,
            "annotated_image_path": annotated_path,
        }

    finally:
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ==================== 页面内容 ====================

# 侧边栏 - 设置
with st.sidebar:
    st.title("⚙️ 设置")

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

    st.divider()

    # 清空历史
    if st.button("🗑️ 清空历史记录"):
        st.session_state.history = []
        st.rerun()

    # 历史记录统计
    st.metric("历史记录", len(st.session_state.history))

# 主界面
st.title("🤖 多模态智能问答系统")
st.markdown("上传图片，描述你看到的内容，我来帮你分析！")

# 布局：左侧输入，右侧结果
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "支持 JPG、PNG 格式", type=["jpg", "jpeg", "png"], help="上传你要分析的图片"
    )

    st.subheader("💬 提问")
    st.text_area(
        "描述你的问题",
        key="question",
        height=80,
        help="描述你想知道的内容",
    )

    # 快速问题按钮
    st.markdown("**快速问题：**")
    quick_questions = [
        "这张图片里有什么？",
        "有多少辆车？",
        "有几个人？",
        "这是什么地方？",
    ]

    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        if cols[i % 2].button(q, key=f"quick_{i}"):
            st.session_state.question = q
            st.rerun()

    # 分析按钮
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

# 处理逻辑
if analyze_btn and uploaded_file and st.session_state.question:
    if not st.session_state.current_api_key:
        st.error("请先配置 API Key")
    else:
        with st.spinner("🔍 正在分析..."):
            try:
                result = process_image_streamlit(
                    uploaded_file, st.session_state.question, st.session_state.current_api_key
                )
                st.toast("✅ 分析完成！", icon="🎉")

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

                    # LLM 回答
                    st.subheader("💡 智能回答")
                    st.info(result["answer"])

                    # 显示详情
                    with st.expander("查看详细信息"):
                        st.json(result["detections"], expanded=False)

            except Exception as e:
                st.error(f"分析失败: {str(e)}")

elif analyze_btn:
    if not uploaded_file:
        st.warning("请先上传图片")
    if not st.session_state.question:
        st.warning("请输入问题")

# 如果没有分析结果，在右侧显示提示
if not (analyze_btn and uploaded_file and st.session_state.question):
    with col2:
        st.info("💡 请在左侧上传图片并提问，分析结果将显示在这里。")
        st.image("https://via.placeholder.com/600x400.png?text=Waiting+for+Analysis", use_container_width=True)

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
    "🤖 多模态智能问答系统 | 基于 YOLO + 通义千问"
    "</div>",
    unsafe_allow_html=True,
)
